//! Live translation as a conversation: microphone → denoise → resample → voice activity → Whisper.
//!
//! Every stage exists because skipping it breaks the transcript:
//!
//! 1. **Capture** — rodio's microphone, asked for 16 kHz mono (hardware usually gives 44.1/48 kHz).
//! 2. **Denoise** — `voice-engine`'s `NoiseReducer` (RNNoise) runs on the device-rate audio.
//! 3. **Resample** — rubato 5.0 resamples to the 16 kHz Whisper requires, in real time. A naive
//!    linear resample folds everything above 8 kHz back into the audio, which is exactly what makes
//!    a transcript fall apart.
//! 4. **Segment** — `voice-engine`'s `VadProcessor` (a Silero port plus the padding logic around
//!    it) decides where a turn starts and ends. This is the stage that keeps the output clean:
//!    Whisper *invents* text when it is handed silence, and a pause of quiet is what closes a turn.
//! 5. **Transcribe + translate** — `whisper-stt` with `translate: true`, on a worker thread so
//!    decoding never blocks capture.
//!
//! ```text
//! cargo run --example live_translate                       # auto-detect language -> English
//! cargo run --example live_translate -- --language zh      # skip auto-detection
//! cargo run --example live_translate -- --model tiny       # quick smoke test
//! cargo run --example live_translate -- --model large-v3   # turbo cannot translate, see below
//! cargo run --example live_translate -- --no-translate     # transcribe, do not translate
//! cargo run --example live_translate -- --source           # show what was said, not just the result
//! cargo run --example live_translate -- --denoise          # also run RNNoise
//! cargo run --example live_translate -- --pause 3000       # wait 3s of quiet before translating
//! cargo run --example live_translate -- --out speech.wav   # keep the speech that was sent
//! ```
//!
//! Two things worth knowing before you run it:
//!
//! - **Turbo checkpoints cannot translate.** They are trained on transcription data only, and
//!   return the source language even when the translate task is requested. Use `large-v3` (or
//!   medium/small/base/tiny) if you want English out of non-English speech.
//! - **Denoising is opt-in.** RNNoise is trained on 48 kHz wideband audio, and on this crate's own
//!   test clip it preserved the level but cut a 2.8 s utterance down to 0.4 s and turned a correct
//!   transcript into nonsense. Pass `--denoise` to enable it and compare on your own microphone.
//!
//! Press **Enter** to start and **Enter** (or Ctrl+C) to stop. Say a full sentence, pause, and the
//! turn comes back translated. This is not low latency: Whisper decodes each turn after you finish
//! it, so expect a pause of a second or two, longer for the bigger models.
//!
//! `--file` swaps the microphone for a media file and runs it through the identical chain, which is
//! how you check the conditioning stages without having to talk:
//!
//! ```text
//! cargo run --example live_translate -- --file recordings/rodio-1790179654.wav
//! ```

use std::num::NonZero;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use rodio::microphone::{MicrophoneBuilder, available_inputs};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use tokio::sync::broadcast;
use voice_engine::event::SessionEvent;
use voice_engine::media::denoiser::NoiseReducer;
use voice_engine::media::processor::Processor;
use voice_engine::media::vad::{TinySilero, VADOption, VadProcessor};
use voice_engine::media::{AudioFrame, Samples};
use whisper_stt::audio::{decode_file, mix_down};
use whisper_stt::{
    ModelStore, Transcriber, TranscriptionOptions, TranscriptionSession, WHISPER_SAMPLE_RATE,
};

#[path = "shared/mod.rs"]
mod shared;
use shared::{Stop, Stopper, write_wav};

/// Asked for first; Whisper works on 16 kHz mono.
const PREFERRED_SAMPLE_RATE: NonZero<u32> = NonZero::new(16_000).expect("non-zero");
const PREFERRED_CHANNELS: NonZero<u16> = NonZero::new(1).expect("non-zero");

/// Frames rubato consumes per call, measured at the device rate — about 11 ms at 44.1 kHz.
const RESAMPLE_CHUNK: usize = 512;
/// Frames per voice-activity decision: 32 ms at 16 kHz, and Silero's own hop size.
const VAD_WINDOW: usize = 512;
/// Speech probability above which a window counts as voice.
const VAD_THRESHOLD: f32 = 0.5;
/// Default quiet time that closes a turn, in milliseconds.
const DEFAULT_PAUSE_MS: u64 = 2_000;
/// A turn shorter than this is a cough or a click, not a sentence.
const MIN_SPEECH_MS: u64 = 500;
/// Audio kept before speech onset so the first word is not clipped.
///
/// Silero is an LSTM: after a long silence it needs a window or two to ramp up, so the frame it
/// first calls speech can be several hundred milliseconds into the sentence. Keeping a third of a
/// second of lead-in is what gets the opening syllable back.
const PRE_ROLL_MS: usize = 350;
/// Quiet audio appended after a turn so the last word is not cut off mid-syllable.
const TAIL_PAD_MS: usize = 300;
/// Fraction of a turn that has to look like speech before it is worth decoding.
///
/// The voice detector can open a turn on a door slam and then hear nothing until the pause closes
/// it. That turn is almost entirely silence, and Whisper will invent a confident sentence for it.
const MIN_SPEECH_RATIO: f32 = 0.4;
/// Longest turn the segmenter is allowed to hold on to.
///
/// Somebody who never pauses would otherwise produce one endless turn that never reaches the model.
/// The ceiling is 25s rather than Whisper's own 30s window because `voice-engine` keeps a ring of
/// the last 1000 detector windows — 32s of audio — and silently drops everything older than 5s once
/// it overflows. Staying under 1000 windows keeps the whole turn intact.
const MAX_UTTERANCE_SECS: u64 = 25;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|arg| arg == "--list" || arg == "-l") {
        return list_inputs();
    }

    let model = flag(&args, "--model")
        .unwrap_or("large-v3-turbo")
        .to_string();
    let language = flag(&args, "--language").map(str::to_string);
    let translate = !args.iter().any(|arg| arg == "--no-translate");
    // A second decode pass, to show what was said alongside what it means.
    let show_source = args.iter().any(|arg| arg == "--source");
    // Opt-in: see the note in `run_session`. RNNoise runs at the device rate.
    let denoise = args.iter().any(|arg| arg == "--denoise");
    let file = flag(&args, "--file").map(PathBuf::from);
    let out = flag(&args, "--out").map(PathBuf::from);
    let pause_ms = match flag(&args, "--pause") {
        Some(value) => value
            .parse::<u64>()
            .map_err(|_| anyhow!("--pause expects milliseconds, got {value:?}"))?,
        None => DEFAULT_PAUSE_MS,
    };

    // Whisper needs a model before it can do anything; this downloads on first run.
    let store = ModelStore::from_name(&model, "models")
        .map_err(|err| anyhow!("unknown model {model:?}: {err}"))?;
    println!("Model: {model} — {}", store.path().display());
    // OpenAI trained the turbo checkpoints on transcription data only: they happily accept the
    // translate task and then return the original language, which looks like a broken pipeline.
    if translate && model.contains("turbo") {
        eprintln!(
            "Note: {model} cannot translate — turbo checkpoints are transcription-only and return \
             the source language even when translation is requested. Use --model large-v3 (or \
             medium/small/base/tiny) for real translation, or --no-translate to drop the warning."
        );
    }
    if !store.exists() {
        println!("Downloading {model} (first run only)...");
    }
    let runtime = tokio::runtime::Runtime::new()?;
    runtime
        .block_on(store.ensure())
        .with_context(|| format!("failed to prepare the {model} model"))?;
    let model_path = store.path();

    let stopper = Stopper::arm()?;
    let stop_flag = Arc::new(AtomicBool::new(false));

    let source = match file {
        Some(path) => open_file(&path)?,
        None => open_microphone(&stopper, &stop_flag)?,
    };
    let Source {
        sample_rate: device_rate,
        channels,
        samples: sample_rx,
        producer: capture,
    } = source;

    // Decoding lives here: the model is loaded once, then fed one turn at a time.
    let (work_tx, work_rx) = channel::<Turn>();
    let worker = thread::spawn(move || {
        // One session for the whole conversation: the state holds the KV cache and the mel and
        // encoder buffers, and reallocating them per turn is the most expensive thing this loop
        // could do. `TranscriptionOptions::no_context` below keeps Whisper from carrying the
        // previous turn's text into the next one now that the state survives between turns.
        let transcriber = match Transcriber::new(&model_path) {
            Ok(transcriber) => transcriber,
            Err(err) => {
                eprintln!("failed to load the model: {err}");
                return;
            }
        };
        let mut session = match transcriber.session() {
            Ok(session) => session,
            Err(err) => {
                eprintln!("failed to allocate a whisper state: {err}");
                return;
            }
        };
        for turn in work_rx {
            if let Err(err) = translate_turn(
                &mut session,
                &turn,
                translate,
                show_source,
                language.as_deref(),
            ) {
                eprintln!("  ! transcription failed: {err}");
            }
        }
    });

    let session = run_session(
        sample_rx,
        &stopper,
        &stop_flag,
        &work_tx,
        Conditioning {
            device_rate,
            channels,
            denoise,
        },
        pause_ms,
        out,
    )?;

    // Dropping the sender ends the worker's loop.
    drop(work_tx);
    let _ = capture.join();
    let _ = worker.join();

    println!(
        "\nStopped. {:.1}s captured, {} turn(s) sent to the model, {} dropped as not speech.",
        session.captured_secs, session.turns, session.dropped
    );
    Ok(())
}

/// Decodes one turn and prints it as a line of dialogue.
fn translate_turn(
    session: &mut TranscriptionSession<'_>,
    turn: &Turn,
    translate: bool,
    show_source: bool,
    language: Option<&str>,
) -> Result<()> {
    let started = Instant::now();

    // `--source` costs a second decode pass, which is why it is not the default: on a large model
    // that is most of the delay a live conversation would notice.
    if show_source && translate {
        let original = decode(session, turn, language, false)?;
        if !original.is_empty() {
            println!("{} {} {}", turn.label(), source_tag(language), original);
        }
    }

    let text = decode(session, turn, language, translate)?;
    if text.is_empty() {
        println!("{} (no speech found)", turn.label());
        return Ok(());
    }
    let tag = if translate {
        "en "
    } else {
        source_tag(language)
    };
    println!("{} {tag} {text}", turn.label());
    println!(
        "{:>8}  ({:.1}s of audio, decoded in {:.1?})",
        "",
        turn.secs(),
        started.elapsed()
    );
    Ok(())
}

/// Runs Whisper over a turn and returns the text, whitespace and all, tidied onto one line.
fn decode(
    session: &mut TranscriptionSession<'_>,
    turn: &Turn,
    language: Option<&str>,
    translate: bool,
) -> Result<String> {
    let options = TranscriptionOptions {
        language: language.map(str::to_string),
        translate,
        // Every turn is its own sentence: without this Whisper drags the tail of the previous one
        // into the new one and mangles both.
        no_context: true,
        // Whisper's timestamp splitting can drop or repeat text in the first few seconds; past ten
        // seconds it is better off splitting.
        single_segment: turn.secs() < 10.0,
        ..TranscriptionOptions::default()
    };
    let output = session.transcribe_samples(turn.samples(), Some(&options))?;
    Ok(output
        .text()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" "))
}

/// What to call the untranslated line: the language we forced, or `you` when we did not force one.
fn source_tag(language: Option<&str>) -> &'static str {
    match language {
        Some("zh") => "zh ",
        Some("en") => "en ",
        Some("ja") => "ja ",
        Some("ko") => "ko ",
        Some("fr") => "fr ",
        Some("de") => "de ",
        Some("es") => "es ",
        Some("ru") => "ru ",
        _ => "you",
    }
}

/// One utterance on its way to the model, plus where it came from.
struct Turn {
    samples: Vec<f32>,
    /// Where in the audio the turn ended, in seconds.
    at_secs: f32,
}

impl Turn {
    fn secs(&self) -> f32 {
        self.samples.len() as f32 / WHISPER_SAMPLE_RATE as f32
    }

    fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// `[mm:ss]`, the way a transcript timestamps a line.
    fn label(&self) -> String {
        stamp(self.at_secs)
    }
}

/// Totals for the closing line.
struct Session {
    captured_secs: f32,
    turns: usize,
    dropped: usize,
}

/// Where the audio comes from. Both sources look identical downstream: a channel of device-rate,
/// possibly multi-channel `f32` chunks.
struct Source {
    sample_rate: u32,
    channels: usize,
    samples: Receiver<Vec<f32>>,
    producer: JoinHandle<()>,
}

/// Opens the default microphone and starts pumping it into a channel.
fn open_microphone(stopper: &Stopper, stop_flag: &Arc<AtomicBool>) -> Result<Source> {
    let microphone = MicrophoneBuilder::new()
        .default_device()?
        .default_config()?
        .prefer_sample_rates([PREFERRED_SAMPLE_RATE])
        .prefer_channel_counts([PREFERRED_CHANNELS])
        .open_stream()?;
    let config = *microphone.config();

    println!(
        "\nMicrophone: {} Hz, {} channel(s) — resampling to {WHISPER_SAMPLE_RATE} Hz mono",
        config.sample_rate.get(),
        config.channel_count.get()
    );
    println!("Press Enter to start (Ctrl+C to abort)...");
    if stopper.wait() == Stop::Interrupt {
        println!("Aborted.");
        // A closed channel ends the session immediately; `main` returns right after.
        let (_, receiver) = channel();
        return Ok(Source {
            sample_rate: config.sample_rate.get(),
            channels: config.channel_count.get() as usize,
            samples: receiver,
            producer: thread::spawn(|| {}),
        });
    }

    // Capture runs on its own thread, so a slow decode can never stall the device.
    let (sender, receiver) = channel();
    let producer = {
        let stop_flag = stop_flag.clone();
        thread::spawn(move || {
            let mut microphone = microphone;
            let mut chunk: Vec<f32> = Vec::with_capacity(1024);
            while !stop_flag.load(Ordering::SeqCst) {
                match microphone.next() {
                    Some(sample) => {
                        chunk.push(sample);
                        if chunk.len() >= 1024 && sender.send(std::mem::take(&mut chunk)).is_err() {
                            break;
                        }
                    }
                    // The stream ended on its own: device error or removal.
                    None => break,
                }
            }
            if !chunk.is_empty() {
                let _ = sender.send(chunk);
            }
        })
    };

    Ok(Source {
        sample_rate: config.sample_rate.get(),
        channels: config.channel_count.get() as usize,
        samples: receiver,
        producer,
    })
}

/// Decodes a media file and feeds it through the same channel the microphone would use.
///
/// Handy for checking the conditioning chain without talking, and for reproducing a bad take.
fn open_file(path: &std::path::Path) -> Result<Source> {
    let decoded = decode_file(path)?;
    if decoded.samples.is_empty() {
        bail!("{} contains no audio", path.display());
    }

    println!(
        "\nFile: {} — {} Hz, {} channel(s), {:.2}s",
        path.display(),
        decoded.sample_rate,
        decoded.channels,
        decoded.samples.len() as f32 / (decoded.sample_rate * decoded.channels as u32) as f32
    );

    let (sender, receiver) = channel();
    let producer = thread::spawn(move || {
        for chunk in decoded.samples.chunks(1024) {
            if sender.send(chunk.to_vec()).is_err() {
                break;
            }
        }
        // Dropping the sender is what tells the session loop the audio is finished.
    });

    Ok(Source {
        sample_rate: decoded.sample_rate,
        channels: decoded.channels,
        samples: receiver,
        producer,
    })
}

/// Everything the conditioning stages need to know about the incoming audio.
struct Conditioning {
    device_rate: u32,
    channels: usize,
    denoise: bool,
}

/// The live loop: drain capture, condition the audio, hand turns to the worker.
fn run_session(
    samples: Receiver<Vec<f32>>,
    stopper: &Stopper,
    stop_flag: &Arc<AtomicBool>,
    work: &Sender<Turn>,
    conditioning: Conditioning,
    pause_ms: u64,
    out: Option<PathBuf>,
) -> Result<Session> {
    let Conditioning {
        device_rate,
        channels,
        denoise,
    } = conditioning;
    // Two reasons RNNoise is off unless asked for:
    //  - it is trained on 48 kHz wideband audio, so a 16 kHz source leaves it seeing a spectrum
    //    with nothing above 8 kHz;
    //  - measured on this crate's own test audio, running it chopped a 2.8 s utterance into a
    //    0.4 s one and turned a correct transcript into nonsense. The level comes back fine, the
    //    speech does not. Try `--denoise` on your own microphone and compare before relying on it.
    if denoise && device_rate < 32_000 {
        println!("Denoise: skipped — {device_rate} Hz is too narrowband for RNNoise.");
    }
    let mut denoiser = if denoise && device_rate >= 32_000 {
        Some(
            NoiseReducer::new(device_rate as usize)
                .map_err(|err| anyhow!("cannot start the noise reducer: {err}"))?,
        )
    } else {
        None
    };
    let mut resampler = LiveResampler::new(device_rate, WHISPER_SAMPLE_RATE)?;
    let mut segmenter = Segmenter::new(pause_ms)?;

    let mut emitter = Emitter {
        work,
        spoken: Vec::new(),
        pre_roll: Vec::new(),
        turns: 0,
        dropped: 0,
    };
    let mut pending_16k: Vec<f32> = Vec::new();
    let mut processed = 0usize;
    let mut captured = 0usize;

    stopper.start_stdin();
    println!(
        "\nListening — {:.1}s of quiet ends a turn. Press Enter or Ctrl+C to stop.\n",
        pause_ms as f32 / 1_000.0
    );

    loop {
        if let Some(stop) = stopper.poll() {
            println!(
                "\nStopping ({}).",
                match stop {
                    Stop::Enter => "Enter",
                    Stop::Interrupt => "Ctrl+C",
                }
            );
            break;
        }

        let mut finished = false;
        let mut idle = true;
        loop {
            match samples.try_recv() {
                Ok(chunk) => {
                    idle = false;
                    captured += chunk.len();

                    let chunk = match denoiser.as_mut() {
                        Some(denoiser) => denoise_chunk(denoiser, &chunk, device_rate)?,
                        None => chunk,
                    };
                    let mono = if channels == 1 {
                        chunk
                    } else {
                        mix_down(&chunk, channels)
                    };

                    resampler.push(&mono);
                    pending_16k.extend(resampler.drain()?);
                }
                // Nothing new right now.
                Err(TryRecvError::Empty) => break,
                // The producer is gone: a file finished, or the device went away.
                Err(TryRecvError::Disconnected) => {
                    finished = true;
                    break;
                }
            }
        }

        // Score whole windows; a partial window waits for the next chunk.
        while pending_16k.len() >= VAD_WINDOW {
            let at_secs = (processed + VAD_WINDOW) as f32 / WHISPER_SAMPLE_RATE as f32;
            emitter.keep_pre_roll(&pending_16k[..VAD_WINDOW]);
            let turns = segmenter.feed(&pending_16k[..VAD_WINDOW])?;
            pending_16k.drain(..VAD_WINDOW);
            processed += VAD_WINDOW;

            for speech in turns {
                emitter.emit(speech, at_secs)?;
            }
        }

        // Only stop once everything that arrived has been scored, or a trailing turn is lost.
        if finished {
            for speech in segmenter.finish()? {
                emitter.emit(speech, processed as f32 / WHISPER_SAMPLE_RATE as f32)?;
            }
            println!("\nEnd of audio.");
            break;
        }

        if idle {
            thread::sleep(Duration::from_millis(20));
        }
    }

    stop_flag.store(true, Ordering::SeqCst);
    // Drain whatever the device already delivered, so the total is honest.
    while let Ok(chunk) = samples.try_recv() {
        captured += chunk.len();
    }

    if let Some(out) = out {
        if emitter.spoken.is_empty() {
            println!("No speech captured, nothing written.");
        } else {
            write_wav(&out, &emitter.spoken, WHISPER_SAMPLE_RATE)?;
            println!(
                "Wrote {} — {:.1}s of speech at {WHISPER_SAMPLE_RATE} Hz.",
                out.display(),
                emitter.spoken.len() as f32 / WHISPER_SAMPLE_RATE as f32
            );
        }
    }

    Ok(Session {
        captured_secs: captured as f32 / (device_rate * channels as u32) as f32,
        turns: emitter.turns,
        dropped: emitter.dropped,
    })
}

/// `[mm:ss]` for a position given in seconds.
fn stamp(at_secs: f32) -> String {
    let whole = at_secs as u32;
    format!("[{:02}:{:02}]", whole / 60, whole % 60)
}

/// Milliseconds to 16 kHz frames.
fn ms_to_samples(ms: usize) -> usize {
    ms * WHISPER_SAMPLE_RATE as usize / 1_000
}

/// Hands finished turns to the worker, and decides which ones are worth sending.
struct Emitter<'a> {
    work: &'a Sender<Turn>,
    /// Every turn that was sent, concatenated — what `--out` writes.
    spoken: Vec<f32>,
    /// The last [`PRE_ROLL_MS`] of audio, oldest sample first.
    pre_roll: Vec<f32>,
    turns: usize,
    dropped: usize,
}

impl Emitter<'_> {
    /// Keeps the tail of `window` so a turn can start with the audio just before its onset.
    fn keep_pre_roll(&mut self, window: &[f32]) {
        self.pre_roll.extend_from_slice(window);
        let limit = ms_to_samples(PRE_ROLL_MS);
        if self.pre_roll.len() > limit {
            self.pre_roll.drain(..self.pre_roll.len() - limit);
        }
    }

    /// Scores one finished turn, then sends it or explains why it was dropped.
    fn emit(&mut self, speech: Vec<f32>, at_secs: f32) -> Result<()> {
        if let Err(why) = accept(&speech)? {
            self.dropped += 1;
            println!("{} skipped — {why}", stamp(at_secs));
            return Ok(());
        }

        // The turn as Whisper will hear it: a little lead-in so the first syllable is not clipped,
        // the turn itself, and a little quiet so the last one is not cut off.
        let mut turn = self.pre_roll.clone();
        turn.extend_from_slice(&speech);
        turn.extend(std::iter::repeat_n(0.0, ms_to_samples(TAIL_PAD_MS)));

        self.spoken.extend_from_slice(&turn);
        if self
            .work
            .send(Turn {
                samples: turn,
                at_secs,
            })
            .is_err()
        {
            // The worker is gone; keep going rather than dying mid-session.
            return Ok(());
        }
        self.turns += 1;
        println!(
            "{} heard {:.1}s",
            stamp(at_secs),
            speech.len() as f32 / WHISPER_SAMPLE_RATE as f32
        );
        Ok(())
    }
}

/// Decides whether a finished turn is worth decoding.
///
/// Everything here already looked like speech to the segmenter once. The question is whether it
/// kept looking like speech: a door slam can open a turn that is then silence until the pause
/// closes it, and Whisper turns silence into confident nonsense. Short turns are just as bad —
/// half a second of anything is not enough for Whisper to hear a sentence in it.
fn accept(speech: &[f32]) -> Result<std::result::Result<(), String>> {
    let seconds = speech.len() as f32 / WHISPER_SAMPLE_RATE as f32;
    if speech.len() < ms_to_samples(MIN_SPEECH_MS as usize) {
        return Ok(Err(format!("only {seconds:.2}s of audio")));
    }

    let windows = speech.len() / VAD_WINDOW;
    let voiced = count_voiced(speech)?;
    let ratio = voiced as f32 / windows as f32;
    if ratio < MIN_SPEECH_RATIO {
        return Ok(Err(format!("only {:.0}% of it is speech", ratio * 100.0)));
    }
    Ok(Ok(()))
}

/// How many windows of `speech` a fresh voice detector calls speech.
///
/// A second opinion rather than a memory of the detector that opened the turn: this one only ever
/// sees the finished utterance, so it cannot be fooled by whatever triggered the onset.
fn count_voiced(speech: &[f32]) -> Result<usize> {
    let mut vad = TinySilero::new(VADOption {
        samplerate: WHISPER_SAMPLE_RATE,
        voice_threshold: VAD_THRESHOLD,
        ..Default::default()
    })
    .map_err(|err| anyhow!("cannot start the verification detector: {err}"))?;

    let mut voiced = 0usize;
    for window in speech.chunks(VAD_WINDOW) {
        if window.len() == VAD_WINDOW && vad.predict(window) > VAD_THRESHOLD {
            voiced += 1;
        }
    }
    Ok(voiced)
}

/// Milliseconds covered by one detector window.
const WINDOW_MS: u64 = VAD_WINDOW as u64 * 1_000 / WHISPER_SAMPLE_RATE as u64;

/// Turns a stream of 16 kHz audio into finished utterances.
///
/// Wraps `voice-engine`'s `VadProcessor`, which owns the part that is easy to get wrong — how much
/// quiet closes a turn, how long a turn has to be to count, and how to keep the audio in between
/// intact. Removing the pauses inside a turn sounds like a good idea and is not: gluing speech
/// windows together hands Whisper time-compressed audio with a click at every join.
///
/// Two things this adds on top:
///
/// - **A closing feed.** The detector only closes a turn once it has *seen* enough quiet, so when
///   the audio runs out we feed it the silence it is waiting for rather than losing the last turn.
/// - **A length limit.** Somebody who never pauses would otherwise produce one endless turn that
///   never reaches the model at all.
struct Segmenter {
    processor: VadProcessor,
    events: broadcast::Receiver<SessionEvent>,
    /// Position of the frame being fed, in milliseconds.
    ///
    /// The segmenter measures its padding in milliseconds, and using the audio clock rather than
    /// the wall clock keeps `--file` runs — which decode minutes in a second — behaving exactly
    /// like live ones.
    clock_ms: u64,
    /// When the turn currently being spoken started, in milliseconds.
    open_since_ms: Option<u64>,
    pause_ms: u64,
}

impl Segmenter {
    fn new(pause_ms: u64) -> Result<Self> {
        let option = VADOption {
            samplerate: WHISPER_SAMPLE_RATE,
            voice_threshold: VAD_THRESHOLD,
            // Minimum length of a turn, in ms.
            speech_padding: MIN_SPEECH_MS,
            // Quiet time that closes a turn, in ms.
            silence_padding: pause_ms,
            max_buffer_duration_secs: MAX_UTTERANCE_SECS,
            ..Default::default()
        };
        let (events_tx, events_rx) = broadcast::channel(64);
        let processor = VadProcessor::new(
            Box::new(
                TinySilero::new(option.clone())
                    .map_err(|err| anyhow!("cannot start the voice detector: {err}"))?,
            ),
            events_tx,
            option,
        )
        .map_err(|err| anyhow!("cannot start the segmenter: {err}"))?;

        Ok(Self {
            processor,
            events: events_rx,
            clock_ms: 0,
            open_since_ms: None,
            pause_ms,
        })
    }

    /// Feeds 16 kHz mono audio and returns every turn that closed because of it.
    fn feed(&mut self, window: &[f32]) -> Result<Vec<Vec<f32>>> {
        for hop in window.chunks(VAD_WINDOW) {
            self.push(hop)?;
        }

        let mut turns = self.take_turns();
        // Somebody mid-monologue never gives us the pause we are waiting for, so impose one.
        if self.open_too_long() {
            turns.extend(self.close_open_turn()?);
        }
        Ok(turns)
    }

    /// Hands one detector window to the segmenter.
    fn push(&mut self, hop: &[f32]) -> Result<()> {
        let pcm: Vec<i16> = hop
            .iter()
            .map(|sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect();
        let length = pcm.len();
        let mut frame = AudioFrame {
            track_id: "mic".to_string(),
            samples: Samples::PCM { samples: pcm },
            timestamp: self.clock_ms,
            sample_rate: WHISPER_SAMPLE_RATE,
        };
        self.clock_ms += length as u64 * 1_000 / WHISPER_SAMPLE_RATE as u64;
        self.processor
            .process_frame(&mut frame)
            .map_err(|err| anyhow!("voice detection failed: {err}"))
    }

    /// Whether the turn being spoken has passed [`MAX_UTTERANCE_SECS`].
    fn open_too_long(&self) -> bool {
        self.open_since_ms
            .is_some_and(|since| self.clock_ms.saturating_sub(since) >= MAX_UTTERANCE_SECS * 1_000)
    }

    /// Closes whatever turn is still open by feeding it the quiet it is waiting for.
    ///
    /// The detector has to *see* the whole pause before it lets go, and it keeps calling speech
    /// speech for a window or two after the audio actually stops, so a fixed length of silence is
    /// not enough — feed one window at a time until the turn comes out, with a bound so a stuck
    /// detector cannot spin here forever.
    fn close_open_turn(&mut self) -> Result<Vec<Vec<f32>>> {
        let quiet = vec![0.0f32; VAD_WINDOW];
        let limit = self.pause_ms / WINDOW_MS + 8;
        for _ in 0..limit {
            self.push(&quiet)?;
            let turns = self.take_turns();
            if !turns.is_empty() {
                return Ok(turns);
            }
        }
        Ok(Vec::new())
    }

    /// Feeds the silence that closes a turn and returns the last turn of the audio.
    fn finish(&mut self) -> Result<Vec<Vec<f32>>> {
        if self.open_since_ms.is_none() {
            return Ok(Vec::new());
        }
        self.close_open_turn()
    }

    /// Takes the utterances that closed since the last call, and tracks the one still open.
    fn take_turns(&mut self) -> Vec<Vec<f32>> {
        let mut turns = Vec::new();
        while let Ok(event) = self.events.try_recv() {
            match event {
                SessionEvent::Speaking { start_time, .. } => self.open_since_ms = Some(start_time),
                SessionEvent::Silence {
                    samples: Some(pcm), ..
                } => {
                    self.open_since_ms = None;
                    turns.push(
                        pcm.iter()
                            .map(|sample| *sample as f32 / i16::MAX as f32)
                            .collect(),
                    );
                }
                _ => {}
            }
        }
        turns
    }
}

/// Runs RNNoise over one chunk of device-rate audio.
fn denoise_chunk(denoiser: &mut NoiseReducer, chunk: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
    let pcm: Vec<i16> = chunk
        .iter()
        .map(|sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect();
    let mut frame = AudioFrame {
        track_id: "mic".to_string(),
        samples: Samples::PCM { samples: pcm },
        timestamp: 0,
        sample_rate,
    };
    denoiser
        .process_frame(&mut frame)
        .map_err(|err| anyhow!("denoise failed: {err}"))?;

    match &frame.samples {
        Samples::PCM { samples } => Ok(samples
            .iter()
            .map(|sample| *sample as f32 / i16::MAX as f32)
            .collect()),
        _ => bail!("the noise reducer returned no PCM"),
    }
}

/// A rubato resampler driven chunk by chunk, for audio that is still arriving.
struct LiveResampler {
    resampler: Async<f32>,
    pending: Vec<f32>,
}

impl LiveResampler {
    fn new(source_rate: u32, target_rate: u32) -> Result<Self> {
        let params = SincInterpolationParameters::new(64, WindowFunction::BlackmanHarris2)
            .oversampling_factor(128)
            .interpolation(SincInterpolationType::Cubic);
        let resampler = Async::<f32>::new_sinc(
            target_rate as f64 / source_rate as f64,
            1.1,
            &params,
            RESAMPLE_CHUNK,
            1,
            FixedAsync::Input,
        )
        .map_err(|err| anyhow!("cannot build the resampler: {err}"))?;
        Ok(Self {
            resampler,
            pending: Vec::new(),
        })
    }

    /// Queues device-rate audio; it is resampled on the next [`Self::drain`].
    fn push(&mut self, samples: &[f32]) {
        self.pending.extend_from_slice(samples);
    }

    /// Resamples every complete chunk that has arrived so far.
    fn drain(&mut self) -> Result<Vec<f32>> {
        let mut output = Vec::new();
        loop {
            let needed = self.resampler.input_frames_next();
            if self.pending.len() < needed {
                break;
            }

            let mut buffer = vec![0.0f32; self.resampler.output_frames_max()];
            let capacity = buffer.len();
            {
                let input = InterleavedSlice::new(&self.pending[..needed], 1, needed)
                    .map_err(|err| anyhow!("{err}"))?;
                let mut adapter = InterleavedSlice::new_mut(&mut buffer, 1, capacity)
                    .map_err(|err| anyhow!("{err}"))?;
                let (_, produced) = self
                    .resampler
                    .process_into_buffer(&input, &mut adapter, None)
                    .map_err(|err| anyhow!("{err}"))?;
                output.extend_from_slice(&buffer[..produced]);
            }
            self.pending.drain(..needed);
        }
        Ok(output)
    }
}

/// Prints every input device rodio can see.
fn list_inputs() -> Result<()> {
    let inputs = available_inputs().context("failed to enumerate input devices")?;
    if inputs.is_empty() {
        bail!("no input devices available");
    }
    println!("{} input device(s):", inputs.len());
    for input in inputs {
        println!("  {input}");
    }
    Ok(())
}

/// Reads `--flag value` out of the argument list.
fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(String::as_str)
}
