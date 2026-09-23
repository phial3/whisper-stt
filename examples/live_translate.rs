//! Live translation: microphone → denoise → resample → VAD → Whisper → English text.
//!
//! The whole crate's pipeline in one loop, and the reason each piece exists:
//!
//! 1. **Capture** — rodio's microphone, asked for 16 kHz mono (hardware usually gives 44.1/48 kHz).
//! 2. **Denoise** — `voice-engine`'s `NoiseReducer` (RNNoise) runs on the device-rate audio.
//! 3. **Resample** — rubato 5.0 resamples to the 16 kHz Whisper requires, in real time. A naive
//!    linear resample would fold everything above 8 kHz back into the audio, which is exactly what
//!    makes a transcript fall apart.
//! 4. **VAD** — `voice-engine`'s Silero port decides which windows are speech, so silence is never
//!    sent to the model. Whisper hallucinates filler text on silence; feeding it whole utterances
//!    instead of a raw stream is what keeps the output clean.
//! 5. **Transcribe + translate** — `whisper-stt` with `translate: true`, on a worker thread so
//!    decoding never blocks capture.
//!
//! ```text
//! cargo run --example live_translate                       # auto-detect language -> English
//! cargo run --example live_translate -- --language zh      # skip auto-detection
//! cargo run --example live_translate -- --model tiny       # quick smoke test
//! cargo run --example live_translate -- --model large-v3   # turbo cannot translate, see below
//! cargo run --example live_translate -- --no-translate     # transcribe, do not translate
//! cargo run --example live_translate -- --denoise          # also run RNNoise
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
//! Press **Enter** to start and **Enter** (or Ctrl+C) to stop. Say a full sentence, pause, and it
//! comes back translated. This is not low latency: Whisper decodes each utterance after you finish
//! it, so expect a pause of a second or two per segment, longer for the bigger models.
//!
//! `--file` swaps the microphone for a media file and runs it through the identical chain, which is
//! how you check the conditioning stages without having to talk:
//!
//! ```text
//! cargo run --example live_translate -- --file recordings/rodio-1790179654.wav
//! ```

use std::collections::VecDeque;
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
use voice_engine::media::denoiser::NoiseReducer;
use voice_engine::media::processor::Processor;
use voice_engine::media::vad::{TinySilero, VADOption};
use voice_engine::media::{AudioFrame, Samples};
use whisper_stt::audio::{decode_file, mix_down};
use whisper_stt::{ModelStore, Transcriber, TranscriptionOptions, WHISPER_SAMPLE_RATE};

#[path = "shared/mod.rs"]
mod shared;
use shared::{Stop, Stopper, write_wav};

/// Asked for first; Whisper works on 16 kHz mono.
const PREFERRED_SAMPLE_RATE: NonZero<u32> = NonZero::new(16_000).expect("non-zero");
const PREFERRED_CHANNELS: NonZero<u16> = NonZero::new(1).expect("non-zero");

/// Frames rubato consumes per call at the device rate — about 11 ms at 44.1 kHz.
const RESAMPLE_CHUNK: usize = 512;
/// Frames the Silero port consumes per call: 32 ms at 16 kHz.
const VAD_WINDOW: usize = 512;
/// Speech probability above which a window counts as voice.
const VAD_THRESHOLD: f32 = 0.5;

/// Silence this long closes an utterance and sends it to the model.
const SILENCE_TIMEOUT: Duration = Duration::from_millis(500);
/// An utterance is cut here even without a pause, so decoding stays bounded.
const MAX_UTTERANCE: Duration = Duration::from_secs(20);
/// Anything shorter than this is dropped as a cough or a click.
const MIN_UTTERANCE: Duration = Duration::from_millis(300);
/// Audio kept before speech onset so the first word is not clipped.
const PRE_ROLL: Duration = Duration::from_millis(250);

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
    // Opt-in: see the note in `run_session`. RNNoise runs at the device rate.
    let denoise = args.iter().any(|arg| arg == "--denoise");
    let file = flag(&args, "--file").map(PathBuf::from);
    let out = flag(&args, "--out").map(PathBuf::from);

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

    // Decoding lives here: the model is loaded once, then fed one utterance at a time.
    let (work_tx, work_rx) = channel::<Vec<f32>>();
    let worker = thread::spawn(move || {
        let transcriber = match Transcriber::new(&model_path) {
            Ok(transcriber) => transcriber,
            Err(err) => {
                eprintln!("failed to load the model: {err}");
                return;
            }
        };
        for samples in work_rx {
            let options = TranscriptionOptions {
                language: language.as_deref(),
                translate,
                ..TranscriptionOptions::default()
            };
            let started = Instant::now();
            match transcriber.transcribe_samples(&samples, Some(&options)) {
                Ok(output) => {
                    let text = output.text().trim();
                    if !text.is_empty() {
                        println!(
                            "  → {}  ({:.1}s of audio, decoded in {:.1?})",
                            text,
                            samples.len() as f32 / WHISPER_SAMPLE_RATE as f32,
                            started.elapsed()
                        );
                    }
                }
                Err(err) => eprintln!("  ! transcription failed: {err}"),
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
        out,
    )?;

    // Dropping the sender ends the worker's loop.
    drop(work_tx);
    let _ = capture.join();
    let _ = worker.join();

    println!(
        "\nStopped. {:.1}s captured, {} utterance(s) sent to the model.",
        session.captured_secs, session.utterances
    );
    Ok(())
}

/// Totals for the closing line.
struct Session {
    captured_secs: f32,
    utterances: usize,
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

/// The live loop: drain capture, condition the audio, hand utterances to the worker.
fn run_session(
    samples: Receiver<Vec<f32>>,
    stopper: &Stopper,
    stop_flag: &Arc<AtomicBool>,
    work: &Sender<Vec<f32>>,
    conditioning: Conditioning,
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
    let mut vad = TinySilero::new(VADOption {
        samplerate: WHISPER_SAMPLE_RATE,
        voice_threshold: VAD_THRESHOLD,
        ..Default::default()
    })
    .map_err(|err| anyhow!("cannot start the VAD: {err}"))?;

    // All segmentation is measured in 16 kHz samples, not wall-clock time: a file mode run
    // processes minutes of audio in a fraction of a second, and even a live run should not change
    // its mind because the CPU stalled.
    let silence_samples = samples_for(SILENCE_TIMEOUT);
    let max_samples = samples_for(MAX_UTTERANCE);
    let min_samples = samples_for(MIN_UTTERANCE);
    let pre_roll_samples = samples_for(PRE_ROLL);

    let mut pending_16k: Vec<f32> = Vec::new();
    let mut pre_roll: VecDeque<f32> = VecDeque::new();
    let mut utterance: Vec<f32> = Vec::new();
    let mut in_utterance = false;
    let mut silence_run = 0usize;
    let mut sent: Vec<f32> = Vec::new();

    let mut processed = 0usize;
    let mut utterances = 0usize;
    let mut captured = 0usize;

    stopper.start_stdin();
    println!("\nListening — press Enter or Ctrl+C to stop.\n");

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

        // Score whole VAD windows; a partial window waits for the next chunk.
        while pending_16k.len() >= VAD_WINDOW {
            let window: Vec<f32> = pending_16k.drain(..VAD_WINDOW).collect();
            processed += VAD_WINDOW;

            if vad.predict(&window) > VAD_THRESHOLD {
                if !in_utterance {
                    // Seed with the audio just before onset, or the first word arrives clipped.
                    utterance.extend(pre_roll.drain(..));
                    in_utterance = true;
                }
                utterance.extend_from_slice(&window);
                silence_run = 0;
            } else if in_utterance {
                silence_run += VAD_WINDOW;
            } else {
                pre_roll.extend(window.iter().copied());
                while pre_roll.len() > pre_roll_samples {
                    pre_roll.pop_front();
                }
            }

            let settled = in_utterance && silence_run >= silence_samples;
            let too_long = utterance.len() >= max_samples;
            if settled || too_long {
                utterances += flush(&mut utterance, &mut sent, work, processed, min_samples)?;
                in_utterance = false;
                silence_run = 0;
            }
        }

        // Only stop once everything that arrived has been scored, or a trailing utterance
        // would be thrown away.
        if finished {
            println!("\nEnd of audio.");
            break;
        }

        if idle {
            thread::sleep(Duration::from_millis(20));
        }
    }

    // The audio can end mid-utterance; that one still deserves to be decoded.
    if in_utterance {
        utterances += flush(&mut utterance, &mut sent, work, processed, min_samples)?;
    }

    stop_flag.store(true, Ordering::SeqCst);
    // Drain whatever the device already delivered, so the total is honest.
    while let Ok(chunk) = samples.try_recv() {
        captured += chunk.len();
    }

    if let Some(out) = out {
        if sent.is_empty() {
            println!("No speech captured, nothing written.");
        } else {
            write_wav(&out, &sent, WHISPER_SAMPLE_RATE)?;
            println!(
                "Wrote {} — {:.1}s of speech at {WHISPER_SAMPLE_RATE} Hz.",
                out.display(),
                sent.len() as f32 / WHISPER_SAMPLE_RATE as f32
            );
        }
    }

    Ok(Session {
        captured_secs: captured as f32 / (device_rate * channels as u32) as f32,
        utterances,
    })
}

/// Converts a duration to 16 kHz frames.
fn samples_for(duration: Duration) -> usize {
    (duration.as_secs_f32() * WHISPER_SAMPLE_RATE as f32) as usize
}

/// Closes the current utterance: keeps it if it is long enough, and sends it for decoding.
///
/// Returns how many utterances were actually sent.
fn flush(
    utterance: &mut Vec<f32>,
    sent: &mut Vec<f32>,
    work: &Sender<Vec<f32>>,
    processed: usize,
    min_samples: usize,
) -> Result<usize> {
    if utterance.len() < min_samples {
        utterance.clear();
        return Ok(0);
    }

    println!(
        "[{:6.1}s] utterance of {:.1}s",
        processed as f32 / WHISPER_SAMPLE_RATE as f32,
        utterance.len() as f32 / WHISPER_SAMPLE_RATE as f32
    );
    sent.extend_from_slice(utterance);
    if work.send(std::mem::take(utterance)).is_err() {
        // The worker is gone; keep going rather than dying mid-session.
        utterance.clear();
        return Ok(0);
    }
    Ok(1)
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
