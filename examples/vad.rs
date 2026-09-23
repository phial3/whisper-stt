//! Speech segmentation with **voice-engine 0.3**'s neural VAD.
//!
//! Transcribing silence is wasted work and it makes Whisper hallucinate filler text. The usual
//! fix is to cut the audio into speech segments first, which needs a voice activity detector.
//! `voice-engine` ships two that run entirely offline with weights baked into the crate: a
//! hand-written Rust port of Silero VAD (`TinySilero`) and a port of Ten VAD (`TinyTen`). No
//! model download, no ONNX runtime, no network.
//!
//! This example decodes a media file, runs the detector over it, prints a timeline of the speech
//! probability, lists the segments it found, and can write a speech-only WAV that you can hand
//! straight to `Transcriber::transcribe_file`.
//!
//! ```text
//! cargo run --example vad                                     # assets/test.mp3
//! cargo run --example vad -- assets/test.mp3 --engine ten     # use the Ten VAD instead
//! cargo run --example vad -- assets/test.mp3 --threshold 0.3  # more permissive
//! cargo run --example vad -- assets/test.mp3 --out speech.wav # keep only the speech
//! ```
//!
//! The detectors are window-based and take different frame sizes: Silero looks at 512 samples
//! (32 ms at 16 kHz), Ten at 256 (16 ms). They also score on different scales — on the same clip
//! Ten's probabilities run well below Silero's, which is why the default threshold follows the
//! engine. Compare them before trusting one for your own audio.

use std::path::PathBuf;

use anyhow::{Result, bail};
use voice_engine::media::vad::{TinySilero, TinyTen, VADOption, VadType};
use whisper_stt::WHISPER_SAMPLE_RATE;
use whisper_stt::audio::{decode_file, mix_down, resample};

#[path = "shared/mod.rs"]
mod shared;
use shared::write_wav;

/// Frames Silero consumes per call. Its STFT uses a 256-sample window with a 128-sample stride
/// over a 512-sample chunk, so 512 it is: 32 ms at 16 kHz.
const SILERO_WINDOW: usize = 512;
/// Frames Ten consumes per call. Ten works with a 16 ms hop and a 768-sample analysis window, so
/// it expects 256 samples per `predict` call.
const TEN_WINDOW: usize = 256;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let input = positional(&args)
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("assets/test.mp3"));
    let engine = flag(&args, "--engine").unwrap_or("silero");
    if !matches!(engine, "silero" | "ten") {
        bail!("unknown engine {engine:?}; use silero or ten");
    }
    // Ten's probabilities sit lower than Silero's on the same audio, so the default threshold
    // follows the engine rather than being one number for both.
    let threshold: f32 = flag(&args, "--threshold")
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(if engine == "ten" { 0.3 } else { 0.5 });
    let out = flag(&args, "--out").map(PathBuf::from);

    // The VAD is fixed to 16 kHz, which is also what Whisper wants.
    let decoded = decode_file(&input)?;
    let mono = mix_down(&decoded.samples, decoded.channels);
    let samples = if decoded.sample_rate == WHISPER_SAMPLE_RATE {
        mono
    } else {
        resample(&mono, decoded.sample_rate, WHISPER_SAMPLE_RATE)
    };
    let window = if engine == "ten" {
        TEN_WINDOW
    } else {
        SILERO_WINDOW
    };

    if samples.len() < window {
        bail!(
            "{} is shorter than one {} ms analysis window",
            input.display(),
            window * 1000 / WHISPER_SAMPLE_RATE as usize
        );
    }

    println!(
        "Input: {} — {} Hz, {} channel(s), resampled to {:.2}s at {WHISPER_SAMPLE_RATE} Hz",
        input.display(),
        decoded.sample_rate,
        decoded.channels,
        samples.len() as f32 / WHISPER_SAMPLE_RATE as f32
    );
    println!(
        "Engine: {engine} — {window}-sample windows ({} ms), threshold {threshold:.2}\n",
        window * 1000 / WHISPER_SAMPLE_RATE as usize
    );

    let option = VADOption {
        samplerate: WHISPER_SAMPLE_RATE,
        r#type: if engine == "ten" {
            VadType::Ten
        } else {
            VadType::Silero
        },
        voice_threshold: threshold,
        ..Default::default()
    };

    // Each detector takes a different sample type, so wrap them behind one closure.
    let mut silero = TinySilero::new(option.clone())?;
    let mut ten = TinyTen::new(option)?;
    let mut predict = move |window: &[f32]| -> Result<f32> {
        if engine == "ten" {
            let pcm: Vec<i16> = window
                .iter()
                .map(|sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
                .collect();
            Ok(ten.predict(&pcm))
        } else {
            Ok(silero.predict(window))
        }
    };

    let mut scores: Vec<f32> = Vec::new();
    for chunk in samples.chunks(window) {
        let mut padded = vec![0.0; window];
        padded[..chunk.len()].copy_from_slice(chunk);
        scores.push(predict(&padded)?);
    }

    print_timeline(&scores, window);
    let segments = segments(&scores, threshold);
    print_segments(&segments, &scores, WHISPER_SAMPLE_RATE, window);

    if let Some(out) = out {
        let speech: Vec<f32> = segments
            .iter()
            .flat_map(|segment| {
                let start = segment.window * window;
                let end = (start + window * segment.windows).min(samples.len());
                samples[start..end].iter().copied()
            })
            .collect();

        write_wav(&out, &speech, WHISPER_SAMPLE_RATE)?;
        println!(
            "\nWrote {} — {:.2}s of speech from {:.2}s of audio ({:.0}% kept).",
            out.display(),
            speech.len() as f32 / WHISPER_SAMPLE_RATE as f32,
            samples.len() as f32 / WHISPER_SAMPLE_RATE as f32,
            100.0 * speech.len() as f32 / samples.len().max(1) as f32
        );
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

/// Collects the arguments that are not a flag and not a flag's value.
///
/// Without this, `--engine ten` would hand `ten` to the input path as well as to the flag.
fn positional(args: &[String]) -> Vec<&str> {
    let mut values = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        if matches!(arg, "--engine" | "--threshold" | "--out") {
            index += 2;
            continue;
        }
        if !arg.starts_with('-') {
            values.push(arg);
        }
        index += 1;
    }
    values
}

/// A run of consecutive windows that scored above the threshold.
struct Segment {
    window: usize,
    windows: usize,
}

/// Groups scored windows into runs of speech.
fn segments(scores: &[f32], threshold: f32) -> Vec<Segment> {
    let mut segments: Vec<Segment> = Vec::new();
    for (index, score) in scores.iter().enumerate() {
        if *score > threshold {
            match segments.last_mut() {
                Some(segment) if segment.window + segment.windows == index => segment.windows += 1,
                _ => segments.push(Segment {
                    window: index,
                    windows: 1,
                }),
            }
        }
    }
    segments
}

/// Draws the probability timeline: `#` is speech, `.` is silence, height is the score.
fn print_timeline(scores: &[f32], window: usize) {
    // 32 windows per line, which is one second per line at 32 ms a window.
    const PER_LINE: usize = 32;

    println!(
        "Speech probability (one column is one {} ms window):",
        window * 1000 / WHISPER_SAMPLE_RATE as usize
    );
    for (line, chunk) in scores.chunks(PER_LINE).enumerate() {
        let start = line * PER_LINE * window;
        let seconds = start as f32 / WHISPER_SAMPLE_RATE as f32;
        let bars: String = chunk
            .iter()
            .map(|score| match *score {
                score if score > 0.75 => '#',
                score if score > 0.5 => '+',
                score if score > 0.25 => '-',
                _ => '.',
            })
            .collect();
        println!("  {seconds:5.1}s |{bars}|");
    }
    println!("         legend: # >0.75   + >0.50   - >0.25   . silence\n");
}

/// Prints the detected segments as timestamps, and how much of the clip they cover.
fn print_segments(segments: &[Segment], scores: &[f32], sample_rate: u32, window: usize) {
    if segments.is_empty() {
        println!("No speech detected — try a lower --threshold.");
        return;
    }

    let window_ms = window as f32 * 1000.0 / sample_rate as f32;
    println!("{} speech segment(s):", segments.len());
    for segment in segments {
        let start = segment.window as f32 * window_ms;
        let end = (segment.window + segment.windows) as f32 * window_ms;
        let peak = scores[segment.window..segment.window + segment.windows]
            .iter()
            .fold(0.0f32, |peak, score| peak.max(*score));
        println!(
            "  {:6.2}s – {:6.2}s  ({:5.2}s, peak {peak:.2})",
            start / 1000.0,
            end / 1000.0,
            (end - start) / 1000.0
        );
    }

    let speech_windows: usize = segments.iter().map(|segment| segment.windows).sum();
    let coverage = 100.0 * speech_windows as f32 / scores.len().max(1) as f32;
    println!("\nSpeech covers {coverage:.0}% of the clip.");
}
