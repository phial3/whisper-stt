//! Records the microphone with **cpal** and writes a WAV file that Whisper can consume directly.
//!
//! Same interactive flow as `recording.rs`:
//!
//! 1. Press **Enter** to start recording (Ctrl+C here aborts).
//! 2. Press **Enter** again — or **Ctrl+C** — to stop.
//! 3. The capture is resampled to 16 kHz mono, written to `recordings/`, played back, and the
//!    program exits by itself once playback finishes.
//!
//! ```text
//! cargo run --example record_cpal                     # -> recordings/cpal-<timestamp>.wav
//! cargo run --example record_cpal -- note.wav         # -> recordings/note.wav
//! cargo run --example record_cpal -- --list           # enumerate devices and their configs
//! cargo run --example record_cpal -- --transcribe     # also transcribe with `tiny`
//! ```
//!
//! Where `recording.rs` leans on rodio's high-level microphone API, this example drives cpal itself and
//! therefore has full control over the stream: it prints supported configs and handles every sample
//! format. The `whisper-stt` resampling helpers are reused here, so the saved file is exactly the
//! input `Transcriber::transcribe_file` would build from any other media file.

use std::sync::mpsc::{Receiver, Sender, channel};

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, StreamConfig};
use whisper_stt::audio::{mix_down, resample};
use whisper_stt::{WHISPER_SAMPLE_RATE, WhisperModel};

#[path = "shared/mod.rs"]
mod shared;
use shared::{Stop, Stopper, output_path, play, write_wav};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|arg| arg == "--list" || arg == "-l") {
        return list_inputs();
    }

    let transcribe = args.iter().any(|arg| arg == "--transcribe");
    let output = output_path(
        "cpal",
        args.iter()
            .find(|arg| !arg.starts_with('-'))
            .map(String::as_str),
    )?;

    let stopper = Stopper::arm()?;

    println!();
    println!("Output file: {}", output.display());
    println!("Press Enter to start recording (Ctrl+C to abort)...");
    if stopper.wait() == Stop::Interrupt {
        println!("Aborted, nothing recorded.");
        return Ok(());
    }

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("no default input device; pass --list to see what is available")?;

    let supported = device
        .default_input_config()
        .context("failed to read the default input config")?;
    println!(
        "Recording on {} at {} Hz, {} channel(s), {:?} — press Enter or Ctrl+C to stop.",
        device.id()?,
        supported.sample_rate(),
        supported.channels(),
        supported.sample_format()
    );

    // The callback runs on the audio thread, so it must never block or allocate: push the chunk down
    // a channel and let the main thread collect it.
    let (sender, receiver): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = channel();
    let stream = build_stream(&device, &supported, sender)?;
    stream.play()?;

    let stop = stopper.wait_with_progress(None, "Recording...");

    // Dropping the stream stops capture; whatever is already queued is still drained below.
    drop(stream);

    let mut samples: Vec<f32> = Vec::new();
    while let Ok(chunk) = receiver.try_recv() {
        samples.extend_from_slice(&chunk);
    }
    if samples.is_empty() {
        bail!("captured no audio — is the microphone permitted and unmuted?");
    }

    // Whisper only understands 16 kHz mono, so downmix first and resample afterwards.
    let channels = supported.channels() as usize;
    let mono = mix_down(&samples, channels);
    let whisper_input = resample(&mono, supported.sample_rate(), WHISPER_SAMPLE_RATE)?;

    write_wav(&output, &whisper_input, WHISPER_SAMPLE_RATE)?;
    println!(
        "Saved {} — {:.2}s at {} Hz mono ({})",
        output.display(),
        whisper_input.len() as f32 / WHISPER_SAMPLE_RATE as f32,
        WHISPER_SAMPLE_RATE,
        match stop {
            Stop::Enter => "stopped with Enter",
            Stop::Interrupt => "stopped with Ctrl+C",
        }
    );

    if transcribe {
        transcribe_file(&output)?;
    }

    play(&output)
}

/// Builds an input stream, normalising whatever sample format the device uses to `f32`.
fn build_stream(
    device: &cpal::Device,
    supported: &cpal::SupportedStreamConfig,
    sender: Sender<Vec<f32>>,
) -> Result<cpal::Stream> {
    let config: StreamConfig = (*supported).into();

    macro_rules! build {
        ($format:ty) => {
            device.build_input_stream(
                config,
                move |data: &[$format], _: &_| {
                    let chunk: Vec<f32> = data.iter().map(|sample| sample.to_sample()).collect();
                    // A full channel means the consumer cannot keep up; dropping is preferable to
                    // blocking the audio thread.
                    let _ = sender.send(chunk);
                },
                |err| eprintln!("stream error: {err}"),
                None,
            )
        };
    }

    let stream = match supported.sample_format() {
        SampleFormat::I8 => build!(i8),
        SampleFormat::I16 => build!(i16),
        SampleFormat::I32 => build!(i32),
        SampleFormat::I64 => build!(i64),
        SampleFormat::U8 => build!(u8),
        SampleFormat::U16 => build!(u16),
        SampleFormat::U32 => build!(u32),
        SampleFormat::U64 => build!(u64),
        SampleFormat::F32 => build!(f32),
        SampleFormat::F64 => build!(f64),
        other => bail!("sample format {other:?} is not supported by this example"),
    }
    .context("failed to build the input stream")?;

    Ok(stream)
}

/// Transcribes the recording with the `tiny` model.
fn transcribe_file(path: &std::path::Path) -> Result<()> {
    // Needs a tokio runtime; `tiny` is ~75 MB and is downloaded on first use.
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        let store = whisper_stt::ModelStore::pretrained(WhisperModel::Tiny, "models");
        store.ensure().await?;
        let transcriber = whisper_stt::Transcriber::new(store.path())?;
        let result = transcriber.transcribe_file(path, None)?;
        println!("Transcript: {}", result.text().trim());
        Ok::<(), anyhow::Error>(())
    })
}

/// Lists input devices together with every configuration they support.
fn list_inputs() -> Result<()> {
    let host = cpal::default_host();
    let devices = host
        .input_devices()
        .context("failed to enumerate input devices")?;

    let mut found = 0;
    for device in devices {
        found += 1;
        println!("{}", device.id()?);
        if let Ok(configs) = device.supported_input_configs() {
            for config in configs {
                println!(
                    "    {}–{} Hz, {} channel(s), {:?}",
                    config.min_sample_rate(),
                    config.max_sample_rate(),
                    config.channels(),
                    config.sample_format()
                );
            }
        }
    }

    if found == 0 {
        bail!("no input devices available");
    }
    Ok(())
}
