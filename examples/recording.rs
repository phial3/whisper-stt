//! Records the microphone with **rodio** and saves the result as a WAV file.
//!
//! The session is interactive:
//!
//! 1. Press **Enter** to start recording (Ctrl+C here aborts).
//! 2. Press **Enter** again — or **Ctrl+C** — to stop.
//! 3. The capture is written to `recordings/` and then played back; the program exits by itself as
//!    soon as playback finishes.
//!
//! ```text
//! cargo run --example recording              # -> recordings/rodio-<timestamp>.wav
//! cargo run --example recording -- hello.wav # -> recordings/hello.wav
//! cargo run --example recording -- --list    # list the available input devices
//! ```
//!
//! Capture prefers 16 kHz mono, exactly what Whisper wants, whenever the hardware supports it.
//! Whatever the device actually delivers, the recording is resampled to 16 kHz mono before it hits
//! the disk, so the file can be fed straight into
//! [`whisper_stt::Transcriber::transcribe_file`].

use std::num::NonZero;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use rodio::microphone::{MicrophoneBuilder, available_inputs};
use whisper_stt::WHISPER_SAMPLE_RATE;
use whisper_stt::audio::{mix_down, resample};

#[path = "shared/mod.rs"]
mod shared;
use shared::{OUTPUT_DIR, Stop, Stopper, output_path, play, write_wav};

/// Sample rate Whisper models are trained on; used if the device can deliver it natively.
const PREFERRED_SAMPLE_RATE: NonZero<u32> = NonZero::new(16_000).expect("non-zero");
/// Whisper wants a single channel.
const PREFERRED_CHANNELS: NonZero<u16> = NonZero::new(1).expect("non-zero");

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|arg| arg == "--list" || arg == "-l") {
        return list_inputs();
    }

    let output = output_path("rodio", args.first().map(String::as_str))?;

    let stopper = Stopper::arm()?;

    println!();
    println!("Output file: {}", output.display());
    println!("Press Enter to start recording (Ctrl+C to abort)...");
    if stopper.wait() == Stop::Interrupt {
        println!("Aborted, nothing recorded.");
        return Ok(());
    }

    let microphone = MicrophoneBuilder::new()
        .default_device()?
        .default_config()?
        .prefer_sample_rates([PREFERRED_SAMPLE_RATE])
        .prefer_channel_counts([PREFERRED_CHANNELS])
        .open_stream()?;

    let config = *microphone.config();
    println!(
        "Recording at {} Hz, {} channel(s) — press Enter or Ctrl+C to stop.",
        config.sample_rate.get(),
        config.channel_count.get()
    );

    // rodio's microphone has no "stop" method, so drain it on a worker thread and tell that thread
    // to finish through a flag. Samples accumulate in a shared buffer instead of being returned on
    // join, so a stalled device can never hang the main thread.
    let stop_flag = Arc::new(AtomicBool::new(false));
    let collected = Arc::new(Mutex::new(Vec::<f32>::new()));

    let worker = {
        let stop_flag = stop_flag.clone();
        let collected = collected.clone();
        thread::spawn(move || {
            let mut microphone = microphone;
            let mut chunk: Vec<f32> = Vec::with_capacity(4096);
            while !stop_flag.load(Ordering::SeqCst) {
                match microphone.next() {
                    Some(sample) => {
                        chunk.push(sample);
                        if chunk.len() >= 4096
                            && let Ok(mut buffer) = collected.lock()
                        {
                            buffer.append(&mut chunk);
                        }
                    }
                    // The stream ended on its own (device error or removal).
                    None => break,
                }
            }
            if let Ok(mut buffer) = collected.lock() {
                buffer.append(&mut chunk);
            }
        })
    };

    let stop = stopper.wait_with_progress(Some(&stop_flag), "Recording...");

    // Give the worker a moment to flush its last chunk before reading the buffer.
    thread::sleep(Duration::from_millis(150));
    let samples = collected
        .lock()
        .map(|buffer| buffer.clone())
        .unwrap_or_default();
    let _ = worker.join();

    if samples.is_empty() {
        bail!("captured no audio — is the microphone permitted and unmuted?");
    }

    // Whisper only understands 16 kHz mono.
    let mono = mix_down(&samples, config.channel_count.get() as usize);
    let whisper_input = resample(&mono, config.sample_rate.get(), WHISPER_SAMPLE_RATE);

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

    play(&output)
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
    println!("\nRecordings are written to {OUTPUT_DIR}/");
    Ok(())
}
