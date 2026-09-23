//! Transcribes Mandarin audio, forcing the source language.
//!
//! ```text
//! cargo run --example usage_example_chinese                       # `tiny`, assets/test.mp3
//! cargo run --example usage_example_chinese -- path/to/zh.wav     # your own clip
//! cargo run --example usage_example_chinese -- --model large-v3 zh.wav
//! ```
//!
//! The default model is `tiny` (~75 MB). `large-v3` (~2.9 GB) is markedly better at Mandarin but a
//! much heavier download, so pass `--model large-v3` when you want it.
//!
//! Tip: `cargo run --example record_cpal -- 5 zh.wav` records a clip you can feed straight in.

use anyhow::{Result, bail};
use whisper_stt::{ModelStore, Transcriber, TranscriptionOptions, WHISPER_SAMPLE_RATE};

/// Clip transcribed when no path is given.
const DEFAULT_AUDIO: &str = "assets/test.mp3";

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let model_name = args
        .iter()
        .position(|arg| arg == "--model")
        .and_then(|index| args.get(index + 1))
        .map(String::as_str)
        .unwrap_or("tiny");
    let audio_path = args
        .iter()
        .find(|arg| !arg.starts_with('-') && *arg != model_name)
        .map(String::as_str)
        .unwrap_or(DEFAULT_AUDIO);

    if !std::path::Path::new(audio_path).exists() {
        bail!("audio file not found: {audio_path}");
    }

    let store = ModelStore::from_name(model_name, "models")?;
    store.ensure().await?;

    let transcriber = Transcriber::new(store.path())?;

    // Naming the language avoids mis-detection on short clips. Audio is resampled to 16 kHz mono
    // (see WHISPER_SAMPLE_RATE) before it reaches the model, which matters most for non-English.
    let options = TranscriptionOptions {
        language: Some("zh"),
        ..TranscriptionOptions::default()
    };

    let result = transcriber.transcribe_file(audio_path, Some(&options))?;

    println!("model:   {}", store.source().file_name());
    println!("sample rate expected: {WHISPER_SAMPLE_RATE} Hz mono");
    println!("language: {}", options.language.unwrap_or("auto"));
    println!(
        "start[{}]-end[{}] {}",
        result.start_timestamp(),
        result.end_timestamp(),
        result.text()
    );

    Ok(())
}
