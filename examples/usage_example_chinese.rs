//! Transcribes Mandarin audio with `large-v3`, forcing the source language.
//!
//! `cargo run --example usage_example_chinese`
//!
//! Note that `large-v3` is a ~2.9 GB download. Swap in [`WhisperModel::Tiny`] for a much cheaper
//! run with worse accuracy.

use anyhow::Result;
use whisper_stt::{ModelStore, Transcriber, TranscriptionOptions, WhisperModel};

#[tokio::main]
async fn main() -> Result<()> {
    let store = ModelStore::pretrained(WhisperModel::LargeV3, "models");
    store.ensure().await?;

    let transcriber = Transcriber::new(store.path())?;

    // Naming the language avoids mis-detection on short clips; Whisper internally resamples to
    // 16 kHz mono, which for non-English audio gives markedly better results.
    let options = TranscriptionOptions {
        language: Some("zh"),
        ..TranscriptionOptions::default()
    };

    let result = transcriber.transcribe_file("assets/gongxifachai.mp3", Some(&options))?;

    println!("detected language: {}", options.language.unwrap_or("auto"));
    println!(
        "start[{}]-end[{}] {}",
        result.start_timestamp(),
        result.end_timestamp(),
        result.text()
    );

    Ok(())
}
