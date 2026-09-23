//! Transcribes an audio file end to end: download the model once, then run it.
//!
//! `cargo run --example usage_example`

use anyhow::Result;
use whisper_stt::{ModelStore, Transcriber, WhisperModel};

#[tokio::main]
async fn main() -> Result<()> {
    // Downloads `ggml-tiny.bin` into `models/` on first run and reuses it afterwards.
    let store = ModelStore::pretrained(WhisperModel::Tiny, "models");
    store.ensure().await?;

    let transcriber = Transcriber::new(store.path())?;

    // Decoding, mono downmixing and resampling to 16 kHz all happen inside `transcribe_file`.
    let result = transcriber.transcribe_file("assets/test.mp3", None)?;

    println!(
        "start[{}]-end[{}] {}",
        result.start_timestamp(),
        result.end_timestamp(),
        result.text()
    );

    for segment in result.segments() {
        println!(
            "  [{} ms - {} ms] {}",
            segment.start_ms(),
            segment.end_ms(),
            segment.text.trim()
        );
    }

    Ok(())
}
