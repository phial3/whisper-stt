//! Shows the model catalogue, or downloads the checkpoint given on the command line.
//!
//! ```text
//! cargo run --example models              # list every supported checkpoint
//! cargo run --example models -- tiny.en   # download that checkpoint into models/
//! ```

use anyhow::{Context, Result};
use whisper_stt::{ModelStore, WhisperModel};

#[tokio::main]
async fn main() -> Result<()> {
    let requested = std::env::args().nth(1);

    let Some(name) = requested else {
        println!("Supported checkpoints ({}):", WhisperModel::all().len());
        for model in WhisperModel::all() {
            let kind = if model.is_multilingual() {
                "multilingual"
            } else {
                "english-only"
            };
            println!(
                "  {:<16} {:<24} {:<14} {}",
                model.id(),
                model.file_name(),
                kind,
                model.download_url()
            );
        }
        println!(
            "\nRun with a checkpoint name to download it, e.g. `cargo run --example models -- tiny`"
        );
        return Ok(());
    };

    let store = ModelStore::from_name(&name, "models")
        .with_context(|| format!("unknown model '{name}'"))?;
    let path = store
        .ensure()
        .await
        .with_context(|| format!("failed to download {name}"))?;

    println!("downloaded {name} to {}", path.display());

    Ok(())
}
