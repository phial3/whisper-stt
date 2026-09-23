use anyhow::{Context, Result};
use hf_hub::{HFClient, split_id};

#[tokio::main]
async fn main() -> Result<()> {
    let client = HFClient::builder()
        .cache_dir(std::path::PathBuf::from("models"))
        .build()?;

    let (owner, name) = split_id("ggerganov/whisper.cpp");
    let path = client
        .model(owner, name)
        .download_file()
        .filename("ggml-tiny-encoder.mlmodelc.zip")
        .send()
        .await
        .context("Failed to download model")?;

    println!("{:?}", path.to_str());

    Ok(())
}
