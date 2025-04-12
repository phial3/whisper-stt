use hf_hub::api::sync::{ApiBuilder};
use hf_hub::Cache;
use anyhow::{Context, Result};

fn main() -> Result<()> {
    let cache_dir = std::path::Path::new("models");
    let cache = Cache::new(cache_dir.to_path_buf());
    let api = ApiBuilder::from_cache(cache)
        .with_progress(true)
        .build()?;

    let repo = api.model("ggerganov/whisper.cpp".to_string());

    let filename = repo.get("ggml-tiny-encoder.mlmodelc.zip")
        .context("Failed to download model")?;

    println!("{:?}", filename.to_str());

    Ok(())
}
