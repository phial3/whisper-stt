//! Model storage: resolves where a ggml checkpoint lives and downloads it when needed.
//!
//! This is the only module that touches the network. It knows nothing about transcription.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::model::{ModelSource, WhisperModel};

/// Manages the on-disk location of a Whisper model.
///
/// A store is cheap and does no I/O when constructed. Call [`ModelStore::ensure`] to guarantee the
/// checkpoint is present on disk, which downloads it on first use.
///
/// # Examples
///
/// ```no_run
/// # async fn example() -> whisper_stt::Result<()> {
/// use whisper_stt::{ModelStore, WhisperModel};
///
/// let store = ModelStore::pretrained(WhisperModel::Tiny, "models");
/// let path = store.ensure().await?;
/// # Ok(()) }
/// ```
#[derive(Debug, Clone)]
pub struct ModelStore {
    source: ModelSource,
    dir: PathBuf,
    path: PathBuf,
}

impl ModelStore {
    /// Creates a store for any [`ModelSource`].
    pub fn new(source: ModelSource, dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        let path = source.path_in(&dir);
        Self { source, dir, path }
    }

    /// Creates a store for one of the published checkpoints.
    pub fn pretrained(model: WhisperModel, dir: impl Into<PathBuf>) -> Self {
        Self::new(ModelSource::Pretrained(model), dir)
    }

    /// Creates a store around an existing local ggml file. Nothing is ever downloaded.
    pub fn file(path: impl Into<PathBuf>, dir: impl Into<PathBuf>) -> Self {
        Self::new(ModelSource::File(path.into()), dir)
    }

    /// Resolves a model name (see [`WhisperModel::from_name`]) into a store.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownModel`] when the name does not match any checkpoint.
    pub fn from_name(name: &str, dir: impl Into<PathBuf>) -> Result<Self> {
        let model =
            WhisperModel::from_name(name).ok_or_else(|| Error::UnknownModel(name.to_string()))?;
        Ok(Self::pretrained(model, dir))
    }

    /// Creates a store for a models directory that already contains the given file.
    ///
    /// Unlike [`Self::from_name`], this accepts quantized and fine-tuned checkpoints that are not
    /// part of the official catalogue, as long as the file exists on disk.
    pub fn from_file_in_dir(name: &str, dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        Self::new(ModelSource::File(dir.join(name)), dir)
    }

    /// The source this store was created for.
    pub fn source(&self) -> &ModelSource {
        &self.source
    }

    /// The models directory backing this store.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Full path this store resolves to.
    pub fn path(&self) -> PathBuf {
        self.path.clone()
    }

    /// Whether the model file already exists on disk.
    pub fn exists(&self) -> bool {
        self.path().is_file()
    }

    /// Creates the models directory if it is missing.
    pub fn ensure_dir(&self) -> Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        Ok(())
    }

    /// Returns the local path to the model, downloading it first if necessary.
    ///
    /// For [`ModelSource::File`] sources nothing is downloaded; a missing file yields
    /// [`Error::ModelNotFound`].
    ///
    /// # Errors
    ///
    /// Propagates download and filesystem errors.
    pub async fn ensure(&self) -> Result<PathBuf> {
        let path = self.path();
        if path.is_file() {
            return Ok(path);
        }
        self.download().await
    }

    /// Downloads the model into the models directory and returns its path.
    ///
    /// The file is written to a `.part` temporary alongside the destination and renamed only on
    /// success, so an interrupted download never leaves a corrupt checkpoint behind.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ModelNotFound`] for [`ModelSource::File`] sources, [`Error::Download`] on
    /// transport failures and [`Error::UnexpectedStatus`] when the repository does not answer 2xx.
    pub async fn download(&self) -> Result<PathBuf> {
        let Some(url) = self.source.download_url() else {
            return Err(Error::ModelNotFound(self.path()));
        };

        self.ensure_dir()?;

        println!("Downloading model from: {url}");
        let response = reqwest::get(&url).await?;
        let status = response.status();
        if !status.is_success() {
            return Err(Error::UnexpectedStatus { url, status });
        }

        let bytes = response.bytes().await?;

        let destination = self.path();
        let temporary = temp_path_for(&destination);
        std::fs::write(&temporary, &bytes)?;
        std::fs::rename(&temporary, &destination)?;

        Ok(destination)
    }
}

impl AsRef<Path> for ModelStore {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

/// Returns `<destination>.part`, used as the staging file for downloads.
fn temp_path_for(destination: &Path) -> PathBuf {
    let mut file_name = destination
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default();
    file_name.push_str(".part");
    destination.with_file_name(file_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretrained_paths_land_in_the_models_dir() {
        let store = ModelStore::pretrained(WhisperModel::Base, "models");
        assert_eq!(store.dir(), Path::new("models"));
        assert_eq!(store.path(), PathBuf::from("models/ggml-base.bin"));
    }

    #[test]
    fn from_name_resolves_every_checkpoint() {
        for model in WhisperModel::all() {
            let store = ModelStore::from_name(model.id(), "models").unwrap();
            assert_eq!(
                store.path(),
                PathBuf::from("models").join(model.file_name())
            );
        }
    }

    #[test]
    fn unknown_names_are_reported() {
        let err = ModelStore::from_name("nope", "models").unwrap_err();
        assert!(matches!(err, Error::UnknownModel(name) if name == "nope"));
    }

    #[test]
    fn file_sources_are_absolute_paths() {
        let store = ModelStore::file("/tmp/ggml-custom.bin", "models");
        assert_eq!(store.path(), PathBuf::from("/tmp/ggml-custom.bin"));
    }

    #[test]
    fn temp_path_appends_part() {
        let destination = PathBuf::from("models/ggml-tiny.bin");
        assert_eq!(
            temp_path_for(&destination),
            PathBuf::from("models/ggml-tiny.bin.part")
        );
    }

    #[test]
    fn local_files_are_not_downloaded() {
        // A local source has no download URL, so `download` must refuse rather than hit the
        // network.
        let store = ModelStore::file("/tmp/does-not-exist.bin", "models");
        assert!(store.source().download_url().is_none());
        assert!(!store.exists());
    }
}
