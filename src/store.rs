//! Model storage: resolves where a ggml checkpoint lives and downloads it when needed.
//!
//! This is the only module that touches the network, and only when the `network` feature is on.
//! It knows nothing about transcription.

use std::path::{Path, PathBuf};

#[cfg(feature = "network")]
use std::io::Write as _;

use crate::error::{Error, Result};
use crate::model::{ModelSource, WhisperModel};

/// How far a model download has got.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    /// Bytes written to the staging file so far.
    pub written: u64,
    /// Total size of the checkpoint, when the server told us.
    pub total: Option<u64>,
}

impl Progress {
    /// Fraction of the download that is done, or `None` when the total size is unknown.
    pub fn fraction(self) -> Option<f32> {
        let total = self.total?;
        if total == 0 {
            return None;
        }
        Some((self.written as f32 / total as f32).min(1.0))
    }
}

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
    /// The response is streamed to disk rather than buffered, so a 2.9 GB checkpoint never has to
    /// fit in memory. The file is written to a `.part` staging file alongside the destination and
    /// renamed only on success, so an interrupted download never leaves a corrupt checkpoint
    /// behind.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ModelNotFound`] for [`ModelSource::File`] sources and for builds without
    /// the `network` feature, [`Error::UnexpectedStatus`] when the repository does not answer 2xx,
    /// and [`Error::Download`] or [`Error::Io`] on transport and filesystem failures.
    pub async fn download(&self) -> Result<PathBuf> {
        self.download_with(|_| {}).await
    }

    /// Downloads the model, reporting progress as it goes.
    ///
    /// Call this instead of [`Self::download`] when you want to print a progress bar or abort a
    /// download that is taking too long.
    ///
    /// # Errors
    ///
    /// The same errors as [`Self::download`].
    pub async fn download_with(&self, mut on_progress: impl FnMut(Progress)) -> Result<PathBuf> {
        let Some(url) = self.source.download_url() else {
            return Err(Error::ModelNotFound(self.path()));
        };

        #[cfg(feature = "network")]
        {
            self.ensure_dir()?;

            let mut response = reqwest::get(&url).await?;
            let status = response.status();
            if !status.is_success() {
                return Err(Error::UnexpectedStatus {
                    url,
                    status: status.as_u16(),
                });
            }

            let destination = self.path();
            let temporary = temp_path_for(&destination);
            let mut file = std::fs::File::create(&temporary)?;
            let total = response.content_length();
            let mut written = 0u64;

            // One chunk at a time: the checkpoint is far larger than anyone's spare RAM.
            while let Some(chunk) = response.chunk().await? {
                file.write_all(&chunk)?;
                written += chunk.len() as u64;
                on_progress(Progress { written, total });
            }
            file.flush()?;
            // Some platforms refuse to rename an open file.
            drop(file);

            std::fs::rename(&temporary, &destination)?;
            Ok(destination)
        }

        #[cfg(not(feature = "network"))]
        {
            // Silence the unused-binding warning; the URL is still what `ensure` would fetch.
            let _ = (&url, &mut on_progress);
            Err(Error::ModelNotFound(self.path()))
        }
    }
}

impl AsRef<Path> for ModelStore {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

/// Returns `<destination>.part`, used as the staging file for downloads.
#[cfg(feature = "network")]
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

    #[cfg(feature = "network")]
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

    #[test]
    fn progress_fraction_is_clamped_and_total_aware() {
        let known = Progress {
            written: 500,
            total: Some(1_000),
        };
        assert_eq!(known.fraction(), Some(0.5));

        // A server that omits Content-Length leaves the fraction unknown rather than zero.
        let unknown = Progress {
            written: 500,
            total: None,
        };
        assert_eq!(unknown.fraction(), None);

        let empty = Progress {
            written: 0,
            total: Some(0),
        };
        assert_eq!(empty.fraction(), None);
    }
}
