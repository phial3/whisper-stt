//! Backwards-compatible wrapper around [`crate::store::ModelStore`].
//!
//! New code should reach for [`ModelStore`] directly — [`WhisperModel`] gives compile-time
//! guarantees that a string model name cannot. This module exists so older integrations keep
//! compiling; it preserves the historical shape of the old API, including the fact that
//! [`ModelHandler::new`] panics on an unknown model name.

use std::path::{Path, PathBuf};

use crate::error::Result;
use crate::model::WhisperModel;
use crate::store::ModelStore;

/// Thin shim around a [`ModelStore`] that keeps the pre-refactor `ModelHandler` API alive.
#[derive(Debug, Clone)]
pub struct ModelHandler {
    store: ModelStore,
}

impl ModelHandler {
    /// Resolves a model name and makes sure the checkpoint is downloaded into `models_dir`.
    ///
    /// The name is resolved with [`WhisperModel::from_name`], so all of `"tiny"`, `"Tiny"`,
    /// `"medium.en"`, `"large-v3-turbo"`, and `"large"` work.
    ///
    /// # Panics
    ///
    /// Panics when `model_name` does not match any known checkpoint. Use
    /// [`ModelHandler::try_new`] to get a recoverable [`crate::Error::UnknownModel`] instead.
    pub async fn new(model_name: &str, models_dir: &str) -> ModelHandler {
        Self::try_new(model_name, models_dir)
            .await
            .expect("failed to prepare model")
    }

    /// Non-panicking variant of [`ModelHandler::new`].
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::UnknownModel`] for unresolvable names and propagates download
    /// failures.
    pub async fn try_new(model_name: &str, models_dir: &str) -> Result<ModelHandler> {
        let store = ModelStore::from_name(model_name, models_dir)?;
        store.ensure().await?;
        Ok(ModelHandler { store })
    }

    /// The underlying store.
    pub fn store(&self) -> &ModelStore {
        &self.store
    }

    /// The checkpoint this handler was created for.
    pub fn model(&self) -> WhisperModel {
        match self.store.source() {
            crate::model::ModelSource::Pretrained(model) => *model,
            crate::model::ModelSource::File(_) => {
                unreachable!("handlers are always created from the catalogue")
            }
        }
    }

    /// Path of the model file, as a `String`.
    pub fn get_model_dir(&self) -> String {
        self.store.path().to_string_lossy().to_string()
    }

    /// Path of the model file.
    pub fn path(&self) -> PathBuf {
        self.store.path()
    }
}

impl AsRef<Path> for ModelHandler {
    fn as_ref(&self) -> &Path {
        self.store.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_DIR: &str = "test_models/";

    #[test]
    fn name_resolution_matches_the_catalogue() {
        // `try_new` downloads, so only assert the synchronous parts here: resolution and paths.
        for model in WhisperModel::all() {
            let store = ModelStore::from_name(model.id(), MODEL_DIR).unwrap();
            assert_eq!(
                store.path(),
                PathBuf::from(MODEL_DIR).join(model.file_name()),
                "unexpected path for {model}"
            );
        }
    }

    #[test]
    fn unknown_names_are_rejected() {
        let err = ModelStore::from_name("nope", MODEL_DIR).unwrap_err();
        assert!(matches!(err, crate::Error::UnknownModel(name) if name == "nope"));
    }

    #[test]
    fn handler_exposes_the_store_path() {
        let store = ModelStore::from_name("Tiny", MODEL_DIR).unwrap();
        let handler = ModelHandler { store };
        assert_eq!(handler.get_model_dir(), format!("{MODEL_DIR}ggml-tiny.bin"));
        assert_eq!(handler.model(), WhisperModel::Tiny);
        assert_eq!(
            handler.path(),
            PathBuf::from(format!("{MODEL_DIR}ggml-tiny.bin"))
        );
        assert_eq!(handler.as_ref(), handler.store().as_ref());
    }

    #[test]
    fn legacy_large_alias_still_means_large_v3() {
        let handler = ModelHandler {
            store: ModelStore::from_name("large", MODEL_DIR).unwrap(),
        };
        assert_eq!(handler.model(), WhisperModel::LargeV3);
        assert_eq!(
            handler.get_model_dir(),
            format!("{MODEL_DIR}ggml-large-v3.bin")
        );
    }
}
