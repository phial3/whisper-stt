//! Catalogue of the Whisper checkpoints supported by whisper-rs.
//!
//! This module is pure data: it never touches the network or the filesystem. It maps a
//! human-friendly model alias (`"tiny"`, `"large-v3-turbo"`, `"medium.en"`, ...) to the ggml
//! checkpoint published in the [whisper.cpp model repository][REPO] and to the extras whisper-rs
//! exposes for it, such as its [`DtwModelPreset`].
//!
//! The catalogue mirrors `whisper_rs::DtwModelPreset`, which is whisper-rs' own enumeration of the
//! checkpoints it knows how to work with, so both lists always agree.
//!
//! [REPO]: https://huggingface.co/ggerganov/whisper.cpp

use std::fmt;
use std::path::{Path, PathBuf};

use whisper_rs::DtwModelPreset;

/// Base URL of the whisper.cpp model repository.
pub const MODEL_REPO_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/// Sample rate Whisper models expect, in Hz.
///
/// Every checkpoint is trained on 16 kHz mono audio, so anything fed into whisper-rs must be
/// resampled to this rate first. See [`crate::audio`].
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Every Whisper checkpoint whisper-rs supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum WhisperModel {
    /// Multilingual `tiny` (~75 MB).
    Tiny,
    /// English-only `tiny.en` (~75 MB).
    TinyEn,
    /// Multilingual `base` (~142 MB).
    Base,
    /// English-only `base.en` (~142 MB).
    BaseEn,
    /// Multilingual `small` (~466 MB).
    Small,
    /// English-only `small.en` (~466 MB).
    SmallEn,
    /// Multilingual `medium` (~1.5 GB).
    Medium,
    /// English-only `medium.en` (~1.5 GB).
    MediumEn,
    /// Multilingual `large-v1` (~2.9 GB).
    LargeV1,
    /// Multilingual `large-v2` (~2.9 GB).
    LargeV2,
    /// Multilingual `large-v3` (~2.9 GB). The default when asking for plain `"large"`.
    LargeV3,
    /// Multilingual `large-v3-turbo` (~1.6 GB). Distilled `large-v3`, much faster.
    LargeV3Turbo,
}

/// Flat list of every [`WhisperModel`], ordered from the smallest to the largest checkpoint.
pub const WHISPER_MODELS: [WhisperModel; 12] = [
    WhisperModel::Tiny,
    WhisperModel::TinyEn,
    WhisperModel::Base,
    WhisperModel::BaseEn,
    WhisperModel::Small,
    WhisperModel::SmallEn,
    WhisperModel::Medium,
    WhisperModel::MediumEn,
    WhisperModel::LargeV1,
    WhisperModel::LargeV2,
    WhisperModel::LargeV3,
    WhisperModel::LargeV3Turbo,
];

/// Where a model comes from: one of the published checkpoints, or an arbitrary local ggml file.
///
/// The local variant covers quantized releases (`ggml-large-v3-turbo-q5_0.bin`), fine-tunes, or
/// models exported by other tools — anything whisper-rs can load that is not part of the official
/// catalogue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// One of the [`WhisperModel`] checkpoints, downloaded from the whisper.cpp repository.
    Pretrained(WhisperModel),
    /// An already existing local file. Nothing is downloaded for this source.
    File(PathBuf),
}

impl ModelSource {
    /// File stem used to build the downloaded file name, without extension.
    pub fn file_stem(&self) -> String {
        match self {
            ModelSource::Pretrained(model) => model.file_stem().to_string(),
            ModelSource::File(path) => path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| String::from("model")),
        }
    }

    /// File name (with extension) of this model inside the models directory.
    pub fn file_name(&self) -> String {
        match self {
            ModelSource::Pretrained(model) => model.file_name(),
            ModelSource::File(path) => path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| String::from("model.bin")),
        }
    }

    /// Download URL of this model, or `None` for [`ModelSource::File`].
    pub fn download_url(&self) -> Option<String> {
        match self {
            ModelSource::Pretrained(model) => Some(model.download_url()),
            ModelSource::File(_) => None,
        }
    }

    /// Resolves this source inside `dir`.
    pub fn path_in(&self, dir: impl AsRef<Path>) -> PathBuf {
        match self {
            ModelSource::File(path) => path.clone(),
            ModelSource::Pretrained(_) => dir.as_ref().join(self.file_name()),
        }
    }
}

impl From<WhisperModel> for ModelSource {
    fn from(model: WhisperModel) -> Self {
        ModelSource::Pretrained(model)
    }
}

impl From<PathBuf> for ModelSource {
    fn from(path: PathBuf) -> Self {
        ModelSource::File(path)
    }
}

impl WhisperModel {
    /// Every known checkpoint, smallest first.
    pub const fn all() -> &'static [WhisperModel] {
        &WHISPER_MODELS
    }

    /// Canonical id of this model, e.g. `"large-v3-turbo"`.
    pub const fn id(self) -> &'static str {
        match self {
            WhisperModel::Tiny => "tiny",
            WhisperModel::TinyEn => "tiny.en",
            WhisperModel::Base => "base",
            WhisperModel::BaseEn => "base.en",
            WhisperModel::Small => "small",
            WhisperModel::SmallEn => "small.en",
            WhisperModel::Medium => "medium",
            WhisperModel::MediumEn => "medium.en",
            WhisperModel::LargeV1 => "large-v1",
            WhisperModel::LargeV2 => "large-v2",
            WhisperModel::LargeV3 => "large-v3",
            WhisperModel::LargeV3Turbo => "large-v3-turbo",
        }
    }

    /// File stem of the ggml checkpoint, without extension, e.g. `"ggml-large-v3-turbo"`.
    pub const fn file_stem(self) -> &'static str {
        match self {
            WhisperModel::Tiny => "ggml-tiny",
            WhisperModel::TinyEn => "ggml-tiny.en",
            WhisperModel::Base => "ggml-base",
            WhisperModel::BaseEn => "ggml-base.en",
            WhisperModel::Small => "ggml-small",
            WhisperModel::SmallEn => "ggml-small.en",
            WhisperModel::Medium => "ggml-medium",
            WhisperModel::MediumEn => "ggml-medium.en",
            WhisperModel::LargeV1 => "ggml-large-v1",
            WhisperModel::LargeV2 => "ggml-large-v2",
            WhisperModel::LargeV3 => "ggml-large-v3",
            WhisperModel::LargeV3Turbo => "ggml-large-v3-turbo",
        }
    }

    /// File name of the ggml checkpoint in the whisper.cpp repository, e.g. `"ggml-tiny.bin"`.
    pub fn file_name(self) -> String {
        format!("{}.bin", self.file_stem())
    }

    /// Direct download URL of the ggml checkpoint.
    pub fn download_url(self) -> String {
        format!("{}/{}.bin", MODEL_REPO_URL, self.file_stem())
    }

    /// Aliases accepted for this model. The first entry is always [`Self::id`].
    pub fn aliases(self) -> &'static [&'static str] {
        match self {
            WhisperModel::Tiny => &["tiny", "tiny-multi", "whisper-tiny"],
            WhisperModel::TinyEn => &["tiny.en", "tiny-en", "tinyen", "tiny-english"],
            WhisperModel::Base => &["base", "base-multi", "whisper-base"],
            WhisperModel::BaseEn => &["base.en", "base-en", "baseen", "base-english"],
            WhisperModel::Small => &["small", "small-multi", "whisper-small"],
            WhisperModel::SmallEn => &["small.en", "small-en", "smallen", "small-english"],
            WhisperModel::Medium => &["medium", "medium-multi", "whisper-medium"],
            WhisperModel::MediumEn => &["medium.en", "medium-en", "mediumen", "medium-english"],
            WhisperModel::LargeV1 => &["large-v1", "largev1", "large1"],
            WhisperModel::LargeV2 => &["large-v2", "largev2", "large2"],
            WhisperModel::LargeV3 => &["large-v3", "largev3", "large3", "large"],
            WhisperModel::LargeV3Turbo => {
                &["large-v3-turbo", "largev3turbo", "large-turbo", "turbo"]
            }
        }
    }

    /// `true` for multilingual checkpoints, `false` for the English-only `*.en` variants.
    pub const fn is_multilingual(self) -> bool {
        !matches!(
            self,
            WhisperModel::TinyEn
                | WhisperModel::BaseEn
                | WhisperModel::SmallEn
                | WhisperModel::MediumEn
        )
    }

    /// Whether this checkpoint can translate speech into English.
    ///
    /// Translation is only available for multilingual models.
    pub const fn supports_translation(self) -> bool {
        self.is_multilingual()
    }

    /// The whisper-rs DTW alignment preset matching this checkpoint.
    pub const fn dtw_preset(self) -> DtwModelPreset {
        match self {
            WhisperModel::Tiny => DtwModelPreset::Tiny,
            WhisperModel::TinyEn => DtwModelPreset::TinyEn,
            WhisperModel::Base => DtwModelPreset::Base,
            WhisperModel::BaseEn => DtwModelPreset::BaseEn,
            WhisperModel::Small => DtwModelPreset::Small,
            WhisperModel::SmallEn => DtwModelPreset::SmallEn,
            WhisperModel::Medium => DtwModelPreset::Medium,
            WhisperModel::MediumEn => DtwModelPreset::MediumEn,
            WhisperModel::LargeV1 => DtwModelPreset::LargeV1,
            WhisperModel::LargeV2 => DtwModelPreset::LargeV2,
            WhisperModel::LargeV3 => DtwModelPreset::LargeV3,
            WhisperModel::LargeV3Turbo => DtwModelPreset::LargeV3Turbo,
        }
    }

    /// The checkpoint a whisper-rs DTW alignment preset refers to.
    ///
    /// The inverse of [`Self::dtw_preset`]. The match is exhaustive on purpose: the moment
    /// whisper-rs adds a checkpoint to `DtwModelPreset`, this stops compiling instead of leaving
    /// the catalogue quietly one model short.
    pub const fn from_dtw_preset(preset: DtwModelPreset) -> Self {
        match preset {
            DtwModelPreset::Tiny => WhisperModel::Tiny,
            DtwModelPreset::TinyEn => WhisperModel::TinyEn,
            DtwModelPreset::Base => WhisperModel::Base,
            DtwModelPreset::BaseEn => WhisperModel::BaseEn,
            DtwModelPreset::Small => WhisperModel::Small,
            DtwModelPreset::SmallEn => WhisperModel::SmallEn,
            DtwModelPreset::Medium => WhisperModel::Medium,
            DtwModelPreset::MediumEn => WhisperModel::MediumEn,
            DtwModelPreset::LargeV1 => WhisperModel::LargeV1,
            DtwModelPreset::LargeV2 => WhisperModel::LargeV2,
            DtwModelPreset::LargeV3 => WhisperModel::LargeV3,
            DtwModelPreset::LargeV3Turbo => WhisperModel::LargeV3Turbo,
        }
    }

    /// Resolves a user-supplied model name to a checkpoint.
    ///
    /// Matching is case-insensitive and tolerant: a leading `ggml-`, a trailing `.bin`,
    /// underscores, and spaces are all ignored, so `"ggml-tiny.en.bin"`, `"tiny_en"`, and
    /// `"tiny.en"` all resolve to [`WhisperModel::TinyEn`].
    ///
    /// Plain `"large"` maps to [`WhisperModel::LargeV3`], keeping the historical behaviour of this
    /// crate.
    ///
    /// Returns `None` when nothing matches.
    pub fn from_name(name: &str) -> Option<Self> {
        let normalized = normalize_name(name);
        if normalized.is_empty() {
            return None;
        }
        WHISPER_MODELS
            .iter()
            .copied()
            .find(|model| model.aliases().contains(&normalized.as_str()))
    }
}

impl fmt::Display for WhisperModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// Normalizes a user-supplied model name for alias matching.
fn normalize_name(name: &str) -> String {
    let lowered = name.trim().to_ascii_lowercase();
    let without_extension = lowered.strip_suffix(".bin").unwrap_or(lowered.as_str());
    let without_prefix = without_extension
        .strip_prefix("ggml-")
        .unwrap_or(without_extension);
    without_prefix.replace(['_', ' '], "-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_round_trips_dtw_presets() {
        // Every model whisper-rs can align against must be present in the catalogue, otherwise the
        // two lists silently drift apart. `from_dtw_preset` matching exhaustively is what turns
        // that drift into a compile error rather than a missing entry.
        for model in WhisperModel::all() {
            assert_eq!(WhisperModel::from_dtw_preset(model.dtw_preset()), *model);
        }
        assert_eq!(WhisperModel::all().len(), 12);
    }

    #[test]
    fn english_only_variants_are_not_multilingual() {
        for model in [
            WhisperModel::TinyEn,
            WhisperModel::BaseEn,
            WhisperModel::SmallEn,
            WhisperModel::MediumEn,
        ] {
            assert!(!model.is_multilingual());
            assert!(!model.supports_translation());
        }
        for model in [WhisperModel::Tiny, WhisperModel::LargeV3Turbo] {
            assert!(model.is_multilingual());
            assert!(model.supports_translation());
        }
    }

    #[test]
    fn file_names_follow_repository_layout() {
        assert_eq!(WhisperModel::Tiny.file_name(), "ggml-tiny.bin");
        assert_eq!(WhisperModel::TinyEn.file_name(), "ggml-tiny.en.bin");
        assert_eq!(
            WhisperModel::LargeV3Turbo.file_name(),
            "ggml-large-v3-turbo.bin"
        );
        assert_eq!(
            WhisperModel::LargeV3Turbo.download_url(),
            format!("{}/ggml-large-v3-turbo.bin", MODEL_REPO_URL)
        );
    }

    #[test]
    fn alias_lookup_is_tolerant() {
        for expected in WhisperModel::all() {
            for alias in expected.aliases() {
                assert_eq!(
                    WhisperModel::from_name(alias),
                    Some(*expected),
                    "alias {alias} did not resolve to {expected}"
                );
                assert_eq!(
                    WhisperModel::from_name(&alias.to_uppercase()),
                    Some(*expected)
                );
                assert_eq!(
                    WhisperModel::from_name(&format!("ggml-{alias}.bin")),
                    Some(*expected)
                );
            }
        }
    }

    #[test]
    fn legacy_names_still_resolve() {
        assert_eq!(WhisperModel::from_name("Tiny"), Some(WhisperModel::Tiny));
        assert_eq!(
            WhisperModel::from_name("large"),
            Some(WhisperModel::LargeV3)
        );
        assert_eq!(
            WhisperModel::from_name("  medium.en  "),
            Some(WhisperModel::MediumEn)
        );
        assert_eq!(WhisperModel::from_name("does-not-exist"), None);
        assert_eq!(WhisperModel::from_name(""), None);
    }

    #[test]
    fn model_source_paths() {
        let source = ModelSource::from(WhisperModel::Medium);
        assert_eq!(source.file_name(), "ggml-medium.bin");
        assert_eq!(
            source.path_in("models"),
            PathBuf::from("models/ggml-medium.bin")
        );

        let local = ModelSource::from(PathBuf::from("/tmp/custom-q5.bin"));
        assert_eq!(local.file_name(), "custom-q5.bin");
        assert_eq!(local.download_url(), None);
        assert_eq!(local.path_in("models"), PathBuf::from("/tmp/custom-q5.bin"));
    }
}
