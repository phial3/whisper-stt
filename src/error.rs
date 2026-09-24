//! Crate-wide error type.
//!
//! Every fallible operation in this crate returns [`Result<T>`], i.e.
//! `std::result::Result<T, `[`Error`]`>`.

use std::fmt;
use std::path::PathBuf;

use whisper_rs::WhisperError;

/// Convenience alias for a result carrying this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Errors returned by this crate.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A model alias or file name does not match any known Whisper checkpoint.
    UnknownModel(String),
    /// The model file could not be found on disk.
    ///
    /// Also returned when the model would have to be downloaded but the crate was built without
    /// the `network` feature.
    ModelNotFound(PathBuf),
    /// Downloading a model from the model repository failed.
    #[cfg(feature = "network")]
    Download(reqwest::Error),
    /// The model repository answered with a non-success HTTP status.
    UnexpectedStatus {
        /// The URL that was requested.
        url: String,
        /// HTTP status returned by the server.
        status: u16,
    },
    /// Whisper returned an error while loading a model or running transcription.
    Whisper(WhisperError),
    /// Reading, writing, creating directories, etc. failed.
    Io(std::io::Error),
    /// The audio file could not be opened, probed, or contains no audio track.
    Audio(String),
    /// The media file contains no playable audio track.
    NoAudioTrack,
    /// The audio file decoded to zero samples.
    EmptyAudio,
    /// Converting audio to the sample rate Whisper needs failed.
    Resample(String),
    /// Translation to English was requested but the loaded model is English-only.
    TranslationUnsupported,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnknownModel(name) => {
                write!(f, "unknown whisper model '{name}'")
            }
            Error::ModelNotFound(path) => {
                write!(f, "model file not found at {}", path.display())
            }
            #[cfg(feature = "network")]
            Error::Download(err) => write!(f, "model download failed: {err}"),
            Error::UnexpectedStatus { url, status } => {
                write!(f, "request to {url} failed with status {status}")
            }
            Error::Whisper(err) => write!(f, "whisper error: {err}"),
            Error::Io(err) => write!(f, "io error: {err}"),
            Error::Audio(msg) => write!(f, "audio decoding failed: {msg}"),
            Error::NoAudioTrack => write!(f, "the media file has no audio track"),
            Error::EmptyAudio => write!(f, "the audio file decoded to zero samples"),
            Error::Resample(msg) => write!(f, "resampling failed: {msg}"),
            Error::TranslationUnsupported => write!(
                f,
                "translation requires a multilingual model, the loaded model is English-only"
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            #[cfg(feature = "network")]
            Error::Download(err) => Some(err),
            Error::Whisper(err) => Some(err),
            Error::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<WhisperError> for Error {
    fn from(err: WhisperError) -> Self {
        Error::Whisper(err)
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

#[cfg(feature = "network")]
impl From<reqwest::Error> for Error {
    fn from(err: reqwest::Error) -> Self {
        Error::Download(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_has_a_message() {
        let cases = [
            Error::UnknownModel("nope".into()),
            Error::ModelNotFound(PathBuf::from("models/ggml-tiny.bin")),
            Error::UnexpectedStatus {
                url: "https://example.invalid/a.bin".into(),
                status: 404,
            },
            Error::Whisper(WhisperError::NoSamples),
            Error::Io(std::io::Error::other("boom")),
            Error::Audio("bad file".into()),
            Error::NoAudioTrack,
            Error::EmptyAudio,
            Error::Resample("bad rate".into()),
            Error::TranslationUnsupported,
        ];
        for error in cases {
            assert!(!error.to_string().is_empty());
        }
    }

    #[test]
    fn io_errors_are_sourced() {
        let error = Error::from(std::io::Error::other("boom"));
        assert!(std::error::Error::source(&error).is_some());
    }
}
