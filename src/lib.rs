//! Simple speech-to-text for Rust, powered by [whisper-rs](https://crates.io/crates/whisper-rs).
//!
//! The crate hides the three things that make whisper.cpp fiddly: getting a model onto disk,
//! getting audio into the shape Whisper expects, and reading results back out.
//!
//! # Architecture
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`model`] | Catalogue of every Whisper checkpoint whisper-rs supports. Pure data. |
//! | [`store`] | Where a model lives on disk, and how to fetch it. Gated on `network`. |
//! | [`audio`] | Decoding media files into mono 16 kHz `f32`, what Whisper is trained on. |
//! | [`transcriber`] | Loading a model and running it, once or in a reusable session. |
//! | [`error`] | The one error type every fallible call returns. |
//!
//! # Supported models
//!
//! All twelve checkpoints whisper-rs knows about are available:
//! `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`,
//! `large-v1`, `large-v2`, `large-v3`, and `large-v3-turbo`. See [`WhisperModel`].
//!
//! # Feature flags
//!
//! | Feature | Default | Enables |
//! |---|---|---|
//! | `network` | yes | [`ModelStore::ensure`] and [`ModelStore::download`], via reqwest |
//! | `mp3` | yes | MPEG audio |
//! | `aac` | yes | AAC, as found in `.m4a` and `.mp4` |
//! | `flac` | yes | FLAC |
//! | `vorbis`, `ogg` | yes | Vorbis and the Ogg container |
//! | `mkv`, `isomp4` | yes | Matroska/WebM and MP4/MOV containers |
//! | `wav`, `pcm` | yes | Uncompressed PCM and the WAVE container |
//! | `aiff`, `caf`, `alac`, `adpcm` | no | The remaining Symphonia codecs |
//! | `metadata` | yes | ID3v1, ID3v2 and APE tag readers |
//!
//! Turn off what you do not need — a build with `default-features = false, features = ["wav",
//! "pcm"]` decodes WAV and nothing else, and drops reqwest and its TLS stack entirely.
//!
//! # Quick start
//!
//! ```no_run
//! # async fn example() -> whisper_stt::Result<()> {
//! use whisper_stt::{ModelStore, Transcriber, WhisperModel};
//!
//! // Downloads `ggml-tiny.bin` into `models/` on first run, then reuses it.
//! // `ensure` is async and needs a runtime; tokio is what this crate is tested against.
//! let store = ModelStore::pretrained(WhisperModel::Tiny, "models");
//! let path = store.ensure().await?;
//!
//! let transcriber = Transcriber::new(path)?;
//! let result = transcriber.transcribe_file("assets/test.mp3", None)?;
//! println!("{}", result.text());
//! # Ok(()) }
//! ```
//!
//! # Supported audio formats
//!
//! Whatever the `symphonia` dependency was built with — see the feature table above. Decoding
//! mixes multi-channel audio down to mono and resamples anything that is not 16 kHz through
//! rubato, which low-passes before it decimates instead of folding high frequencies back into the
//! audio as noise.

pub mod audio;
pub mod error;
pub mod model;
pub mod store;
pub mod transcriber;

pub use error::{Error, Result};
pub use model::{ModelSource, WHISPER_MODELS, WHISPER_SAMPLE_RATE, WhisperModel};
pub use store::ModelStore;
pub use transcriber::{
    Segment, Transcriber, TranscriberOutput, TranscriptionOptions, TranscriptionSession,
};
