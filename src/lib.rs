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
//! | [`store`] | Where a model lives on disk, and how to fetch it. The only networked module. |
//! | [`audio`] | Decoding media files into mono 16 kHz `f32`, what Whisper is trained on. |
//! | [`transcriber`] | Loading a model and running it. |
//! | [`handler`] | Compatibility shim for the pre-refactor string-based API. |
//!
//! # Supported models
//!
//! All twelve checkpoints whisper-rs knows about are available:
//! `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`,
//! `large-v1`, `large-v2`, `large-v3`, and `large-v3-turbo`. See [`WhisperModel`].
//!
//! # Quick start
//!
//! ```no_run
//! # async fn example() -> whisper_stt::Result<()> {
//! use whisper_stt::{ModelStore, Transcriber, WhisperModel};
//!
//! // Downloads `ggml-tiny.bin` into `models/` on first run, then reuses it.
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
//! Whatever the `symphonia` dependency was built with. Decoding mixes multi-channel audio down to
//! mono and resamples anything that is not 16 kHz, which is what Whisper requires.

pub mod audio;
pub mod error;
pub mod handler;
pub mod model;
pub mod store;
pub mod transcriber;

pub use error::{Error, Result};
pub use model::{ModelSource, WHISPER_MODELS, WHISPER_SAMPLE_RATE, WhisperModel};
pub use store::ModelStore;
pub use transcriber::{Segment, Transcriber, TranscriberOutput, TranscriptionOptions};
