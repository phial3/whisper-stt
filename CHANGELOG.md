# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

First release of the rewritten crate. Everything below is a breaking change relative to the
previous, unpublished shape of the code; there is no migration path from the pre-refactor
`model_handler` API, which has been removed.

### Added

- `TranscriptionSession`, a reusable Whisper state obtained from `Transcriber::session`. Creating a
  state allocates the KV cache and the mel, encoder and decoder buffers — measured at ~357 MB on
  `large-v3-turbo` — and whisper.cpp redoes that allocation on every call. A session pays it once,
  which is what a live-transcription loop wants.
- `Segment`, exposed through `TranscriberOutput::segments`, with start and end timestamps and the
  model's own `no_speech_probability`. `TranscriberOutput::no_speech_probability` reports the worst
  segment's, which is the one-line guard against Whisper inventing text over silence.
- `TranscriptionOptions` fields `no_context`, `single_segment`, `suppress_non_speech` and
  `no_speech_threshold`.
- `ModelStore::download_with`, which reports progress as bytes land on disk.
- Feature flags: `network` for model downloads, and one per audio codec so a build can ship with
  only the formats it needs. See the table in the crate documentation.
- Criterion benchmarks for the audio path (`cargo bench`).
- CI covering tests, doc tests, formatting, clippy with `-D warnings`, and a feature-flag matrix.

### Changed

- Resampling now goes through rubato's FFT resampler instead of hand-rolled linear interpolation.
  Linear interpolation folds everything above the destination Nyquist limit back into the audio, and
  Whisper hears that folded noise as speech. `audio::resample` and
  `DecodedAudio::to_whisper_input` therefore return `Result` now.
- Model downloads stream to a `.part` staging file and are renamed into place on success, so a
  1.6 GB checkpoint never has to fit in memory and an interrupted download never leaves a corrupt
  file behind.
- `TranscriptionOptions` owns its strings (`Option<String>`) instead of borrowing them, so it has no
  lifetime parameter and can be stored.
- `tokio` is no longer a runtime dependency. Nothing in the library starts a runtime; `ModelStore`
  is async and runs on whatever runtime the caller is already on. Examples depend on tokio directly.
- `Error` carries `u16` for HTTP statuses and a dedicated `Error::Resample` variant, so it no longer
  depends on reqwest unless the `network` feature is on.

### Removed

- `handler` and its `ModelHandler`: a compatibility shim over `ModelStore` whose `new` panicked on
  an unknown model name. Use `ModelStore::from_name` and handle `Error::UnknownModel`.
- `TranscriberOutput::get_text`, `get_start_timestamp` and `get_end_timestamp`, which duplicated
  `text`, `start_timestamp` and `end_timestamp`.
- `transcriber::EXPECTED_SAMPLE_RATE`, which duplicated `WHISPER_SAMPLE_RATE`.
- `Transcriber::transcribe_with_params`, replaced by
  `TranscriptionSession::transcribe_samples_with_params`, which composes with audio the caller has
  already decoded.
