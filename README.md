# whisper-stt 🔈 📖

An audio to text transcription library written in Rust that utilizes
[whisper-rs](https://crates.io/crates/whisper-rs) bindings.

<img src="readme_logo.jpg" width="400" height="400">

## What is whisper-stt?

whisper-stt is a library written in Rust with the goal of making audio to text transcription simple
for developers. It handles the boring parts of running Whisper: downloading and caching a model,
decoding arbitrary audio into the mono 16 kHz float stream Whisper expects, and reading the results
back out as structured segments. The aim is for developers to be able to incorporate transcription in
their projects quickly 🌩️

## Features

- Automatically downloads models that have not already been installed. **All twelve checkpoints
  whisper-rs supports** are available:

  | Model | Multilingual | File |
  |---|---|---|
  | `tiny` | ✅ | `ggml-tiny.bin` |
  | `tiny.en` | ❌ | `ggml-tiny.en.bin` |
  | `base` | ✅ | `ggml-base.bin` |
  | `base.en` | ❌ | `ggml-base.en.bin` |
  | `small` | ✅ | `ggml-small.bin` |
  | `small.en` | ❌ | `ggml-small.en.bin` |
  | `medium` | ✅ | `ggml-medium.bin` |
  | `medium.en` | ❌ | `ggml-medium.en.bin` |
  | `large-v1` | ✅ | `ggml-large-v1.bin` |
  | `large-v2` | ✅ | `ggml-large-v2.bin` |
  | `large-v3` | ✅ | `ggml-large-v3.bin` |
  | `large-v3-turbo` | ✅ | `ggml-large-v3-turbo.bin` |

  Plain `"large"` resolves to `large-v3`, as before. Names are matched case-insensitively and
  tolerate a `ggml-` prefix, a `.bin` suffix, and underscores, so `"ggml-tiny.en.bin"`, `"tiny_en"`
  and `"tiny.en"` all work.

- Quantized checkpoints, fine-tunes, and anything else whisper-rs can load can be used directly by
  handing a path to [`Transcriber::new`].

- Transcribes audio from any container/codec the `symphonia` dependency is built with — including
  mp3, wav, flac, ogg and mkv — and automatically:
  - mixes multi-channel audio down to mono, and
  - resamples anything that is not 16 kHz.

- Returns per-segment results with timestamps, so subtitles and word timelines come for free.

## Getting started

Add the crate to your project's `Cargo.toml`:

```toml
[dependencies]
whisper-stt = "0.0.1"
tokio = { version = "1", features = ["full"] }
```

Due to the nature of downloading models, preparing them requires `.await`, so an async runtime is
needed. [Tokio](https://github.com/tokio-rs/tokio) is what the library is developed against, and is
the recommended runtime.

## Usage

```rust
use whisper_stt::{ModelStore, Transcriber, WhisperModel};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Downloads `ggml-tiny.bin` into `models/` on first run, then reuses it.
    let store = ModelStore::pretrained(WhisperModel::Tiny, "models");
    store.ensure().await?;

    let transcriber = Transcriber::new(store.path())?;
    let result = transcriber.transcribe_file("assets/test.mp3", None)?;

    println!("start[{}]-end[{}] {}", result.start_timestamp(), result.end_timestamp(), result.text());

    for segment in result.segments() {
        println!("  [{} ms - {} ms] {}", segment.start_ms(), segment.end_ms(), segment.text.trim());
    }
    Ok(())
}
```

The snippet can be run via `cargo run --example usage_example`.

### Choosing a language, translating, and other options

```rust
use whisper_stt::TranscriptionOptions;

let options = TranscriptionOptions {
    language: Some("zh"),   // skip auto-detection
    translate: true,        // translate the speech into English
    n_threads: Some(8),
    ..TranscriptionOptions::default()
};

let result = transcriber.transcribe_file("assets/gongxifachai.mp3", Some(&options))?;
```

See `cargo run --example usage_example_chinese`.

Anything not covered by `TranscriptionOptions` (grammars, callbacks, VAD, token-level DTW timestamps,
GPU offload) stays reachable: build a `whisper_rs::FullParams` yourself and call
`Transcriber::transcribe_with_params`, or use `Transcriber::new_with_params` to pass
`WhisperContextParameters`.

### Listing models

```text
cargo run --example models              # print the catalogue
cargo run --example models -- tiny.en   # download one checkpoint into models/
```

## Migration from the old `model_handler` API

`ModelHandler` still exists as a thin shim so existing code keeps working:

```rust
let handler = whisper_stt::model_handler::ModelHandler::new("tiny", "models/").await;
let transcriber = Transcriber::new(handler)?;
```

Prefer moving to `ModelStore`, which is checked against the catalogue at compile time and reports
failures instead of panicking.

## License

MIT
