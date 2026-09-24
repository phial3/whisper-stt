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
  mp3, wav, flac, ogg, mkv, mp4 and m4a — and automatically:
  - mixes multi-channel audio down to mono, and
  - resamples anything that is not 16 kHz through an FFT resampler that anti-aliases properly.

- Returns per-segment results with timestamps, so subtitles and word timelines come for free.

- Reuses one Whisper state across transcriptions when you ask it to, instead of reallocating the
  model's buffers on every call.

- Ships runnable examples for the whole pipeline: `recording` / `record_cpal` capture the microphone
  interactively, `resample` compares resampler engines (`rubato`), and `vad` cuts silence out before
  transcribing with an offline neural VAD (`voice-engine`).

## Getting started

Add the crate to your project's `Cargo.toml`:

```toml
[dependencies]
whisper-stt = "0.1"
```

Nothing in the library starts a thread or a runtime. Preparing a model is async, so it needs
whatever runtime your application already runs on —
[Tokio](https://github.com/tokio-rs/tokio) is what this crate is developed and tested against:

```toml
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

### Feature flags

Pick the audio formats you need and drop the rest; every codec is a separate flag.

| Feature | Default | Enables |
|---|---|---|
| `network` | ✅ | `ModelStore::ensure` / `download`, via reqwest |
| `mp3` | ✅ | MPEG audio |
| `aac` | ✅ | AAC, as found in `.m4a` and `.mp4` |
| `flac` | ✅ | FLAC |
| `vorbis`, `ogg` | ✅ | Vorbis and the Ogg container |
| `mkv`, `isomp4` | ✅ | Matroska/WebM and MP4/MOV containers |
| `wav`, `pcm` | ✅ | Uncompressed PCM and the WAVE container |
| `metadata` | ✅ | ID3v1, ID3v2 and APE tag readers |
| `aiff`, `caf`, `alac`, `adpcm` | ❌ | The remaining Symphonia codecs |

```toml
# WAV in, no downloader, no TLS stack in the binary.
whisper-stt = { version = "0.1", default-features = false, features = ["wav", "pcm"] }
```

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

### Reusing one Whisper state

`Transcriber::transcribe_*` creates a Whisper state, runs it, and throws it away. Creating a state
allocates the KV cache and the mel, encoder and decoder buffers — on `large-v3-turbo`, whisper.cpp
reports ~357 MB of it, and it redoes the work on every call. For a loop, ask for a session:

```rust
let transcriber = Transcriber::new("models/ggml-large-v3-turbo.bin")?;
let mut session = transcriber.session()?;

for path in ["take-1.wav", "take-2.wav", "take-3.wav"] {
    let result = session.transcribe_file(path, None)?;
    println!("{}", result.text());
}
```

Whisper feeds the text it has already produced back in as a prompt for what comes next, and a reused
state keeps that history between calls. That is what you want when you are reading one long recording
in order, and the opposite of what you want when every call is a separate utterance — set
`no_context: true` in that case, which is what `live_translate` does.

### Choosing a language, translating, and other options

```rust
use whisper_stt::TranscriptionOptions;

let options = TranscriptionOptions {
    language: Some("zh".into()),   // skip auto-detection
    translate: true,               // translate the speech into English
    n_threads: Some(8),
    ..TranscriptionOptions::default()
};

let result = transcriber.transcribe_file("assets/gongxifachai.mp3", Some(&options))?;
```

See `cargo run --example usage_example_chinese`.

Anything not covered by `TranscriptionOptions` (grammars, callbacks, VAD, token-level DTW timestamps,
GPU offload) stays reachable: build a `whisper_rs::FullParams` yourself and call
`session.transcribe_samples_with_params`, or use `Transcriber::new_with_params` to pass
`WhisperContextParameters`.

### Live translation: the whole pipeline

`examples/live_translate.rs` wires every stage together: capture → denoise → resample → voice
activity → transcribe, with decoding on a worker thread so it never blocks the microphone. Output
comes back one line per turn, like a transcript:

```text
[00:06] heard 2.9s
[00:06] zh  你叫什么名字?
[00:15] heard 2.8s
[00:15] zh  你叫什么名字?
```

```text
cargo run --example live_translate                       # auto-detect language -> English
cargo run --example live_translate -- --model large-v3   # turbo cannot translate, see below
cargo run --example live_translate -- --file take.wav    # run the same chain over a file
cargo run --example live_translate -- --no-translate     # transcribe, do not translate
cargo run --example live_translate -- --source           # show what was said, not just the result
cargo run --example live_translate -- --pause 3000       # wait 3s of quiet, not 2s
cargo run --example live_translate -- --denoise          # also run RNNoise
cargo run --example live_translate -- --out speech.wav   # keep the speech that was sent
```

Press **Enter** to start and **Enter** (or Ctrl+C) to stop. **2 seconds of quiet ends a turn**
(`--pause` changes it), and turns are measured in audio samples rather than wall clock, so a file
run — which chews through minutes of audio in a fraction of a second — segments exactly like a live
one does.

The voice-activity stage is `voice-engine`'s `VadProcessor`, not hand-rolled glue, and it is the one
that keeps the transcript honest. Whisper does not say "I heard nothing" — give it silence and it
confidently invents a sentence. On this crate's own test clip, half a second of quiet came back as
*请不吝点赞 订阅 转发 打赏支持明镜与点点栏目*. Nothing in Whisper's own settings stops that
(`suppress_nst`, `no_speech_thold` and friends all left it intact), so the fix has to be upstream:

- a turn is closed by 2 s of quiet, and must be at least 500 ms long;
- a turn whose windows are mostly *not* speech is dropped with a `skipped` line;
- the audio between onset and offset is kept whole. Gluing only the speech windows together — the
  obvious "optimisation" — hands Whisper time-compressed audio with a click at every join;
- each turn gets a third of a second of lead-in and 300 ms of trailing quiet, because Silero needs a
  window or two to let go of what it was hearing.

Three findings that matter when you run it:

- **Turbo checkpoints cannot translate.** They are trained on transcription data only and return
  the source language even when the translate task is requested. Use `large-v3` (or
  medium/small/base/tiny) for real translation into English; the example says so at startup.
- **Denoising is opt-in** (`--denoise`). RNNoise is trained on 48 kHz wideband audio, and on this
  crate's own test clip it preserved the level but cut a 2.8 s utterance to 0.4 s and turned a
  correct transcript into nonsense. Compare on your own microphone before relying on it.
- **A turn is capped at 25 s.** `voice-engine` keeps a ring of the last 1000 detector windows and
  drops everything older than 5 s once it overflows, so a monologue that never pauses is cut
  deliberately rather than silently truncated.

`--file` swaps the microphone for a media file and runs the identical chain, which is how you check
the conditioning stages without talking.

### Resampling

Whisper wants 16 kHz, and whatever the source rate is, something has to bridge the gap. That
something used to be linear interpolation, which is cheap and quietly wrong: everything above the new
Nyquist limit folds back into the audio instead of being filtered out, and Whisper hears the folded
noise as speech. `whisper_stt::audio::resample` now runs rubato's FFT resampler, which low-passes
before it decimates.

`examples/resample.rs` measures the built-in engine against rubato's heavier ones:

```text
cargo run --example resample                                   # assets/test.mp3, every engine
cargo run --example resample -- assets/test.mp3 --engine sinc
cargo run --example resample -- assets/test.mp3 --out out.wav  # write a 16 kHz WAV
```

It resamples the clip with each engine and times it, then runs an anti-aliasing probe: a synthetic
24 kHz signal holding a 3 kHz tone (which must survive) and a 10 kHz tone (which sits above the
8 kHz Nyquist limit of 16 kHz and must be filtered out). Whatever lands back at 6 kHz is aliasing:

```text
engine               3 kHz kept    6 kHz alias
fft                     -6.0 dB      < -120 dB
sinc                    -6.0 dB      < -120 dB
poly                    -6.0 dB        -8.4 dB
built-in                -6.0 dB      < -120 dB
```

`Fft` — the built-in choice — is right for fixed-rate file conversion, `Async::new_sinc` for a ratio
that can drift, `Async::new_poly` when CPU matters more than the filter. rubato 5.0 works on
`audioadapter` buffers, so the input is wrapped in an `InterleavedSlice` and read back through the
`Adapter` trait.

`cargo bench` measures the same path in samples per second.

### Cutting silence before transcribing

Transcribing silence wastes time and invites hallucinated filler. `examples/vad.rs` segments audio
with the neural VADs built into `voice-engine` — Rust ports of Silero and Ten VAD with the weights
baked into the crate, so it runs fully offline with no model download and no ONNX runtime:

```text
cargo run --example vad                                      # assets/test.mp3, Silero
cargo run --example vad -- assets/test.mp3 --engine ten      # the Ten VAD instead
cargo run --example vad -- assets/test.mp3 --threshold 0.3   # more permissive
cargo run --example vad -- assets/test.mp3 --out speech.wav  # keep only the speech
```

It prints a probability timeline, lists the segments it found with timestamps, and can write a
speech-only 16 kHz WAV you can hand straight to `Transcriber::transcribe_file`. The two detectors
take different frame sizes (Silero 512 samples, Ten 256) and score on different scales — on the
same clip Ten's probabilities run well below Silero's, so the default threshold follows the engine.
Compare both on your own audio before trusting one.

### Recording the microphone

Two examples capture the microphone and write a WAV file into `recordings/`. Both are **interactive**:

1. Press **Enter** to start recording — Ctrl+C here aborts without writing anything.
2. Press **Enter** again, or hit **Ctrl+C**, to stop. A live elapsed timer is printed while you talk.
3. The capture is resampled to 16 kHz mono, written to disk, then played back on the default output
   device. The program exits by itself once playback finishes.

```text
cargo run --example recording                    # rodio  -> recordings/rodio-<timestamp>.wav
cargo run --example recording -- hello.wav       # rodio  -> recordings/hello.wav
cargo run --example recording -- --list          # list the input devices rodio can see

cargo run --example record_cpal                  # cpal   -> recordings/cpal-<timestamp>.wav
cargo run --example record_cpal -- note.wav      # cpal   -> recordings/note.wav
cargo run --example record_cpal -- --list        # list devices and every config they support
cargo run --example record_cpal -- --transcribe  # also transcribe the take with `tiny`
```

A bare filename is resolved inside `recordings/`; any other path is used as given. The extension must
be `.wav`.

`recording.rs` uses rodio's high-level `microphone` module and asks the hardware for 16 kHz mono
directly. `record_cpal.rs` drives cpal itself, handles every sample format the device offers, and
always resamples through `whisper_stt::audio`. Either way the file on disk is 16 kHz mono, so it can
be handed to `Transcriber::transcribe_file` untouched.

Ctrl+C is caught by a real signal handler rather than killing the process, so a take stopped with
Ctrl+C is still written and played back normally. On macOS the first run needs microphone permission
for your terminal/IDE.

### Listing models

```text
cargo run --example models              # print the catalogue
cargo run --example models -- tiny.en   # download one checkpoint into models/
```

## License

MIT
