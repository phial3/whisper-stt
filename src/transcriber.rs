//! Loads a ggml checkpoint and runs transcription.

use std::path::Path;

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::audio::{self, DecodedAudio};
use crate::error::{Error, Result};
use crate::model::WHISPER_SAMPLE_RATE;

/// One transcribed chunk of speech, as produced by Whisper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// Start of the segment, in Whisper's 10 ms units (multiply by 10 to get milliseconds).
    pub start_timestamp: i64,
    /// End of the segment, in Whisper's 10 ms units.
    pub end_timestamp: i64,
    /// Transcribed text for this segment, including its leading space if Whisper emitted one.
    pub text: String,
}

impl Segment {
    /// Segment start in milliseconds.
    pub fn start_ms(&self) -> i64 {
        self.start_timestamp * 10
    }

    /// Segment end in milliseconds.
    pub fn end_ms(&self) -> i64 {
        self.end_timestamp * 10
    }
}

/// Result of a transcription run.
///
/// Beside the full text, the individual Whisper segments are kept so callers can build subtitles or
/// word-level timelines without re-running the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriberOutput {
    /// Concatenated text of every segment.
    pub text: String,
    /// Individual segments, in the order Whisper produced them.
    pub segments: Vec<Segment>,
    start_timestamp: i64,
    end_timestamp: i64,
}

impl TranscriberOutput {
    /// Concatenated text of every segment.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Individual segments, in the order Whisper produced them.
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// Start of the first segment, in Whisper's 10 ms units.
    pub fn start_timestamp(&self) -> i64 {
        self.start_timestamp
    }

    /// End of the last segment, in Whisper's 10 ms units.
    pub fn end_timestamp(&self) -> i64 {
        self.end_timestamp
    }

    /// Duration covered by the transcription, in milliseconds.
    pub fn duration_ms(&self) -> i64 {
        (self.end_timestamp - self.start_timestamp) * 10
    }

    /// Number of transcribed segments.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Whether any segment was produced.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Start of the first segment, in Whisper's 10 ms units.
    ///
    /// Retained for source compatibility with releases that did not expose the segment list.
    pub fn get_start_timestamp(&self) -> &i64 {
        &self.start_timestamp
    }

    /// End of the last segment, in Whisper's 10 ms units.
    ///
    /// Retained for source compatibility with releases that did not expose the segment list.
    pub fn get_end_timestamp(&self) -> &i64 {
        &self.end_timestamp
    }

    /// Concatenated text of every segment.
    ///
    /// Retained for source compatibility; prefer [`Self::text`].
    pub fn get_text(&self) -> &str {
        &self.text
    }
}

/// High-level transcription settings.
///
/// All defaults are Whisper's own defaults. Construct one, tweak the fields you care about, and pass
/// it to [`Transcriber::transcribe_file`]; for anything not covered here, build a
/// [`whisper_rs::FullParams`] by hand and use [`Transcriber::transcribe_with_params`].
#[derive(Debug, Clone)]
pub struct TranscriptionOptions<'a> {
    /// Force a source language (`"en"`, `"zh"`, `"auto"`, ...) instead of letting Whisper detect it.
    pub language: Option<&'a str>,
    /// Translate the speech into English instead of transcribing it.
    ///
    /// Requires a multilingual model; requests against an English-only checkpoint fail with
    /// [`Error::TranslationUnsupported`].
    pub translate: bool,
    /// Sampling strategy used when decoding tokens.
    pub sampling: SamplingStrategy,
    /// Emit progress lines while decoding.
    pub print_progress: bool,
    /// Emit special tokens (segment markers, `<|Novalidate|>`, etc.).
    pub print_special: bool,
    /// Number of threads to use. `None` lets whisper.cpp pick.
    pub n_threads: Option<i32>,
    /// Optional prompt prepended to the first window, useful for steering spelling and vocabulary.
    pub initial_prompt: Option<&'a str>,
}

impl Default for TranscriptionOptions<'_> {
    fn default() -> Self {
        Self {
            language: None,
            translate: false,
            sampling: SamplingStrategy::Greedy { best_of: 1 },
            print_progress: false,
            print_special: false,
            n_threads: None,
            initial_prompt: None,
        }
    }
}

impl TranscriptionOptions<'_> {
    /// Options with all defaults applied.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds the equivalent [`whisper_rs::FullParams`].
    pub fn to_full_params(&self) -> FullParams<'_, '_> {
        let mut params = FullParams::new(self.sampling.clone());
        params.set_translate(self.translate);
        params.set_print_progress(self.print_progress);
        params.set_print_special(self.print_special);
        params.set_language(self.language);
        if let Some(threads) = self.n_threads {
            params.set_n_threads(threads);
        }
        if let Some(prompt) = self.initial_prompt {
            params.set_initial_prompt(prompt);
        }
        params
    }
}

/// A loaded Whisper model, ready to transcribe audio.
///
/// ```no_run
/// # fn example() -> whisper_stt::Result<()> {
/// use whisper_stt::Transcriber;
///
/// let transcriber = Transcriber::new("models/ggml-tiny.bin")?;
/// let result = transcriber.transcribe_file("assets/test.mp3", None)?;
/// println!("{}", result.text());
/// # Ok(()) }
/// ```
#[derive(Debug)]
pub struct Transcriber {
    ctx: WhisperContext,
}

impl Transcriber {
    /// Loads a Whisper model from disk.
    ///
    /// Accepts anything path-like: a `&str`, a `PathBuf`, a [`crate::ModelStore`], or the legacy
    /// [`crate::handler::ModelHandler`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Whisper`] if whisper.cpp rejects the file.
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        Self::new_with_params(model_path, WhisperContextParameters::default())
    }

    /// Loads a Whisper model with explicit [`WhisperContextParameters`].
    ///
    /// Use this to enable GPU offload or token-level DTW timestamps.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Whisper`] if whisper.cpp rejects the file.
    pub fn new_with_params<P: AsRef<Path>>(
        model_path: P,
        context_params: WhisperContextParameters<'static>,
    ) -> Result<Self> {
        let ctx = WhisperContext::new_with_params(model_path.as_ref(), context_params)?;
        Ok(Self { ctx })
    }

    /// The underlying whisper-rs context.
    pub fn context(&self) -> &WhisperContext {
        &self.ctx
    }

    /// Whether the loaded model is multilingual.
    pub fn is_multilingual(&self) -> bool {
        self.ctx.is_multilingual()
    }

    /// Maximum text context size (number of tokens per Whisper window) of the loaded model.
    pub fn n_text_ctx(&self) -> i32 {
        self.ctx.n_text_ctx()
    }

    /// Transcribes an audio file, decoding it and resampling it to what Whisper expects.
    ///
    /// `options` may be `None` to use [`TranscriptionOptions::default`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Audio`] if the file cannot be decoded and [`Error::Whisper`] if the model
    /// fails to run.
    pub fn transcribe_file<P: AsRef<Path>>(
        &self,
        audio_path: P,
        options: Option<&TranscriptionOptions<'_>>,
    ) -> Result<TranscriberOutput> {
        let audio = audio::decode_file(audio_path)?;
        self.transcribe_decoded(&audio, options)
    }

    /// Transcribes already-decoded audio.
    ///
    /// Useful when the same file is transcribed several times, or when samples come from a
    /// microphone rather than a file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Whisper`] if the model fails to run.
    pub fn transcribe_decoded(
        &self,
        audio: &DecodedAudio,
        options: Option<&TranscriptionOptions<'_>>,
    ) -> Result<TranscriberOutput> {
        let samples = audio.to_whisper_input();
        self.transcribe_samples(&samples, options)
    }

    /// Transcribes raw audio samples.
    ///
    /// The samples **must** be mono, 16 kHz, `f32`. Anything else yields garbage; prefer
    /// [`Self::transcribe_file`] which guarantees that.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmptyAudio`] for an empty slice, [`Error::TranslationUnsupported`] when
    /// translation is requested for an English-only model, and [`Error::Whisper`] if the model
    /// fails to run.
    pub fn transcribe_samples(
        &self,
        samples: &[f32],
        options: Option<&TranscriptionOptions<'_>>,
    ) -> Result<TranscriberOutput> {
        if samples.is_empty() {
            return Err(Error::EmptyAudio);
        }

        let default_options = TranscriptionOptions::default();
        let options = options.unwrap_or(&default_options);

        if options.translate && !self.is_multilingual() {
            return Err(Error::TranslationUnsupported);
        }

        let mut state = self.create_state()?;
        state.full(options.to_full_params(), samples)?;

        self.collect_output(&state)
    }

    /// Transcribes an audio file with hand-built [`FullParams`].
    ///
    /// Escape hatch for everything [`TranscriptionOptions`] does not cover (grammars, callbacks,
    /// token-level timestamps, VAD, ...).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Audio`] if the file cannot be decoded and [`Error::Whisper`] if the model
    /// fails to run.
    pub fn transcribe_with_params<P: AsRef<Path>>(
        &self,
        audio_path: P,
        params: FullParams<'_, '_>,
    ) -> Result<TranscriberOutput> {
        let audio = audio::decode_file(audio_path)?;
        let samples = audio.to_whisper_input();

        let mut state = self.create_state()?;
        state.full(params, &samples)?;

        self.collect_output(&state)
    }

    /// Creates a Whisper state bound to this context.
    fn create_state(&self) -> Result<WhisperState> {
        self.ctx
            .create_state()
            .map_err(|_| Error::Whisper(whisper_rs::WhisperError::FailedToCreateState))
    }

    /// Collects every segment Whisper produced for `state`.
    fn collect_output(&self, state: &WhisperState) -> Result<TranscriberOutput> {
        let mut text = String::new();
        let mut segments = Vec::new();

        for index in 0..state.full_n_segments() {
            let segment = state
                .get_segment(index)
                .ok_or(Error::Whisper(whisper_rs::WhisperError::NullPointer))?;

            let start_timestamp = segment.start_timestamp();
            let end_timestamp = segment.end_timestamp();
            let segment_text = segment.to_str_lossy()?.into_owned();

            text.push_str(&segment_text);
            segments.push(Segment {
                start_timestamp,
                end_timestamp,
                text: segment_text,
            });
        }

        let start_timestamp = segments
            .first()
            .map_or(0, |segment| segment.start_timestamp);
        let end_timestamp = segments.last().map_or(0, |segment| segment.end_timestamp);

        Ok(TranscriberOutput {
            text,
            segments,
            start_timestamp,
            end_timestamp,
        })
    }
}

/// Sample rate [`Transcriber::transcribe_samples`] expects, in Hz.
pub const EXPECTED_SAMPLE_RATE: u32 = WHISPER_SAMPLE_RATE;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_disable_extras() {
        let options = TranscriptionOptions::default();
        assert!(!options.translate);
        assert!(!options.print_progress);
        assert!(options.language.is_none());
        assert!(matches!(
            options.sampling,
            SamplingStrategy::Greedy { best_of: 1 }
        ));
    }

    #[test]
    fn options_convert_to_full_params() {
        let options = TranscriptionOptions {
            language: Some("zh"),
            translate: true,
            print_progress: true,
            n_threads: Some(4),
            ..TranscriptionOptions::default()
        };
        let _params = options.to_full_params();
    }

    #[test]
    fn segments_expose_milliseconds() {
        let segment = Segment {
            start_timestamp: 0,
            end_timestamp: 320,
            text: " hello".into(),
        };
        assert_eq!(segment.start_ms(), 0);
        assert_eq!(segment.end_ms(), 3_200);
    }

    #[test]
    fn output_aggregates_timestamps() {
        let output = TranscriberOutput {
            text: " a b".into(),
            segments: vec![
                Segment {
                    start_timestamp: 0,
                    end_timestamp: 100,
                    text: " a".into(),
                },
                Segment {
                    start_timestamp: 100,
                    end_timestamp: 250,
                    text: " b".into(),
                },
            ],
            start_timestamp: 0,
            end_timestamp: 250,
        };

        // Unlike the pre-refactor version, timestamps span the whole transcript rather than only
        // the last segment.
        assert_eq!(output.start_timestamp(), 0);
        assert_eq!(output.end_timestamp(), 250);
        assert_eq!(output.duration_ms(), 2_500);
        assert_eq!(output.len(), 2);
        assert!(!output.is_empty());
        assert_eq!(output.text(), " a b");
        assert_eq!(output.get_text(), " a b");
        assert_eq!(*output.get_start_timestamp(), 0);
        assert_eq!(*output.get_end_timestamp(), 250);
    }

    #[test]
    fn empty_output_is_consistent() {
        let output = TranscriberOutput {
            text: String::new(),
            segments: Vec::new(),
            start_timestamp: 0,
            end_timestamp: 0,
        };
        assert!(output.is_empty());
        assert_eq!(output.duration_ms(), 0);
        assert_eq!(output.segments().len(), 0);
    }
}
