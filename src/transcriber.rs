//! Loads a ggml checkpoint and runs transcription.

use std::path::Path;

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::audio::{self, DecodedAudio};
use crate::error::{Error, Result};

/// One transcribed chunk of speech, as produced by Whisper.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// Start of the segment, in Whisper's 10 ms units (multiply by 10 to get milliseconds).
    pub start_timestamp: i64,
    /// End of the segment, in Whisper's 10 ms units.
    pub end_timestamp: i64,
    /// Transcribed text for this segment, including its leading space if Whisper emitted one.
    pub text: String,
    /// How sure Whisper is that this segment holds no speech at all, between 0 and 1.
    ///
    /// Whisper happily invents text over silence, noise and music — the well-known "thank you for
    /// watching" on an empty room. This is the model's own opinion on whether the segment was
    /// speech. How much it can be trusted depends on the checkpoint: it separates clean speech from
    /// silence on the classic multilingual models, and stays near zero for everything on the turbo
    /// ones. Gate on it only after checking what it actually reports for your model.
    pub no_speech_probability: f32,
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
#[derive(Debug, Clone, PartialEq)]
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

    /// Highest no-speech probability across the segments, or 1.0 when nothing was decoded.
    ///
    /// A one-line guard against invented text — see [`Segment::no_speech_probability`] for how much
    /// weight it carries for a given checkpoint.
    pub fn no_speech_probability(&self) -> f32 {
        if self.segments.is_empty() {
            return 1.0;
        }
        self.segments
            .iter()
            .map(|segment| segment.no_speech_probability)
            .fold(0.0, f32::max)
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
}

/// High-level transcription settings.
///
/// All defaults are Whisper's own defaults. Construct one, tweak the fields you care about, and pass
/// it to [`Transcriber::transcribe_file`]; for anything not covered here, build a
/// [`whisper_rs::FullParams`] by hand and use [`TranscriptionSession::transcribe_samples_with_params`].
#[derive(Debug, Clone)]
pub struct TranscriptionOptions {
    /// Force a source language (`"en"`, `"zh"`, `"auto"`, ...) instead of letting Whisper detect it.
    pub language: Option<String>,
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
    /// Do not reuse the previous window's text as context for the next one.
    ///
    /// Leave this off when transcribing one long recording. Turn it on when every call is an
    /// independent utterance — a live-transcription loop, say — otherwise Whisper carries the tail
    /// of the previous sentence into the new one and both come out wrong.
    pub no_context: bool,
    /// Force the whole input into a single segment instead of splitting it on timestamps.
    ///
    /// Short utterances (a few seconds) often confuse Whisper's timestamp logic, which shows up as
    /// text being dropped or repeated. One segment is the right shape for a single utterance.
    pub single_segment: bool,
    /// Suppress tokens that the model considers non-speech.
    ///
    /// This is what stops Whisper from inventing text over silence, noise, or music — the classic
    /// "thank you for watching" on an empty room. It needs token-level timestamps internally, which
    /// this option enables on its own.
    pub suppress_non_speech: bool,
    /// Probability above which a window is declared silence and produces no text.
    ///
    /// `None` keeps Whisper's default (0.6). Raise it to be stricter about what counts as speech;
    /// lower it if short, quiet utterances come back empty.
    pub no_speech_threshold: Option<f32>,
    /// Number of threads to use. `None` lets whisper.cpp pick.
    pub n_threads: Option<i32>,
    /// Optional prompt prepended to the first window, useful for steering spelling and vocabulary.
    pub initial_prompt: Option<String>,
}

impl Default for TranscriptionOptions {
    fn default() -> Self {
        Self {
            language: None,
            translate: false,
            sampling: SamplingStrategy::Greedy { best_of: 1 },
            print_progress: false,
            print_special: false,
            no_context: false,
            single_segment: false,
            suppress_non_speech: false,
            no_speech_threshold: None,
            n_threads: None,
            initial_prompt: None,
        }
    }
}

impl TranscriptionOptions {
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
        params.set_language(self.language.as_deref());
        params.set_no_context(self.no_context);
        params.set_single_segment(self.single_segment);
        // whisper.cpp only has the per-token probabilities this needs when token timestamps are on.
        if self.suppress_non_speech {
            params.set_token_timestamps(true);
            params.set_suppress_nst(true);
        }
        if let Some(threshold) = self.no_speech_threshold {
            params.set_no_speech_thold(threshold);
        }
        if let Some(threads) = self.n_threads {
            params.set_n_threads(threads);
        }
        if let Some(prompt) = self.initial_prompt.as_deref() {
            params.set_initial_prompt(prompt);
        }
        params
    }
}

/// A loaded Whisper model, ready to transcribe audio.
///
/// Each `transcribe_*` call creates a Whisper state, runs it, and drops it again. That is the safe
/// default and costs nothing for a handful of files; for a loop that runs the same model over and
/// over, take a [`TranscriptionSession`] and reuse one state instead.
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
    /// Accepts anything path-like: a `&str`, a `PathBuf`, or a [`crate::ModelStore`].
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

    /// Opens a session that reuses a single Whisper state for every transcription.
    ///
    /// Creating a state allocates the KV cache plus the mel and encoder buffers — hundreds of
    /// megabytes on the large checkpoints — and whisper.cpp redoes that work on every call. A
    /// session pays for it once.
    ///
    /// # Text context carries over
    ///
    /// Whisper feeds the text it already produced back in as a prompt for what comes next. With a
    /// fresh state that history starts empty; with a reused state it survives from one call to the
    /// next, and only [`TranscriptionOptions::no_context`] clears it. That is what you want when
    /// feeding one long recording in order, and the opposite of what you want when every call is
    /// an independent utterance — set `no_context: true` for a live-transcription loop.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Whisper`] if whisper.cpp cannot allocate the state.
    pub fn session(&self) -> Result<TranscriptionSession<'_>> {
        Ok(TranscriptionSession {
            model: self,
            state: self.create_state()?,
        })
    }

    /// Transcribes an audio file, decoding it and resampling it to what Whisper expects.
    ///
    /// Convenience wrapper: opens a throwaway session, runs it, drops it. See
    /// [`Self::session`] for the reusable variant.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Audio`] if the file cannot be decoded and [`Error::Whisper`] if the model
    /// fails to run.
    pub fn transcribe_file<P: AsRef<Path>>(
        &self,
        audio_path: P,
        options: Option<&TranscriptionOptions>,
    ) -> Result<TranscriberOutput> {
        self.session()?.transcribe_file(audio_path, options)
    }

    /// Transcribes already-decoded audio in a throwaway session.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Resample`] if the audio cannot be resampled, and [`Error::Whisper`] if the
    /// model fails to run.
    pub fn transcribe_decoded(
        &self,
        audio: &DecodedAudio,
        options: Option<&TranscriptionOptions>,
    ) -> Result<TranscriberOutput> {
        self.session()?.transcribe_decoded(audio, options)
    }

    /// Transcribes raw audio samples in a throwaway session.
    ///
    /// # Errors
    ///
    /// The same errors as [`TranscriptionSession::transcribe_samples`].
    pub fn transcribe_samples(
        &self,
        samples: &[f32],
        options: Option<&TranscriptionOptions>,
    ) -> Result<TranscriberOutput> {
        self.session()?.transcribe_samples(samples, options)
    }

    /// Creates a Whisper state bound to this context.
    fn create_state(&self) -> Result<WhisperState> {
        self.ctx
            .create_state()
            .map_err(|_| Error::Whisper(whisper_rs::WhisperError::FailedToCreateState))
    }
}

/// A [`Transcriber`] bound to one reusable Whisper state.
///
/// Created by [`Transcriber::session`]. Prefer it over calling `Transcriber::transcribe_*` in a
/// loop: the state is allocated once instead of per call. Read [`Transcriber::session`] for what
/// reusing a state means for text context.
///
/// ```no_run
/// # fn example() -> whisper_stt::Result<()> {
/// use whisper_stt::Transcriber;
///
/// let transcriber = Transcriber::new("models/ggml-tiny.bin")?;
/// let mut session = transcriber.session()?;
/// for chunk in ["one.wav", "two.wav"] {
///     let output = session.transcribe_file(chunk, None)?;
///     println!("{}", output.text());
/// }
/// # Ok(()) }
/// ```
#[derive(Debug)]
pub struct TranscriptionSession<'model> {
    model: &'model Transcriber,
    state: WhisperState,
}

impl<'model> TranscriptionSession<'model> {
    /// The transcriber this session runs on.
    pub fn model(&self) -> &'model Transcriber {
        self.model
    }

    /// Transcribes an audio file, decoding it and resampling it to what Whisper expects.
    ///
    /// `options` may be `None` to use [`TranscriptionOptions::default`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Audio`] if the file cannot be decoded, [`Error::Resample`] if it cannot be
    /// resampled to 16 kHz, and [`Error::Whisper`] if the model fails to run.
    pub fn transcribe_file<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
        options: Option<&TranscriptionOptions>,
    ) -> Result<TranscriberOutput> {
        let audio = audio::decode_file(audio_path)?;
        self.transcribe_decoded(&audio, options)
    }

    /// Transcribes already-decoded audio.
    ///
    /// Useful when samples come from a microphone rather than a file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Resample`] if the audio cannot be resampled to 16 kHz, and
    /// [`Error::Whisper`] if the model fails to run.
    pub fn transcribe_decoded(
        &mut self,
        audio: &DecodedAudio,
        options: Option<&TranscriptionOptions>,
    ) -> Result<TranscriberOutput> {
        let samples = audio.to_whisper_input()?;
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
        &mut self,
        samples: &[f32],
        options: Option<&TranscriptionOptions>,
    ) -> Result<TranscriberOutput> {
        if samples.is_empty() {
            return Err(Error::EmptyAudio);
        }

        let default_options = TranscriptionOptions::default();
        let options = options.unwrap_or(&default_options);

        if options.translate && !self.model.is_multilingual() {
            return Err(Error::TranslationUnsupported);
        }

        self.state.full(options.to_full_params(), samples)?;
        self.collect()
    }

    /// Transcribes raw audio samples with hand-built [`FullParams`].
    ///
    /// Escape hatch for everything [`TranscriptionOptions`] does not cover (grammars, callbacks,
    /// token-level timestamps, VAD, ...).
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmptyAudio`] for an empty slice and [`Error::Whisper`] if the model fails
    /// to run.
    pub fn transcribe_samples_with_params(
        &mut self,
        samples: &[f32],
        params: FullParams<'_, '_>,
    ) -> Result<TranscriberOutput> {
        if samples.is_empty() {
            return Err(Error::EmptyAudio);
        }
        self.state.full(params, samples)?;
        self.collect()
    }

    /// Collects every segment Whisper produced in the last run.
    fn collect(&self) -> Result<TranscriberOutput> {
        let mut text = String::new();
        let mut segments = Vec::new();

        for index in 0..self.state.full_n_segments() {
            let segment = self
                .state
                .get_segment(index)
                .ok_or(Error::Whisper(whisper_rs::WhisperError::NullPointer))?;

            let start_timestamp = segment.start_timestamp();
            let end_timestamp = segment.end_timestamp();
            let segment_text = segment.to_str_lossy()?.into_owned();
            let no_speech_probability = segment.no_speech_probability();

            text.push_str(&segment_text);
            segments.push(Segment {
                start_timestamp,
                end_timestamp,
                text: segment_text,
                no_speech_probability,
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
            language: Some("zh".into()),
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
            no_speech_probability: 0.1,
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
                    no_speech_probability: 0.05,
                },
                Segment {
                    start_timestamp: 100,
                    end_timestamp: 250,
                    text: " b".into(),
                    no_speech_probability: 0.42,
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
        // The reported probability is the worst segment's, not an average.
        assert!((output.no_speech_probability() - 0.42).abs() < f32::EPSILON);
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
        // Nothing decoded at all is as silent as it gets.
        assert_eq!(output.no_speech_probability(), 1.0);
    }
}
