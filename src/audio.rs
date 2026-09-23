//! Audio decoding helpers.
//!
//! Whisper models are trained on **16 kHz mono** `f32` PCM. Media files rarely come in that shape,
//! so this module has two responsibilities:
//!
//! 1. Decode any container/codec Symphonia supports into interleaved `f32` samples
//!    ([`decode_file`]).
//! 2. Convert that into what Whisper wants: a mono stream resampled to
//!    [`WHISPER_SAMPLE_RATE`] ([`DecodedAudio::to_whisper_input`]).
//!
//! Only format and codec features enabled on the `symphonia` dependency can be decoded.

use std::path::Path;

use symphonia::core::audio::Channels;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::error::{Error, Result};
use crate::model::WHISPER_SAMPLE_RATE;

/// Decoded PCM audio, in the channel-interleaved layout produced by the decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedAudio {
    /// Interleaved samples: frame `i`, channel `c` lives at index `i * channels + c`.
    pub samples: Vec<f32>,
    /// Sample rate of the decoded audio, in Hz.
    pub sample_rate: u32,
    /// Number of channels in the decoded audio.
    pub channels: usize,
}

impl DecodedAudio {
    /// Duration of the decoded audio in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        if self.channels == 0 || self.sample_rate == 0 {
            return 0;
        }
        let frames = self.samples.len() / self.channels;
        (frames as u64 * 1000) / self.sample_rate as u64
    }

    /// Mixes all channels down to a single mono channel.
    ///
    /// Already-mono audio is returned as-is (with a copy).
    pub fn to_mono(&self) -> Vec<f32> {
        mix_down(&self.samples, self.channels)
    }

    /// Converts the decoded audio into the layout Whisper expects: mono `[f32]` resampled to
    /// [`WHISPER_SAMPLE_RATE`].
    ///
    /// Feeding anything else to Whisper produces severely degraded transcripts, so callers should
    /// always route decoder output through this method.
    pub fn to_whisper_input(&self) -> Vec<f32> {
        let mono = self.to_mono();
        resample(&mono, self.sample_rate, WHISPER_SAMPLE_RATE)
    }
}

/// Decodes an audio file into interleaved `f32` samples.
///
/// The container and codec are auto-detected; anything whose feature flag is enabled on the
/// `symphonia` dependency can be read.
///
/// # Errors
///
/// Returns [`Error::Audio`] if the file cannot be opened or probed, [`Error::NoAudioTrack`] if the
/// file contains no audio track, and [`Error::EmptyAudio`] if nothing was decoded.
pub fn decode_file(path: impl AsRef<Path>) -> Result<DecodedAudio> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .map_err(|err| Error::Audio(format!("cannot open {}: {err}", path.display())))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let hint = Hint::new();

    let format_opts: FormatOptions = Default::default();
    let metadata_opts: MetadataOptions = Default::default();
    let decoder_opts: AudioDecoderOptions = Default::default();

    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, format_opts, metadata_opts)
        .map_err(|err| Error::Audio(format!("cannot probe {}: {err}", path.display())))?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or(Error::NoAudioTrack)?;

    let codec_params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or(Error::NoAudioTrack)?;

    let sample_rate = codec_params.sample_rate.unwrap_or(WHISPER_SAMPLE_RATE);
    let channels = codec_params
        .channels
        .as_ref()
        .map(Channels::count)
        .unwrap_or(1);

    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(codec_params, &decoder_opts)
        .map_err(|err| Error::Audio(format!("cannot create decoder: {err}")))?;

    let mut samples: Vec<f32> = Vec::new();
    // `copy_to_vec_interleaved` resizes the destination to the exact number of copied samples, so
    // it cannot append to `samples` directly.
    let mut packet_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            // End of stream.
            Ok(None) => break,
            Err(SymphoniaError::ResetRequired) => {
                // The track list changed underneath us; rebuilding the decoder set is out of scope
                // for a whole-file decode. Only chained OGG streams hit this today.
                break;
            }
            Err(SymphoniaError::IoError(err)) => {
                return Err(Error::Audio(format!(
                    "io error while reading packets: {err}"
                )));
            }
            Err(err) => return Err(Error::Audio(format!("cannot read packet: {err}"))),
        };

        if packet.track_id != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                audio_buf.copy_to_vec_interleaved(&mut packet_samples);
                samples.extend_from_slice(&packet_samples);
            }
            // A malformed packet is skipped; anything else stops the decode.
            Err(SymphoniaError::DecodeError(_)) => (),
            Err(_) => break,
        }
    }

    if samples.is_empty() {
        return Err(Error::EmptyAudio);
    }

    Ok(DecodedAudio {
        samples,
        sample_rate,
        channels: channels.max(1),
    })
}

/// Mixes interleaved multi-channel audio down to mono by averaging all channels per frame.
pub fn mix_down(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    if samples.is_empty() {
        return Vec::new();
    }

    let frames = samples.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    for frame in samples.chunks_exact(channels) {
        mono.push(frame.iter().sum::<f32>() / channels as f32);
    }
    mono
}

/// Resamples mono audio to `dst_rate` using linear interpolation.
///
/// Returns the input unchanged when `src_rate == dst_rate`.
pub fn resample(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || src_rate == 0 || dst_rate == 0 {
        return samples.to_vec();
    }
    if samples.len() < 2 {
        return samples.to_vec();
    }

    let src_rate = src_rate as f64;
    let dst_rate = dst_rate as f64;
    let out_len = (samples.len() as f64 * dst_rate / src_rate).floor() as usize;
    let mut out = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let position = i as f64 * src_rate / dst_rate;
        let left = position.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let weight = position - left as f64;
        out.push(samples[left] + (samples[right] - samples[left]) * weight as f32);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix_down_is_a_no_op_for_mono() {
        let samples = vec![1.0f32, 2.0, 3.0];
        assert_eq!(mix_down(&samples, 1), samples);
        assert_eq!(mix_down(&samples, 0), samples);
    }

    #[test]
    fn mix_down_averages_channels() {
        // Frames: [L0, R0], [L1, R1]
        let samples = vec![1.0f32, 3.0, 5.0, 5.0];
        let mono = mix_down(&samples, 2);
        assert_eq!(mono, vec![2.0, 5.0]);
    }

    #[test]
    fn mix_down_ignores_partial_trailing_frame() {
        let samples = vec![1.0f32, 3.0, 5.0];
        assert_eq!(mix_down(&samples, 2), vec![2.0]);
    }

    #[test]
    fn resample_is_identity_when_rates_match() {
        let samples = vec![0.1f32, 0.2, 0.3];
        assert_eq!(resample(&samples, 16_000, 16_000), samples);
    }

    #[test]
    fn resample_downscales_length() {
        let samples: Vec<f32> = (0..48_000).map(|i| i as f32).collect();
        let resampled = resample(&samples, 48_000, WHISPER_SAMPLE_RATE);
        assert_eq!(resampled.len(), 16_000);
        // The first and last taps stay anchored to the original endpoints.
        assert_eq!(resampled[0], 0.0);
        assert!(resampled[resampled.len() - 1] <= 48_000.0);
    }

    #[test]
    fn resample_upscales_length() {
        let samples = vec![0.0f32, 1.0];
        let up = resample(&samples, 2, 4);
        assert_eq!(up.len(), 4);
    }

    #[test]
    fn resample_handles_tiny_inputs() {
        assert!(resample(&[], 8_000, 16_000).is_empty());
        assert_eq!(resample(&[0.5], 8_000, 16_000), vec![0.5]);
    }

    #[test]
    fn decoded_audio_reports_duration() {
        let audio = DecodedAudio {
            samples: vec![0.0; 16_000],
            sample_rate: 16_000,
            channels: 1,
        };
        assert_eq!(audio.duration_ms(), 1_000);
    }

    #[test]
    fn decoded_audio_is_empty_safe() {
        let audio = DecodedAudio {
            samples: Vec::new(),
            sample_rate: 0,
            channels: 0,
        };
        assert_eq!(audio.duration_ms(), 0);
        assert!(audio.to_mono().is_empty());
    }
}
