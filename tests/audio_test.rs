//! Integration coverage for the decoding path, using the audio files shipped in `assets/`.
//!
//! No model download and no network access is required.

use whisper_stt::audio::{self, DecodedAudio};
use whisper_stt::{Result, WHISPER_SAMPLE_RATE};

fn decode(name: &str) -> Result<DecodedAudio> {
    audio::decode_file(format!("assets/{name}"))
}

#[test]
fn english_mp3_decodes_to_the_expected_layout() {
    let audio = decode("test.mp3").unwrap();

    // Source file is mono 24 kHz (MPEG-2 Layer III).
    assert_eq!(audio.channels, 1);
    assert_eq!(audio.sample_rate, 24_000);
    assert!(audio.samples.len() > 16_000);
    assert!(audio.duration_ms() > 1_000);
}

#[test]
fn whisper_input_is_resampled_to_16khz() {
    let audio = decode("test.mp3").unwrap();
    let samples = audio.to_whisper_input().unwrap();

    // 24 kHz -> 16 kHz downsamples by two thirds.
    let expected = audio.samples.len() * 2 / 3;
    assert!(
        (samples.len() as i64 - expected as i64).unsigned_abs() <= 1,
        "resampled length {} is not within a sample of {expected}",
        samples.len()
    );
    assert!(samples.iter().all(|sample| sample.is_finite()));
    assert!(samples.iter().any(|sample| *sample != 0.0));
}

#[test]
fn already_16khz_audio_passes_through() {
    // No shipped asset is natively 16 kHz, so build one by hand: resampling to the target rate
    // must be a no-op.
    let samples: Vec<f32> = (0..16_000)
        .map(|index| (index as f32 / 100.0).sin())
        .collect();
    let audio = DecodedAudio {
        samples: samples.clone(),
        sample_rate: WHISPER_SAMPLE_RATE,
        channels: 1,
    };
    assert_eq!(audio.to_whisper_input().unwrap(), samples);
}

#[test]
fn multi_channel_audio_is_mixed_down() {
    // Two channels of identical data average to the original signal.
    let frames: Vec<f32> = (0..1_000).map(|index| index as f32 / 1_000.0).collect();
    let interleaved: Vec<f32> = frames
        .iter()
        .flat_map(|sample| [*sample, *sample])
        .collect();

    let audio = DecodedAudio {
        samples: interleaved,
        sample_rate: WHISPER_SAMPLE_RATE,
        channels: 2,
    };

    assert_eq!(audio.to_whisper_input().unwrap(), frames);
}

#[test]
fn duration_survives_resampling() {
    let audio = decode("test.mp3").unwrap();
    let before = audio.duration_ms();

    let resampled = DecodedAudio {
        samples: audio.to_whisper_input().unwrap(),
        sample_rate: WHISPER_SAMPLE_RATE,
        channels: 1,
    };

    // Resampling must not change how long the clip is, beyond rounding.
    assert!(
        (resampled.duration_ms() as i64 - before as i64).unsigned_abs() <= 2,
        "duration drifted: {} ms -> {} ms",
        before,
        resampled.duration_ms()
    );
}

#[test]
fn missing_files_are_reported_not_panicked() {
    let err = audio::decode_file("assets/definitely-not-here.mp3").unwrap_err();
    assert!(matches!(err, whisper_stt::Error::Audio(_)));
}
