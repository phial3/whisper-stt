//! Throughput of the audio path — the part of a transcription that costs CPU before Whisper ever
//! sees a sample.
//!
//! ```text
//! cargo bench                     # everything here
//! cargo bench -- resample         # just the resamplers
//! ```
//!
//! Every measurement is reported against the number of input samples, so the numbers scale to
//! whatever clip length you actually feed the model. Decoding is included because it is the other
//! half of the same cost, and it is the easiest stage to make accidentally quadratic.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use whisper_stt::WHISPER_SAMPLE_RATE;
use whisper_stt::audio::{DecodedAudio, decode_file, mix_down, resample};

/// Seconds of synthetic audio each sample-rate benchmark works on.
const SECONDS: usize = 10;

/// Builds `seconds` of a 440 Hz tone at `sample_rate`, interleaved across `channels`.
fn tone(sample_rate: u32, seconds: usize, channels: usize) -> Vec<f32> {
    let frames = sample_rate as usize * seconds;
    let mut samples = Vec::with_capacity(frames * channels);
    for frame in 0..frames {
        let value = (2.0 * std::f32::consts::PI * 440.0 * frame as f32 / sample_rate as f32).sin();
        samples.resize(samples.len() + channels, value);
    }
    samples
}

/// Runs `group` over every interesting source rate, mono.
///
/// 48 kHz and 44.1 kHz are what capture devices and music deliver, 24 kHz is a common podcast and
/// VoIP rate, and 8 kHz is the telephone floor.
fn resamplers(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    for rate in [8_000u32, 24_000, 44_100, 48_000] {
        let mono = tone(rate, SECONDS, 1);
        let decoded = DecodedAudio {
            samples: mono.clone(),
            sample_rate: rate,
            channels: 1,
        };

        group.throughput(Throughput::Elements(mono.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("resample", format!("{rate} Hz")),
            &mono,
            |bencher, mono| bencher.iter(|| resample(mono, rate, WHISPER_SAMPLE_RATE).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("to_whisper_input", format!("{rate} Hz")),
            &decoded,
            |bencher, decoded| bencher.iter(|| decoded.to_whisper_input().unwrap()),
        );
    }
}

/// Downmixing, which every multi-channel file pays before it is resampled.
fn downmixes(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    for channels in [2usize, 6] {
        let interleaved = tone(48_000, SECONDS, channels);
        group.throughput(Throughput::Elements(interleaved.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("mix_down", format!("{channels} channels")),
            &interleaved,
            |bencher, interleaved| bencher.iter(|| mix_down(interleaved, channels)),
        );
    }
}

/// Decoding plus resampling on a real file, so one number covers the whole path.
fn decoding(criterion: &mut Criterion) {
    criterion
        .benchmark_group("decode")
        .bench_function("assets/test.mp3", |bencher| {
            bencher.iter(|| {
                let audio = decode_file("assets/test.mp3").unwrap();
                audio.to_whisper_input().unwrap()
            });
        });
}

fn benches(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("resample");
    resamplers(&mut group);
    group.finish();

    let mut group = criterion.benchmark_group("mix_down");
    downmixes(&mut group);
    group.finish();

    decoding(criterion);
}

criterion_group!(audio, benches);
criterion_main!(audio);
