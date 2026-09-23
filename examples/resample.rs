//! High-quality sample-rate conversion with **rubato 5.0**.
//!
//! Whisper needs 16 kHz mono. `whisper_stt::audio::resample` gets there with cheap linear
//! interpolation, which is fine for a first pass but leaves everything above the new Nyquist
//! limit to alias back into the audio. rubato is what you reach for when the resampling itself
//! matters: an FFT-based synchronous resampler and two asynchronous ones (windowed-sinc and
//! polynomial), all of which anti-alias properly.
//!
//! The example runs two measurements:
//!
//! 1. **A real clip** (`assets/test.mp3`, 24 kHz) resampled to 16 kHz with every engine, timed.
//! 2. **An anti-aliasing probe**: a synthetic 24 kHz tone pair — 3 kHz, which must survive, and
//!    10 kHz, which sits above the 8 kHz Nyquist limit of 16 kHz and must be filtered out. Any
//!    energy left at 6 kHz (`|10 kHz − 16 kHz|`) in the output is aliasing.
//!
//! ```text
//! cargo run --example resample                                   # both measurements
//! cargo run --example resample -- assets/test.mp3 --engine sinc  # pick the engine for --out
//! cargo run --example resample -- assets/test.mp3 --out out.wav  # write a 16 kHz WAV
//! ```
//!
//! rubato 5.0 works on `audioadapter` buffers instead of `Vec<Vec<T>>`: the input is wrapped in an
//! `InterleavedSlice` and the result comes back as an owned interleaved buffer that is read
//! through the `Adapter` trait.

use std::f64::consts::PI;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use rubato::audioadapter::Adapter;
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, Fft, FixedAsync, FixedSync, PolynomialDegree, Resampler, SincInterpolationParameters,
    SincInterpolationType, WindowFunction,
};
use whisper_stt::WHISPER_SAMPLE_RATE;
use whisper_stt::audio::{decode_file, mix_down, resample};

#[path = "shared/mod.rs"]
mod shared;
use shared::write_wav;

/// Frames per resampler chunk. Bigger means less per-frame overhead but a longer delay.
const CHUNK: usize = 1024;

/// Sample rate of the synthetic probe signal.
const PROBE_RATE: u32 = 24_000;
/// Tone that must survive the conversion (below the 8 kHz Nyquist of 16 kHz).
const PROBE_KEEP_HZ: f64 = 3_000.0;
/// Tone that must be filtered out (above the 8 kHz Nyquist of 16 kHz).
const PROBE_DROP_HZ: f64 = 10_000.0;

/// One engine's resampled take, plus what it cost.
struct Take {
    name: &'static str,
    samples: Vec<f64>,
    elapsed: Duration,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let input = positional(&args)
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("assets/test.mp3"));
    let wanted = flag(&args, "--engine").unwrap_or("fft");
    let out = flag(&args, "--out").map(PathBuf::from);

    let decoded = decode_file(&input)?;
    let mono = mix_down(&decoded.samples, decoded.channels);
    if mono.is_empty() {
        return Err(anyhow!("{} contains no audio", input.display()));
    }

    println!(
        "Input:  {} — {} Hz, {} channel(s), {:.2}s",
        input.display(),
        decoded.sample_rate,
        decoded.channels,
        mono.len() as f32 / decoded.sample_rate as f32
    );
    println!("Target: {WHISPER_SAMPLE_RATE} Hz mono (what Whisper expects)\n");

    // rubato runs in f64 here; f32 is faster but noisier.
    let samples: Vec<f64> = mono.iter().map(|sample| *sample as f64).collect();
    let frames = samples.len();
    let input_adapter =
        InterleavedSlice::new(&samples, 1, frames).map_err(|err| anyhow!("{err}"))?;

    let ratio = WHISPER_SAMPLE_RATE as f64 / decoded.sample_rate as f64;
    let params = SincInterpolationParameters::new(128, WindowFunction::BlackmanHarris2)
        .oversampling_factor(256)
        .interpolation(SincInterpolationType::Cubic);

    let mut fft = Fft::<f64>::new(
        decoded.sample_rate as usize,
        WHISPER_SAMPLE_RATE as usize,
        CHUNK,
        1,
        FixedSync::Both,
    )
    .map_err(|err| anyhow!("{err}"))?;
    let mut sinc = Async::<f64>::new_sinc(ratio, 1.1, &params, CHUNK, 1, FixedAsync::Input)
        .map_err(|err| anyhow!("{err}"))?;
    let mut poly = Async::<f64>::new_poly(
        ratio,
        1.1,
        PolynomialDegree::Septic,
        CHUNK,
        1,
        FixedAsync::Input,
    )
    .map_err(|err| anyhow!("{err}"))?;

    let mut takes = vec![
        // `process_all` resamples a whole clip in one call: it resets the resampler, loops over
        // the input, and trims the startup delay, so the result is exactly the resampled frames.
        process("fft", &mut fft, &input_adapter, frames)?,
        process("sinc", &mut sinc, &input_adapter, frames)?,
        process("poly", &mut poly, &input_adapter, frames)?,
    ];

    // The built-in linear resampler, for reference.
    let started = Instant::now();
    let linear = resample(&mono, decoded.sample_rate, WHISPER_SAMPLE_RATE);
    takes.push(Take {
        name: "linear (built-in)",
        samples: linear.iter().map(|sample| *sample as f64).collect(),
        elapsed: started.elapsed(),
    });

    report(&takes, WHISPER_SAMPLE_RATE);
    alias_probe(&mut fft, &mut sinc, &mut poly)?;

    if let Some(out) = out {
        let take = takes
            .iter()
            .find(|take| take.name == wanted)
            .ok_or_else(|| anyhow!("unknown engine {wanted:?}; use fft, sinc or poly"))?;
        let samples: Vec<f32> = take.samples.iter().map(|sample| *sample as f32).collect();
        write_wav(&out, &samples, WHISPER_SAMPLE_RATE)?;
        println!(
            "\nWrote {} via the {} engine — {:.2}s at {WHISPER_SAMPLE_RATE} Hz.",
            out.display(),
            take.name,
            samples.len() as f32 / WHISPER_SAMPLE_RATE as f32
        );
    }

    Ok(())
}

/// Reads `--flag value` out of the argument list.
fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(String::as_str)
}

/// Collects the arguments that are not a flag and not a flag's value.
///
/// Without this, `--engine sinc` would hand `sinc` to the input path as well as to the flag.
fn positional(args: &[String]) -> Vec<&str> {
    let mut values = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        if matches!(arg, "--engine" | "--out") {
            index += 2;
            continue;
        }
        if !arg.starts_with('-') {
            values.push(arg);
        }
        index += 1;
    }
    values
}

/// Resamples the whole clip with one engine and times it.
fn process(
    name: &'static str,
    resampler: &mut dyn Resampler<f64>,
    input: &dyn Adapter<f64>,
    frames: usize,
) -> Result<Take> {
    let started = Instant::now();
    let output = resampler
        .process_all(input, frames, None)
        .map_err(|err| anyhow!("{err}"))?;
    let elapsed = started.elapsed();

    let samples = (0..output.frames())
        .map(|frame| output.read_sample(0, frame).unwrap_or(0.0))
        .collect();

    Ok(Take {
        name,
        samples,
        elapsed,
    })
}

/// Prints the per-engine table for the real clip.
fn report(takes: &[Take], sample_rate: u32) {
    let reference = &takes.first().expect("at least one take").samples;

    println!(
        "{:<18} {:>9} {:>10} {:>10} {:>12}",
        "engine", "frames", "duration", "time", "vs fft"
    );
    for take in takes {
        let delta = if take.name == "fft" {
            "reference".to_string()
        } else {
            format!("{:+.1} dB", deviation_db(reference, &take.samples))
        };
        println!(
            "{:<18} {:>9} {:>9.2}s {:>9.1?} {:>12}",
            take.name,
            take.samples.len(),
            take.samples.len() as f32 / sample_rate as f32,
            take.elapsed,
            delta
        );
    }

    println!("\n`vs fft` is the RMS difference from the FFT take in dB (more negative = closer),");
    println!(
        "measured after aligning the two takes, since each engine has its own delay. Closeness"
    );
    println!("is not quality — see the anti-aliasing probe below for that.");
}

/// Measures how much of an out-of-band tone each engine leaks back as aliasing.
///
/// A {PROBE_KEEP_HZ} Hz tone has to survive the conversion untouched; a {PROBE_DROP_HZ} Hz tone
/// sits above the 8 kHz Nyquist limit of 16 kHz and has to be filtered out. Whatever shows up at
/// `|{PROBE_DROP_HZ} − 16 kHz|` = 6 kHz afterwards is aliasing, and it is exactly the kind of
/// artifact that makes a transcript worse.
fn alias_probe(fft: &mut Fft<f64>, sinc: &mut Async<f64>, poly: &mut Async<f64>) -> Result<()> {
    let seconds = 1.0;
    let frames = (PROBE_RATE as f64 * seconds) as usize;
    let probe: Vec<f64> = (0..frames)
        .map(|index| {
            let t = index as f64 / PROBE_RATE as f64;
            0.5 * (2.0 * PI * PROBE_KEEP_HZ * t).sin() + 0.5 * (2.0 * PI * PROBE_DROP_HZ * t).sin()
        })
        .collect();

    let input = InterleavedSlice::new(&probe, 1, frames).map_err(|err| anyhow!("{err}"))?;

    let mut takes = vec![
        process("fft", fft, &input, frames)?,
        process("sinc", sinc, &input, frames)?,
        process("poly", poly, &input, frames)?,
    ];

    // The linear resampler operates on f32, so round-trip the probe through it.
    let probe_f32: Vec<f32> = probe.iter().map(|sample| *sample as f32).collect();
    let linear = resample(&probe_f32, PROBE_RATE, WHISPER_SAMPLE_RATE);
    takes.push(Take {
        name: "linear (built-in)",
        samples: linear.iter().map(|sample| *sample as f64).collect(),
        elapsed: Duration::ZERO,
    });

    println!(
        "\nAnti-aliasing probe — {PROBE_RATE} Hz signal, {} kHz tone (keep) + {} kHz tone (drop)",
        PROBE_KEEP_HZ / 1000.0,
        PROBE_DROP_HZ / 1000.0
    );
    println!(
        "{:<18} {:>12} {:>14}",
        "engine", "3 kHz kept", "6 kHz alias"
    );
    for take in &takes {
        println!(
            "{:<18} {:>12} {:>14}",
            take.name,
            format_db(level_db(&take.samples, WHISPER_SAMPLE_RATE, PROBE_KEEP_HZ)),
            format_db(level_db(&take.samples, WHISPER_SAMPLE_RATE, 6_000.0))
        );
    }
    println!("\nThe 6 kHz column is aliasing: lower is better. Linear interpolation folds the");
    println!("10 kHz tone straight back onto the audio, which the filtered engines suppress.");

    Ok(())
}

/// RMS difference between two takes in dB, after aligning them on the best of a few lags.
///
/// Each resampler has its own delay, so comparing sample-for-sample without alignment would
/// mostly measure that delay rather than the resampling quality.
fn deviation_db(reference: &[f64], other: &[f64]) -> f64 {
    let frames = reference.len().min(other.len());
    if frames <= 16 {
        return 0.0;
    }

    let mut best = f64::INFINITY;
    for lag in -8..=8_i64 {
        let (diff_power, ref_power) = reference[..frames].iter().enumerate().fold(
            (0.0, 0.0),
            |(diff, power), (index, left)| {
                let shifted = index
                    .checked_add_signed(lag as isize)
                    .and_then(|index| other.get(index))
                    .copied()
                    .unwrap_or(0.0);
                let delta = left - shifted;
                (diff + delta * delta, power + left * left)
            },
        );
        if ref_power > 0.0 && diff_power > 0.0 {
            best = best.min(10.0 * (diff_power / ref_power).log10());
        }
    }

    if best.is_infinite() {
        f64::NEG_INFINITY
    } else {
        best
    }
}

/// Formats a level for the table, clamping the f64 noise floor to something readable.
fn format_db(level: f64) -> String {
    if level < -120.0 {
        "< -120 dB".to_string()
    } else {
        format!("{level:.1} dB")
    }
}

/// Level of a single frequency in a signal, in dB relative to a unit-amplitude sine.
///
/// Goertzel's algorithm, Hann-windowed to keep leakage from other tones out of the estimate.
fn level_db(samples: &[f64], sample_rate: u32, frequency: f64) -> f64 {
    let frames = samples.len();
    if frames == 0 {
        return f64::NEG_INFINITY;
    }

    let omega = 2.0 * PI * frequency / sample_rate as f64;
    let coefficient = 2.0 * omega.cos();
    let mut previous = 0.0;
    let mut before_that = 0.0;
    let mut window_power = 0.0;

    for (index, sample) in samples.iter().enumerate() {
        // Hann window, amplitude-corrected by the 0.5 mean of the window.
        let window = 0.5 - 0.5 * (2.0 * PI * index as f64 / frames as f64).cos();
        window_power += window;
        let current = sample * window + coefficient * previous - before_that;
        before_that = previous;
        previous = current;
    }

    let real = previous - before_that * omega.cos();
    let imaginary = before_that * omega.sin();
    let amplitude =
        2.0 * (real * real + imaginary * imaginary).sqrt() / window_power.max(f64::EPSILON);

    if amplitude <= 0.0 {
        f64::NEG_INFINITY
    } else {
        20.0 * amplitude.log10()
    }
}
