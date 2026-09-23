//! Helpers shared by the examples: interactive start/stop, WAV writing, and playback.
//!
//! Every example pulls in a different subset of these helpers — the VAD and resampling examples
//! only want `write_wav` — so unused-item warnings are expected here and silenced for the module.
#![allow(dead_code)]

use std::fs::File;
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use rodio::{DeviceSinkBuilder, Player};

/// How the user asked us to stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// Enter was pressed on stdin.
    Enter,
    /// Ctrl+C (SIGINT) was received.
    Interrupt,
}

/// Directory recordings are written into.
pub const OUTPUT_DIR: &str = "recordings";

/// Interactive stop switch driven by either Enter on stdin or Ctrl+C.
///
/// Installing the Ctrl+C handler replaces the default "kill the process" behaviour, which is what
/// lets us finish writing the file instead of dying mid-recording.
pub struct Stopper {
    tx: Sender<Stop>,
    rx: Receiver<Stop>,
}

impl Stopper {
    /// Installs the Ctrl+C handler. Call once, early.
    pub fn arm() -> Result<Self> {
        let (tx, rx) = channel::<Stop>();
        let signal_tx = tx.clone();
        ctrlc::set_handler(move || {
            // A closed receiver just means the program is already on its way out.
            let _ = signal_tx.send(Stop::Interrupt);
        })
        .context("failed to install the Ctrl+C handler")?;
        Ok(Self { tx, rx })
    }

    /// Blocks until the user presses Enter or hits Ctrl+C.
    pub fn wait(&self) -> Stop {
        self.spawn_stdin_reader();
        self.rx.recv().unwrap_or(Stop::Enter)
    }

    /// Blocks like [`Stopper::wait`], but also raises `stop_flag` and prints a live elapsed timer.
    ///
    /// `stop_flag` is what the audio capture thread watches, so capture ends the moment the user
    /// asks it to without either side having to poll the other.
    pub fn wait_with_progress(&self, stop_flag: Option<&AtomicBool>, label: &str) -> Stop {
        self.spawn_stdin_reader();

        let started = Instant::now();
        loop {
            match self.rx.recv_timeout(Duration::from_millis(200)) {
                Ok(stop) => {
                    raise(stop_flag);
                    println!();
                    return stop;
                }
                Err(RecvTimeoutError::Timeout) => {
                    print!("\r{label} {:>5.1}s ", started.elapsed().as_secs_f32());
                    let _ = std::io::stdout().flush();
                }
                Err(RecvTimeoutError::Disconnected) => {
                    raise(stop_flag);
                    println!();
                    return Stop::Enter;
                }
            }
        }
    }

    /// Reads one line from stdin in a background thread and reports it as a stop.
    fn spawn_stdin_reader(&self) {
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let mut line = String::new();
            // `Err` here usually means stdin was closed; Ctrl+C is reported by the signal handler.
            if std::io::stdin().read_line(&mut line).is_ok() {
                let _ = tx.send(Stop::Enter);
            }
        });
    }
}

/// Raises the shared stop flag, if the caller is using one.
fn raise(stop_flag: Option<&AtomicBool>) {
    if let Some(flag) = stop_flag {
        flag.store(true, Ordering::SeqCst);
    }
}

/// Builds `recordings/<prefix>-<unix seconds>.wav`, or uses `requested` as-is.
pub fn output_path(prefix: &str, requested: Option<&str>) -> Result<PathBuf> {
    let Some(requested) = requested else {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or_default();
        return Ok(PathBuf::from(OUTPUT_DIR).join(format!("{prefix}-{stamp}.wav")));
    };

    let path = PathBuf::from(requested);
    let path = if path.parent() == Some(Path::new("")) {
        PathBuf::from(OUTPUT_DIR).join(path)
    } else {
        path
    };

    if path.extension().is_none_or(|extension| extension != "wav") {
        bail!(
            "output file must have a .wav extension, got {}",
            path.display()
        );
    }
    Ok(path)
}

/// Writes mono 16-bit PCM WAV, the layout every decoder understands.
pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;
    for sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        writer.write_sample((clamped * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Plays a WAV file on the default output device and returns only once it has finished.
pub fn play(path: &Path) -> Result<()> {
    let mut sink = DeviceSinkBuilder::open_default_sink()
        .context("cannot open the default output device for playback")?;

    let file = File::open(path)?;
    let source = rodio::Decoder::try_from(BufReader::new(file))
        .with_context(|| format!("cannot decode {}", path.display()))?;

    let player = Player::connect_new(sink.mixer());
    player.append(source);
    println!("Playing back {}...", path.display());

    // Blocks until the last queued sample has been played, then the program ends on its own.
    player.sleep_until_end();
    sink.log_on_drop(false);
    println!("Done.");
    Ok(())
}
