//! Throttled progress on stderr: a redrawn line on a terminal, sparse lines otherwise.

use std::io::{IsTerminal, Write, stderr};
use std::time::{Duration, Instant};

use burns_depth::DepthProgress;

/// Shortest interval between terminal redraws.
const REDRAW: Duration = Duration::from_millis(100);

/// Frame progress for one export.
pub struct FrameProgress {
    terminal: bool,
    started: Instant,
    last_draw: Option<Instant>,
    /// Last reported tenth, for non-terminal output.
    last_tenth: Option<u64>,
    drawn: bool,
}

impl FrameProgress {
    pub fn new() -> Self {
        Self {
            terminal: stderr().is_terminal(),
            started: Instant::now(),
            last_draw: None,
            last_tenth: None,
            drawn: false,
        }
    }

    pub fn update(&mut self, completed: u32, total: u32) {
        let percent = u64::from(completed) * 100 / u64::from(total.max(1));
        if self.terminal {
            let now = Instant::now();
            if self
                .last_draw
                .is_some_and(|last| now - last < REDRAW && completed < total)
            {
                return;
            }
            self.last_draw = Some(now);
            self.drawn = true;
            let mut err = stderr().lock();
            let _ = write!(
                err,
                "\r\x1b[2Kframe {completed}/{total} ({percent}%) {:.1}s",
                self.started.elapsed().as_secs_f64()
            );
            let _ = err.flush();
        } else if self.last_tenth != Some(percent / 10) {
            self.last_tenth = Some(percent / 10);
            eprintln!(
                "frame {completed}/{total} ({percent}%) {:.1}s",
                self.started.elapsed().as_secs_f64()
            );
        }
    }

    /// Ends a redrawn line so later messages start on their own line.
    pub fn finish(&mut self) {
        if self.drawn {
            eprintln!();
            self.drawn = false;
        }
    }
}

/// Reports depth cache, download, and inference events.
pub fn depth_reporter() -> impl FnMut(DepthProgress) {
    let started = Instant::now();
    let mut last_tenth = None;
    move |event| {
        let elapsed = started.elapsed().as_secs_f64();
        match event {
            DepthProgress::CacheHit => eprintln!("depth: using the cached map for this image"),
            DepthProgress::Verifying => eprintln!("depth: verifying the cached model file"),
            DepthProgress::Downloading { received, total } => {
                let percent = received * 100 / total.max(1);
                if last_tenth != Some(percent / 10) {
                    last_tenth = Some(percent / 10);
                    eprintln!(
                        "depth: downloading model {:.1}/{:.1} MB ({percent}%) {elapsed:.1}s",
                        received as f64 / 1e6,
                        total as f64 / 1e6
                    );
                }
            }
            DepthProgress::Loading => eprintln!("depth: loading ONNX Runtime and the model"),
            DepthProgress::Inferring => eprintln!("depth: estimating depth {elapsed:.1}s"),
            DepthProgress::Saving => eprintln!("depth: caching the generated map {elapsed:.1}s"),
            DepthProgress::PersistenceWarning { message } => eprintln!("warning: {message}"),
        }
    }
}
