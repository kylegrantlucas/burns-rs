//! Frame-exact timing: rational frame rates, frame counts, and per-frame time,
//! progress, and phase. Everything derives from frame indices; there is no
//! wall clock.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::settings::ConfigError;

/// Rational frames per second, `num / den` (for example `30000/1001`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameRate {
    pub num: u32,
    pub den: u32,
}

impl FrameRate {
    pub const fn new(num: u32, den: u32) -> Self {
        Self { num, den }
    }

    /// Both terms must be nonzero.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.num == 0 {
            return Err(ConfigError::invalid(
                "num",
                "frame rate numerator must be at least 1",
            ));
        }
        if self.den == 0 {
            return Err(ConfigError::invalid(
                "den",
                "frame rate denominator must be at least 1",
            ));
        }
        Ok(())
    }

    /// Frames per second as a float.
    pub fn as_f64(&self) -> f64 {
        f64::from(self.num) / f64::from(self.den)
    }

    /// Half the frame rate: periodic motion must stay strictly below this.
    pub fn nyquist_hz(&self) -> f64 {
        self.as_f64() / 2.0
    }
}

impl fmt::Display for FrameRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

/// `frames` output frames at `fps`. Frame `n` is shown at `n * den / num` seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timeline {
    pub fps: FrameRate,
    pub frames: u32,
}

/// Timing of a single frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameTime {
    /// Frame index `n`, `0 <= n < N`.
    pub index: u32,
    /// Presentation time `t = n * den / num`.
    pub seconds: f64,
    /// Endpoint-inclusive progress `u = n / (N - 1)`; `0` when `N == 1`.
    /// The first frame is at `0` and the last at `1`, for one-shot motion.
    pub progress: f64,
    /// Loop phase `tau = n / N`, in `[0, 1)`. Frame `N` would equal frame `0`,
    /// so periodic motion loops seamlessly.
    pub phase: f64,
}

impl Timeline {
    /// Validated timeline with an explicit frame count.
    pub fn new(fps: FrameRate, frames: u32) -> Result<Self, ConfigError> {
        let timeline = Self { fps, frames };
        timeline.validate()?;
        Ok(timeline)
    }

    /// Frame count `N = max(1, round(seconds * num / den))` for a finite,
    /// positive duration. The actual duration is `N / fps`.
    pub fn from_duration(fps: FrameRate, seconds: f64) -> Result<Self, ConfigError> {
        fps.validate().map_err(|e| e.within("fps"))?;
        if !seconds.is_finite() || seconds <= 0.0 {
            return Err(ConfigError::invalid(
                "duration",
                format!("must be a finite, positive number of seconds, got {seconds}"),
            ));
        }
        let frames = (seconds * f64::from(fps.num) / f64::from(fps.den)).round();
        if !frames.is_finite() || frames > f64::from(u32::MAX) {
            return Err(ConfigError::invalid(
                "duration",
                format!(
                    "{seconds} s at {fps} fps needs {frames} frames, more than the maximum {}",
                    u32::MAX
                ),
            ));
        }
        // `frames` is finite, non-negative, and at most `u32::MAX`.
        Ok(Self {
            fps,
            frames: (frames as u32).max(1),
        })
    }

    /// Valid frame rate and at least one frame.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.fps.validate().map_err(|e| e.within("fps"))?;
        if self.frames == 0 {
            return Err(ConfigError::invalid(
                "frames",
                "timeline must contain at least 1 frame",
            ));
        }
        Ok(())
    }

    /// Actual duration `N * den / num`.
    pub fn duration_seconds(&self) -> f64 {
        f64::from(self.frames) * f64::from(self.fps.den) / f64::from(self.fps.num)
    }

    /// Same duration at a different frame rate: `N' = max(1, round(N * fps' / fps))`,
    /// computed exactly in integers (halves round up).
    pub fn retimed(&self, fps: FrameRate) -> Result<Self, ConfigError> {
        self.validate()?;
        fps.validate().map_err(|e| e.within("fps"))?;
        let numer = u128::from(self.frames) * u128::from(self.fps.den) * u128::from(fps.num);
        let denom = u128::from(self.fps.num) * u128::from(fps.den);
        let frames = (2 * numer + denom) / (2 * denom);
        let frames = u32::try_from(frames.max(1)).map_err(|_| {
            ConfigError::invalid(
                "fps",
                format!(
                    "keeping {:.3} s at {fps} fps needs {frames} frames, more than the maximum {}",
                    self.duration_seconds(),
                    u32::MAX
                ),
            )
        })?;
        Ok(Self { fps, frames })
    }

    /// Time, progress, and phase of frame `index`; rejects `index >= frames`.
    pub fn frame_time(&self, index: u32) -> Result<FrameTime, ConfigError> {
        self.validate()?;
        if index >= self.frames {
            return Err(ConfigError::invalid(
                "frame",
                format!(
                    "frame index {index} is out of range; the timeline has {} frames (0..={})",
                    self.frames,
                    self.frames - 1
                ),
            ));
        }
        let n = f64::from(index);
        let count = f64::from(self.frames);
        Ok(FrameTime {
            index,
            seconds: n * f64::from(self.fps.den) / f64::from(self.fps.num),
            progress: if self.frames == 1 {
                0.0
            } else {
                n / (count - 1.0)
            },
            phase: n / count,
        })
    }
}
