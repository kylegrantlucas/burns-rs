//! Building the project a command renders: base document, then `--set`, then dedicated flags.
//!
//! Precedence: built-in defaults → the selected preset *or* a strictly loaded project file →
//! repeated `--set ID=VALUE` in order → dedicated `--width/--height`, `--fps/--duration`, and
//! `--seed` flags. The result is validated once as a whole.

use std::fmt;
use std::str::FromStr;

use burns::{ConfigError, FrameRate, MotionProject, Size, Timeline};

/// One `--set ID=VALUE` assignment of a registry setting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment {
    pub id: String,
    pub value: String,
}

impl FromStr for Assignment {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let (id, value) = raw
            .split_once('=')
            .ok_or_else(|| format!("expected ID=VALUE (for example shake.speed=2), got `{raw}`"))?;
        let id = id.trim();
        if id.is_empty() {
            return Err(format!("missing setting id before `=` in `{raw}`"));
        }
        Ok(Self {
            id: id.to_owned(),
            value: value.to_owned(),
        })
    }
}

impl fmt::Display for Assignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={}", self.id, self.value)
    }
}

/// Parses a strict `NUM[/DEN]` frame rate: ASCII digits only, each term a nonzero `u32`.
pub fn parse_fps(raw: &str) -> Result<FrameRate, String> {
    let (num, den) = match raw.split_once('/') {
        Some((num, den)) => (num, Some(den)),
        None => (raw, None),
    };
    let term = |text: &str, name: &str| -> Result<u32, String> {
        if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!(
                "expected NUM or NUM/DEN with whole-number terms (for example 30 or 30000/1001), \
                 got `{raw}`"
            ));
        }
        let value: u32 = text
            .parse()
            .map_err(|_| format!("frame rate {name} `{text}` exceeds {}", u32::MAX))?;
        if value == 0 {
            return Err(format!("frame rate {name} must be at least 1, got `{raw}`"));
        }
        Ok(value)
    };
    let num = term(num, "numerator")?;
    let den = match den {
        Some(den) => term(den, "denominator")?,
        None => 1,
    };
    Ok(FrameRate::new(num, den))
}

/// Command-line edits on top of a base project.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Overrides {
    pub set: Vec<Assignment>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<FrameRate>,
    pub duration: Option<f64>,
    pub seed: Option<u32>,
}

/// Applies `overrides` to `base` and validates the complete result.
pub fn apply_overrides(
    mut project: MotionProject,
    overrides: &Overrides,
) -> Result<MotionProject, ConfigError> {
    // `--set` cannot edit timing, so the final timeline is known up front.
    project.timeline = final_timeline(project.timeline, overrides.fps, overrides.duration)?;
    for assignment in &overrides.set {
        reject_dedicated(&assignment.id)?;
    }
    project.set_many_by_id(
        overrides
            .set
            .iter()
            .map(|assignment| (assignment.id.as_str(), assignment.value.as_str())),
    )?;
    if let Some(seed) = overrides.seed {
        project.seed = seed;
    }
    project.output = Size::new(
        overrides.width.unwrap_or(project.output.width),
        overrides.height.unwrap_or(project.output.height),
    );
    project.validate()?;
    Ok(project)
}

/// The timeline after dedicated timing flags: a new frame rate alone keeps the saved duration;
/// an explicit duration replaces it and is converted at the final frame rate.
pub fn final_timeline(
    saved: Timeline,
    fps: Option<FrameRate>,
    duration: Option<f64>,
) -> Result<Timeline, ConfigError> {
    match (fps, duration) {
        (None, None) => Ok(saved),
        (Some(fps), None) => saved.retimed(fps),
        (fps, Some(seconds)) => Timeline::from_duration(fps.unwrap_or(saved.fps), seconds),
    }
}

/// Output size, timing, and format version have dedicated flags and are not registry settings.
fn reject_dedicated(id: &str) -> Result<(), ConfigError> {
    let flag = match id.split('.').next().unwrap_or(id) {
        "output" | "width" | "height" => "--width/--height",
        "timeline" | "fps" | "frames" | "duration" => "--fps/--duration",
        "version" => {
            return Err(ConfigError::invalid(
                id,
                "the project version is fixed by the file format and cannot be set",
            ));
        }
        _ => return Ok(()),
    };
    Err(ConfigError::invalid(
        id,
        format!("`--set` edits motion settings and seed only; use {flag} instead"),
    ))
}

#[cfg(test)]
mod tests {
    use burns::{CameraMotion, preset};

    use super::*;

    fn assignments(pairs: &[&str]) -> Vec<Assignment> {
        pairs.iter().map(|pair| pair.parse().unwrap()).collect()
    }

    fn build(name: &str, overrides: Overrides) -> Result<MotionProject, ConfigError> {
        apply_overrides(preset(name).unwrap(), &overrides)
    }

    #[test]
    fn final_fps_admits_shake_speed_the_saved_fps_rejects() {
        let set = assignments(&["shake.speed=2"]);
        let project = build(
            "vibrate-tight",
            Overrides {
                set: set.clone(),
                fps: Some(FrameRate::new(60, 1)),
                ..Overrides::default()
            },
        )
        .unwrap();
        assert_eq!(project.shake.speed, 2.0);
        assert_eq!(
            project.timeline,
            Timeline::new(FrameRate::new(60, 1), 480).unwrap()
        );

        // 13.7 Hz * 2 is above the 15 Hz Nyquist limit of the saved 30 fps.
        let at_saved_fps = build(
            "vibrate-tight",
            Overrides {
                set,
                ..Overrides::default()
            },
        )
        .unwrap_err();
        assert_eq!(at_saved_fps.field(), "shake.speed");
    }

    #[test]
    fn temporal_limits_use_final_timing_regardless_of_set_order() {
        // 24 fps puts vibrate-tight (13.7 Hz) over its 12 Hz limit until the speed drops.
        let lower_fps = Overrides {
            fps: Some(FrameRate::new(24, 1)),
            ..Overrides::default()
        };
        assert_eq!(
            build("vibrate-tight", lower_fps.clone())
                .unwrap_err()
                .field(),
            "shake.speed"
        );
        let project = build(
            "vibrate-tight",
            Overrides {
                set: assignments(&["shake.amount=0.5", "shake.speed=0.5"]),
                ..lower_fps
            },
        )
        .unwrap();
        assert_eq!((project.shake.amount, project.shake.speed), (0.5, 0.5));
    }

    #[test]
    fn fps_alone_preserves_saved_duration_and_explicit_duration_uses_final_fps() {
        let saved = Timeline::new(FrameRate::new(30, 1), 240).unwrap();
        let fps = |num, den| Some(FrameRate::new(num, den));
        let frames = |fps, duration| final_timeline(saved, fps, duration).unwrap().frames;

        assert_eq!(frames(None, None), 240);
        assert_eq!(frames(fps(24, 1), None), 192);
        assert_eq!(frames(fps(30000, 1001), None), 240);
        assert_eq!(frames(None, Some(5.0)), 150);
        assert_eq!(frames(fps(24, 1), Some(5.0)), 120);
        assert_eq!(frames(fps(60, 1), Some(0.001)), 1);
        assert_eq!(
            final_timeline(saved, fps(24, 1), Some(5.0)).unwrap().fps,
            FrameRate::new(24, 1)
        );
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(final_timeline(saved, None, Some(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn dedicated_seed_wins_over_set_seed() {
        let set = assignments(&["seed=5"]);
        let from_set = build(
            "shake",
            Overrides {
                set: set.clone(),
                ..Overrides::default()
            },
        )
        .unwrap();
        assert_eq!(from_set.seed, 5);
        let dedicated = build(
            "shake",
            Overrides {
                set,
                seed: Some(7),
                ..Overrides::default()
            },
        )
        .unwrap();
        assert_eq!(dedicated.seed, 7);
    }

    #[test]
    fn repeated_sets_apply_in_order() {
        let project = build(
            "kenburns",
            Overrides {
                set: assignments(&[
                    "shake.amount=0.3",
                    "shake.amount=0.7",
                    "camera.type=breathing",
                    "camera.cycles=3",
                ]),
                ..Overrides::default()
            },
        )
        .unwrap();
        assert_eq!(project.shake.amount, 0.7);
        assert!(matches!(
            project.camera,
            CameraMotion::Breathing { cycles: 3, .. }
        ));

        // Before the type change, `camera.cycles` does not exist on the transform camera.
        let reversed = build(
            "kenburns",
            Overrides {
                set: assignments(&["camera.cycles=3", "camera.type=breathing"]),
                ..Overrides::default()
            },
        )
        .unwrap_err();
        assert!(matches!(reversed, ConfigError::InactiveSetting { .. }));
    }

    #[test]
    fn set_rejects_unknown_out_of_range_and_non_motion_ids() {
        let fails = |pair: &str| {
            build(
                "kenburns",
                Overrides {
                    set: assignments(&[pair]),
                    ..Overrides::default()
                },
            )
            .unwrap_err()
        };
        assert!(matches!(
            fails("shake.sped=1"),
            ConfigError::UnknownSetting { .. }
        ));
        assert_eq!(fails("shake.speed=9").field(), "shake.speed");
        assert_eq!(fails("shake.speed=fast").field(), "shake.speed");
        for id in ["output.width", "timeline.frames", "fps", "version"] {
            assert_eq!(fails(&format!("{id}=1")).field(), id);
        }
    }

    #[test]
    fn dimensions_override_independently_and_must_be_valid() {
        let project = build(
            "kenburns",
            Overrides {
                width: Some(641),
                ..Overrides::default()
            },
        )
        .unwrap();
        assert_eq!(project.output, Size::new(641, 1080));
        let zero = build(
            "kenburns",
            Overrides {
                height: Some(0),
                ..Overrides::default()
            },
        )
        .unwrap_err();
        assert_eq!(zero.field(), "output.height");
    }

    #[test]
    fn overrides_preserve_untouched_parts_of_a_loaded_project() {
        let mut base = preset("ripple").unwrap();
        base.seed = 42;
        base.depth.invert = true;
        let project = apply_overrides(base.clone(), &Overrides::default()).unwrap();
        assert_eq!(project, base);
    }

    #[test]
    fn fps_parsing_is_strict() {
        assert_eq!(parse_fps("60"), Ok(FrameRate::new(60, 1)));
        assert_eq!(parse_fps("30000/1001"), Ok(FrameRate::new(30000, 1001)));
        for bad in [
            "",
            "0",
            "30/0",
            "0/1",
            "30.0",
            "+30",
            "-30",
            " 30",
            "30/",
            "/1",
            "30/1/1",
            "4294967296",
            "30/4294967296",
            "thirty",
        ] {
            assert!(parse_fps(bad).is_err(), "`{bad}` should be rejected");
        }
    }

    #[test]
    fn assignments_split_at_the_first_equals() {
        assert_eq!(
            "camera.type=depth".parse::<Assignment>(),
            Ok(Assignment {
                id: "camera.type".to_owned(),
                value: "depth".to_owned()
            })
        );
        assert!("shake.speed".parse::<Assignment>().is_err());
        assert!("=2".parse::<Assignment>().is_err());
    }
}
