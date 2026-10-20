//! Built-in motion presets and the default constructors every variant switch
//! and preset installs.

use serde::Serialize;

use crate::settings::{
    CameraKind, CameraMotion, ConfigError, DepthAdjustments, DepthMotion, Easing, MotionProject,
    ShakeProfile, ShakeSettings, Size, View, WarpKind, WarpSettings,
};
use crate::timeline::{FrameRate, Timeline};

/// Project, preset, and settings schema format version.
pub const PRESET_VERSION: u32 = 1;

/// Preset of [`MotionProject::default`].
pub const DEFAULT_PRESET: &str = "kenburns";

pub const DEFAULT_OUTPUT: Size = Size::new(1920, 1080);

pub const DEFAULT_TIMELINE: Timeline = Timeline {
    fps: FrameRate::new(30, 1),
    frames: 240,
};

const KENBURNS_START: View = View::new(1.04, 0.35, 0.5, 0.0);
const KENBURNS_END: View = View::new(1.22, 0.65, 0.5, 0.0);

/// Built-in preset metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PresetDescriptor {
    pub name: &'static str,
    pub label: &'static str,
    /// Needs a depth map to render.
    pub requires_depth: bool,
    /// Loops seamlessly (periodic over the timeline) rather than playing once.
    pub periodic: bool,
}

const fn descriptor(
    name: &'static str,
    label: &'static str,
    requires_depth: bool,
    periodic: bool,
) -> PresetDescriptor {
    PresetDescriptor {
        name,
        label,
        requires_depth,
        periodic,
    }
}

static PRESETS: [PresetDescriptor; 16] = [
    descriptor("kenburns", "Ken Burns", false, false),
    descriptor("shake", "Shake", false, false),
    descriptor("vibrate", "Vibrate", false, false),
    descriptor("vibrate-tight", "Vibrate (tight)", false, false),
    descriptor("tilt", "Tilt", false, false),
    descriptor("orbital", "Orbital", true, true),
    descriptor("circle", "Circle", true, true),
    descriptor("dolly", "Dolly", true, true),
    descriptor("dolly-out", "Dolly out", true, false),
    descriptor("pan", "Pan", false, false),
    descriptor("pull-back", "Pull back", false, false),
    descriptor("roll", "Roll", false, false),
    descriptor("breathing", "Breathing", false, true),
    descriptor("figure-eight", "Figure eight", false, true),
    descriptor("ripple", "Ripple", false, true),
    descriptor("heat-shimmer", "Heat shimmer", false, true),
];

/// Every built-in preset, in display order.
pub fn presets() -> &'static [PresetDescriptor] {
    &PRESETS
}

/// The motion blocks a preset installs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PresetMotion {
    pub camera: CameraMotion,
    pub shake: ShakeSettings,
    pub warp: WarpSettings,
}

/// Motion blocks of the named preset.
pub fn preset_motion(name: &str) -> Result<PresetMotion, ConfigError> {
    motion(name).ok_or_else(|| ConfigError::UnknownPreset {
        name: name.to_owned(),
        expected: PRESETS
            .iter()
            .map(|preset| preset.name)
            .collect::<Vec<_>>()
            .join(", "),
    })
}

/// Default project with the named preset applied.
pub fn preset(name: &str) -> Result<MotionProject, ConfigError> {
    let mut project = MotionProject::default();
    project.apply_preset(name)?;
    Ok(project)
}

fn motion(name: &str) -> Option<PresetMotion> {
    let camera_only = |camera| PresetMotion {
        camera,
        shake: ShakeSettings::default(),
        warp: WarpSettings::default(),
    };
    let shaken = |profile| PresetMotion {
        camera: CameraMotion::Still {
            view: View::centered(1.0),
        },
        shake: ShakeSettings {
            enabled: true,
            profile,
            ..ShakeSettings::default()
        },
        warp: WarpSettings::default(),
    };
    let warped = |kind| PresetMotion {
        camera: CameraMotion::Still {
            view: View::centered(1.04),
        },
        shake: ShakeSettings::default(),
        warp: WarpSettings::default_for(kind, true),
    };
    let transform = |start, end| CameraMotion::Transform {
        start,
        end,
        easing: Easing::Cosine,
    };

    Some(match name {
        "kenburns" => camera_only(CameraMotion::default_for(CameraKind::Transform)),
        "shake" => shaken(ShakeProfile::Shake),
        "vibrate" => shaken(ShakeProfile::Vibrate),
        "vibrate-tight" => shaken(ShakeProfile::VibrateTight),
        "tilt" => camera_only(CameraMotion::default_for(CameraKind::Tilt)),
        "orbital" => camera_only(CameraMotion::depth(DepthMotion::Orbital)),
        "circle" => camera_only(CameraMotion::depth(DepthMotion::Circle)),
        "dolly" => camera_only(CameraMotion::depth(DepthMotion::Dolly)),
        "dolly-out" => camera_only(CameraMotion::depth(DepthMotion::DollyOut)),
        "pan" => camera_only(transform(
            View::new(1.15, 0.15, 0.5, 0.0),
            View::new(1.15, 0.85, 0.5, 0.0),
        )),
        "pull-back" => camera_only(transform(KENBURNS_END, KENBURNS_START)),
        "roll" => camera_only(transform(
            View::new(1.06, 0.5, 0.5, -1.5),
            View::new(1.06, 0.5, 0.5, 1.5),
        )),
        "breathing" => camera_only(CameraMotion::default_for(CameraKind::Breathing)),
        "figure-eight" => camera_only(CameraMotion::default_for(CameraKind::FigureEight)),
        "ripple" => warped(WarpKind::Ripple),
        "heat-shimmer" => warped(WarpKind::HeatShimmer),
        _ => return None,
    })
}

impl CameraMotion {
    /// Parameters installed when switching to `kind`.
    pub const fn default_for(kind: CameraKind) -> Self {
        match kind {
            CameraKind::Still => Self::Still {
                view: View::centered(1.0),
            },
            CameraKind::Transform => Self::Transform {
                start: KENBURNS_START,
                end: KENBURNS_END,
                easing: Easing::Cosine,
            },
            CameraKind::Tilt => Self::Tilt {
                amount: 1.0,
                easing: Easing::Cosine,
            },
            CameraKind::Breathing => Self::Breathing {
                base_zoom: 1.06,
                amplitude: 0.04,
                cycles: 1,
            },
            CameraKind::FigureEight => Self::FigureEight {
                zoom: 1.10,
                amplitude: [0.30, 0.20],
                cycles: 1,
            },
            CameraKind::Depth => Self::depth(DepthMotion::Orbital),
        }
    }

    /// Depth camera with the built-in parameters for `motion`.
    pub const fn depth(motion: DepthMotion) -> Self {
        let (height, steady, focus, zoom) = match motion {
            DepthMotion::Orbital => (0.20, 0.30, 0.30, 0.98),
            DepthMotion::Circle => (0.20, 0.30, 0.0, 1.0),
            DepthMotion::Dolly | DepthMotion::DollyOut => (0.30, 0.35, 0.35, 0.95),
        };
        Self::Depth {
            motion,
            height,
            steady,
            focus,
            zoom,
            amount: 1.0,
            cycles: 1,
        }
    }
}

impl Default for CameraMotion {
    fn default() -> Self {
        Self::default_for(CameraKind::Transform)
    }
}

impl WarpSettings {
    /// Parameters installed when switching to `kind`, with `enabled` kept.
    pub const fn default_for(kind: WarpKind, enabled: bool) -> Self {
        match kind {
            WarpKind::Ripple => Self::Ripple {
                enabled,
                amplitude: 0.006,
                wavelength: 0.18,
                center: [0.5, 0.5],
                cycles: 2,
            },
            WarpKind::HeatShimmer => Self::HeatShimmer {
                enabled,
                amplitude: 0.003,
                spatial_scale: 1.0,
                cycles: 2,
            },
        }
    }
}

/// Disabled default ripple.
impl Default for WarpSettings {
    fn default() -> Self {
        Self::default_for(WarpKind::Ripple, false)
    }
}

/// Disabled handheld shake at unit amount and speed.
impl Default for ShakeSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            profile: ShakeProfile::Shake,
            amount: 1.0,
            speed: 1.0,
        }
    }
}

/// Ken Burns at 1920x1080, 30 fps, 240 frames, seed 0, depth not inverted.
impl Default for MotionProject {
    fn default() -> Self {
        Self {
            version: PRESET_VERSION,
            output: DEFAULT_OUTPUT,
            timeline: DEFAULT_TIMELINE,
            seed: 0,
            camera: CameraMotion::default(),
            depth: DepthAdjustments::default(),
            shake: ShakeSettings::default(),
            warp: WarpSettings::default(),
        }
    }
}
