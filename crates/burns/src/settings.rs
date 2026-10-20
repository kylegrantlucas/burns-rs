//! Motion project configuration: typed, strictly deserialized project files,
//! validation (ranges and temporal Nyquist limits), and the settings
//! descriptor registry that every front end uses for ranges, defaults,
//! options, and visibility.

use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::presets::{PRESET_VERSION, preset_motion};
use crate::timeline::{FrameRate, Timeline};

// ---------------------------------------------------------------------------
// Errors

/// Configuration error naming the offending field, setting id, or preset.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    /// A value violates its range or contract.
    #[error("invalid `{field}`: {message}")]
    Invalid { field: String, message: String },
    /// Project JSON is malformed, has unknown or missing fields, or wrong types.
    /// `field` is the serde path of the failure (`.` for the document root).
    #[error("malformed project JSON at `{field}`: {message}")]
    Json { field: String, message: String },
    /// No descriptor has this id.
    #[error("unknown setting `{id}`")]
    UnknownSetting { id: String },
    /// The id exists but belongs to a camera/warp variant the project does not use.
    #[error("setting `{id}` does not apply to this project; it requires {requires}")]
    InactiveSetting { id: String, requires: String },
    /// No built-in preset has this name.
    #[error("unknown preset `{name}`; expected one of: {expected}")]
    UnknownPreset { name: String, expected: String },
}

impl ConfigError {
    pub fn invalid(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Invalid {
            field: field.into(),
            message: message.into(),
        }
    }

    /// Field path, setting id, or `preset` this error refers to.
    pub fn field(&self) -> &str {
        match self {
            Self::Invalid { field, .. } | Self::Json { field, .. } => field,
            Self::UnknownSetting { id } | Self::InactiveSetting { id, .. } => id,
            Self::UnknownPreset { .. } => "preset",
        }
    }

    /// Prefixes an [`ConfigError::Invalid`] field with its parent path,
    /// e.g. `width` within `output` becomes `output.width`.
    pub fn within(self, parent: &str) -> Self {
        match self {
            Self::Invalid { field, message } => Self::Invalid {
                field: format!("{parent}.{field}"),
                message,
            },
            other => other,
        }
    }
}

// ---------------------------------------------------------------------------
// Project types

/// Pixel dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}

impl Size {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Nonzero dimensions whose RGBA8 buffer length is addressable.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.rgba_len().map(|_| ())
    }

    /// `width * height`, checked against `usize` overflow; rejects zero dimensions.
    pub fn pixel_count(&self) -> Result<usize, ConfigError> {
        if self.width == 0 {
            return Err(ConfigError::invalid("width", "must be at least 1 pixel"));
        }
        if self.height == 0 {
            return Err(ConfigError::invalid("height", "must be at least 1 pixel"));
        }
        usize::try_from(self.width)
            .ok()
            .zip(usize::try_from(self.height).ok())
            .and_then(|(w, h)| w.checked_mul(h))
            .ok_or_else(|| self.too_large())
    }

    /// Byte length of an RGBA8 buffer, at most `isize::MAX`.
    pub fn rgba_len(&self) -> Result<usize, ConfigError> {
        self.pixel_count()?
            .checked_mul(4)
            .filter(|len| isize::try_from(*len).is_ok())
            .ok_or_else(|| self.too_large())
    }

    fn too_large(&self) -> ConfigError {
        ConfigError::invalid(
            "width",
            format!(
                "{}x{} pixels exceeds the addressable RGBA buffer size",
                self.width, self.height
            ),
        )
    }
}

/// Camera framing in output-normalized coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct View {
    /// Magnification of the cover-cropped frame; `1` shows the whole crop.
    pub zoom: f32,
    /// Fraction of available horizontal crop travel: `0` left, `1` right.
    pub x: f32,
    /// Fraction of available vertical crop travel: `0` top, `1` bottom.
    pub y: f32,
    pub roll_degrees: f32,
}

impl View {
    pub const fn new(zoom: f32, x: f32, y: f32, roll_degrees: f32) -> Self {
        Self {
            zoom,
            x,
            y,
            roll_degrees,
        }
    }

    /// Centered, unrolled view at `zoom`.
    pub const fn centered(zoom: f32) -> Self {
        Self::new(zoom, 0.5, 0.5, 0.0)
    }
}

/// Defines a fieldless choice enum whose serialized value, descriptor option
/// value, label, and index order come from one table.
macro_rules! choice_enum {
    (
        $(#[$meta:meta])*
        pub enum $name:ident {
            $( $(#[$vmeta:meta])* $variant:ident => $value:literal, $label:literal; )+
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum $name {
            $( $(#[$vmeta])* #[serde(rename = $value)] $variant, )+
        }

        impl $name {
            /// Every option, in stable descriptor index order.
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            /// Stable serialized value.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $value,)+
                }
            }

            pub const fn label(self) -> &'static str {
                match self {
                    $(Self::$variant => $label,)+
                }
            }

            /// Descriptor choice index.
            pub const fn index(self) -> u32 {
                self as u32
            }

            pub fn from_index(index: u32) -> Option<Self> {
                Self::ALL.get(index as usize).copied()
            }

            /// Parses the stable serialized value.
            pub fn from_value(value: &str) -> Option<Self> {
                Self::ALL.iter().copied().find(|v| v.as_str() == value)
            }

            fn choice_options() -> Vec<ChoiceOption> {
                Self::ALL
                    .iter()
                    .map(|v| ChoiceOption {
                        value: v.as_str().to_owned(),
                        label: v.label().to_owned(),
                    })
                    .collect()
            }
        }
    };
}

choice_enum! {
    /// Interpolation curve for one-shot motion.
    pub enum Easing {
        Cosine => "cosine", "Cosine (ease in and out)";
        Linear => "linear", "Linear";
    }
}

choice_enum! {
    /// Depth parallax camera path.
    pub enum DepthMotion {
        Orbital => "orbital", "Orbital";
        Circle => "circle", "Circle";
        Dolly => "dolly", "Dolly";
        /// One-shot dolly; ignores `cycles`.
        DollyOut => "dolly-out", "Dolly out";
    }
}

choice_enum! {
    /// Procedural camera shake character.
    pub enum ShakeProfile {
        Shake => "shake", "Handheld shake";
        Vibrate => "vibrate", "Vibrate";
        VibrateTight => "vibrate-tight", "Vibrate (tight)";
    }
}

choice_enum! {
    /// Discriminant of [`CameraMotion`], serialized as its `type`.
    pub enum CameraKind {
        Still => "still", "Still";
        Transform => "transform", "Transform";
        Tilt => "tilt", "Tilt";
        Breathing => "breathing", "Breathing";
        FigureEight => "figure-eight", "Figure eight";
        Depth => "depth", "Depth parallax";
    }
}

choice_enum! {
    /// Discriminant of [`WarpSettings`], serialized as its `type`.
    pub enum WarpKind {
        Ripple => "ripple", "Ripple";
        HeatShimmer => "heat-shimmer", "Heat shimmer";
    }
}

impl ShakeProfile {
    /// Highest base noise frequency of the profile at `speed = 1`.
    pub const fn max_frequency_hz(self) -> f64 {
        match self {
            Self::Shake => 3.7,
            Self::Vibrate => 13.1,
            Self::VibrateTight => 13.7,
        }
    }
}

/// Camera path over the timeline.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum CameraMotion {
    Still {
        view: View,
    },
    Transform {
        start: View,
        end: View,
        easing: Easing,
    },
    Tilt {
        amount: f32,
        easing: Easing,
    },
    Breathing {
        base_zoom: f32,
        amplitude: f32,
        cycles: u32,
    },
    FigureEight {
        zoom: f32,
        amplitude: [f32; 2],
        cycles: u32,
    },
    Depth {
        motion: DepthMotion,
        height: f32,
        steady: f32,
        focus: f32,
        zoom: f32,
        amount: f32,
        cycles: u32,
    },
}

impl CameraMotion {
    pub const fn kind(&self) -> CameraKind {
        match self {
            Self::Still { .. } => CameraKind::Still,
            Self::Transform { .. } => CameraKind::Transform,
            Self::Tilt { .. } => CameraKind::Tilt,
            Self::Breathing { .. } => CameraKind::Breathing,
            Self::FigureEight { .. } => CameraKind::FigureEight,
            Self::Depth { .. } => CameraKind::Depth,
        }
    }
}

/// Depth map interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DepthAdjustments {
    /// Treat dark values as near instead of white.
    pub invert: bool,
}

/// Procedural camera shake layered over the camera path.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShakeSettings {
    pub enabled: bool,
    pub profile: ShakeProfile,
    pub amount: f32,
    pub speed: f32,
}

/// Image-space warp layered over the camera path.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum WarpSettings {
    Ripple {
        enabled: bool,
        amplitude: f32,
        wavelength: f32,
        center: [f32; 2],
        cycles: u32,
    },
    HeatShimmer {
        enabled: bool,
        amplitude: f32,
        spatial_scale: f32,
        cycles: u32,
    },
}

impl WarpSettings {
    pub const fn kind(&self) -> WarpKind {
        match self {
            Self::Ripple { .. } => WarpKind::Ripple,
            Self::HeatShimmer { .. } => WarpKind::HeatShimmer,
        }
    }

    pub const fn enabled(&self) -> bool {
        match *self {
            Self::Ripple { enabled, .. } | Self::HeatShimmer { enabled, .. } => enabled,
        }
    }
}

/// Complete, source-independent motion project. Every field is mandatory in
/// JSON and unknown fields are rejected at every level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MotionProject {
    pub version: u32,
    pub output: Size,
    pub timeline: Timeline,
    pub seed: u32,
    pub camera: CameraMotion,
    pub depth: DepthAdjustments,
    pub shake: ShakeSettings,
    pub warp: WarpSettings,
}

// ---------------------------------------------------------------------------
// Ranges: the single source for validation and descriptor kinds.

#[derive(Debug, Clone, Copy)]
struct FloatRange {
    min: f32,
    max: f32,
    step: f32,
}

impl FloatRange {
    const fn new(min: f32, max: f32, step: f32) -> Self {
        Self { min, max, step }
    }

    fn check(self, field: &str, value: f32) -> Result<(), ConfigError> {
        check_float(field, value, self.min, self.max)
    }

    fn kind(self) -> SettingKind {
        SettingKind::Float {
            min: self.min,
            max: self.max,
            step: self.step,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct IntRange {
    min: i64,
    max: i64,
}

impl IntRange {
    fn check(self, field: &str, value: u32) -> Result<(), ConfigError> {
        check_int(field, i64::from(value), self.min, self.max)
    }

    fn kind(self) -> SettingKind {
        SettingKind::Int {
            min: self.min,
            max: self.max,
        }
    }
}

fn check_float(field: &str, value: f32, min: f32, max: f32) -> Result<(), ConfigError> {
    if !value.is_finite() {
        return Err(ConfigError::invalid(
            field,
            format!("must be a finite number, got {value}"),
        ));
    }
    if value < min || value > max {
        return Err(ConfigError::invalid(
            field,
            format!("must be between {min} and {max}, got {value}"),
        ));
    }
    Ok(())
}

fn check_int(field: &str, value: i64, min: i64, max: i64) -> Result<(), ConfigError> {
    if value < min || value > max {
        return Err(ConfigError::invalid(
            field,
            format!("must be between {min} and {max}, got {value}"),
        ));
    }
    Ok(())
}

const POSITION: FloatRange = FloatRange::new(0.0, 1.0, 0.001);
const ZOOM: FloatRange = FloatRange::new(1.0, 10.0, 0.01);
const ROLL: FloatRange = FloatRange::new(-45.0, 45.0, 0.1);
const CYCLES: IntRange = IntRange { min: 1, max: 32 };
const AMOUNT: FloatRange = FloatRange::new(0.0, 1.0, 0.01);
const BREATHING_AMPLITUDE: FloatRange = FloatRange::new(0.0, 0.2, 0.001);
const FIGURE_EIGHT_AMPLITUDE: FloatRange = FloatRange::new(0.0, 0.5, 0.005);
const DEPTH_HEIGHT: FloatRange = FloatRange::new(0.0, 0.8, 0.01);
const DEPTH_PLANE: FloatRange = FloatRange::new(0.0, 1.0, 0.01);
const DEPTH_ZOOM: FloatRange = FloatRange::new(0.5, 2.0, 0.01);
const SHAKE_AMOUNT: FloatRange = FloatRange::new(0.0, 2.0, 0.01);
const SHAKE_SPEED: FloatRange = FloatRange::new(0.1, 4.0, 0.01);
const WARP_AMPLITUDE: FloatRange = FloatRange::new(0.0, 0.02, 0.0005);
const WAVELENGTH: FloatRange = FloatRange::new(0.02, 1.0, 0.005);
const SPATIAL_SCALE: FloatRange = FloatRange::new(0.25, 4.0, 0.05);
const SEED: IntRange = IntRange {
    min: 0,
    max: u32::MAX as i64,
};

/// Setting ids of one [`View`].
struct ViewIds {
    zoom: &'static str,
    x: &'static str,
    y: &'static str,
    roll: &'static str,
}

const STILL_VIEW: ViewIds = ViewIds {
    zoom: "camera.view.zoom",
    x: "camera.view.x",
    y: "camera.view.y",
    roll: "camera.view.roll_degrees",
};
const START_VIEW: ViewIds = ViewIds {
    zoom: "camera.start.zoom",
    x: "camera.start.x",
    y: "camera.start.y",
    roll: "camera.start.roll_degrees",
};
const END_VIEW: ViewIds = ViewIds {
    zoom: "camera.end.zoom",
    x: "camera.end.x",
    y: "camera.end.y",
    roll: "camera.end.roll_degrees",
};

// ---------------------------------------------------------------------------
// Validation

fn validate_view(view: &View, ids: &ViewIds) -> Result<(), ConfigError> {
    ZOOM.check(ids.zoom, view.zoom)?;
    POSITION.check(ids.x, view.x)?;
    POSITION.check(ids.y, view.y)?;
    ROLL.check(ids.roll, view.roll_degrees)
}

fn validate_camera(camera: &CameraMotion) -> Result<(), ConfigError> {
    match *camera {
        CameraMotion::Still { view } => validate_view(&view, &STILL_VIEW),
        CameraMotion::Transform { start, end, .. } => {
            validate_view(&start, &START_VIEW)?;
            validate_view(&end, &END_VIEW)
        }
        CameraMotion::Tilt { amount, .. } => AMOUNT.check("camera.amount", amount),
        CameraMotion::Breathing {
            base_zoom,
            amplitude,
            cycles,
        } => {
            ZOOM.check("camera.base_zoom", base_zoom)?;
            BREATHING_AMPLITUDE.check("camera.amplitude", amplitude)?;
            CYCLES.check("camera.cycles", cycles)
        }
        CameraMotion::FigureEight {
            zoom,
            amplitude,
            cycles,
        } => {
            ZOOM.check("camera.zoom", zoom)?;
            FIGURE_EIGHT_AMPLITUDE.check("camera.amplitude.0", amplitude[0])?;
            FIGURE_EIGHT_AMPLITUDE.check("camera.amplitude.1", amplitude[1])?;
            CYCLES.check("camera.cycles", cycles)
        }
        CameraMotion::Depth {
            height,
            steady,
            focus,
            zoom,
            amount,
            cycles,
            ..
        } => {
            DEPTH_HEIGHT.check("camera.height", height)?;
            DEPTH_PLANE.check("camera.steady", steady)?;
            DEPTH_PLANE.check("camera.focus", focus)?;
            DEPTH_ZOOM.check("camera.zoom", zoom)?;
            AMOUNT.check("camera.amount", amount)?;
            CYCLES.check("camera.cycles", cycles)
        }
    }
}

fn validate_shake(shake: &ShakeSettings) -> Result<(), ConfigError> {
    SHAKE_AMOUNT.check("shake.amount", shake.amount)?;
    SHAKE_SPEED.check("shake.speed", shake.speed)
}

fn validate_warp(warp: &WarpSettings) -> Result<(), ConfigError> {
    match *warp {
        WarpSettings::Ripple {
            amplitude,
            wavelength,
            center,
            cycles,
            ..
        } => {
            WARP_AMPLITUDE.check("warp.amplitude", amplitude)?;
            WAVELENGTH.check("warp.wavelength", wavelength)?;
            POSITION.check("warp.center.0", center[0])?;
            POSITION.check("warp.center.1", center[1])?;
            CYCLES.check("warp.cycles", cycles)
        }
        WarpSettings::HeatShimmer {
            amplitude,
            spatial_scale,
            cycles,
            ..
        } => {
            WARP_AMPLITUDE.check("warp.amplitude", amplitude)?;
            SPATIAL_SCALE.check("warp.spatial_scale", spatial_scale)?;
            CYCLES.check("warp.cycles", cycles)
        }
    }
}

fn nyquist_error(field: &str, hz: f64, fps: FrameRate, remedy: &str) -> ConfigError {
    ConfigError::invalid(
        field,
        format!(
            "highest temporal frequency {hz:.3} Hz is not below the Nyquist limit {:.3} Hz at {fps} fps; {remedy}",
            fps.nyquist_hz()
        ),
    )
}

impl MotionProject {
    /// Parses and validates a saved project. Unknown, missing, and mistyped
    /// fields are errors at every level, including disabled blocks.
    pub fn from_json(json: &str) -> Result<Self, ConfigError> {
        let mut deserializer = serde_json::Deserializer::from_str(json);
        let project: Self =
            serde_path_to_error::deserialize(&mut deserializer).map_err(|error| {
                ConfigError::Json {
                    field: error.path().to_string(),
                    message: error.inner().to_string(),
                }
            })?;
        deserializer.end().map_err(|error| ConfigError::Json {
            field: ".".to_owned(),
            message: error.to_string(),
        })?;
        project.validate()?;
        Ok(project)
    }

    /// Validates, then serializes as pretty JSON.
    pub fn to_json(&self) -> Result<String, ConfigError> {
        self.validate()?;
        serde_json::to_string_pretty(self).map_err(|error| ConfigError::Json {
            field: ".".to_owned(),
            message: error.to_string(),
        })
    }

    /// Checks version, output, timeline, every block's ranges (disabled
    /// blocks included), and that active periodic motion stays strictly below
    /// half the frame rate, and that displacement can be covered at this aspect.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.validated_geometry().map(|_| ())
    }

    /// Validates once and returns (output aspect, constant displacement safety scale).
    pub(crate) fn validated_geometry(&self) -> Result<(f64, f64), ConfigError> {
        if self.version != PRESET_VERSION {
            return Err(ConfigError::invalid(
                "version",
                format!(
                    "unsupported project version {}; expected {PRESET_VERSION}",
                    self.version
                ),
            ));
        }
        self.output.validate().map_err(|e| e.within("output"))?;
        self.timeline.validate().map_err(|e| e.within("timeline"))?;
        validate_camera(&self.camera)?;
        validate_shake(&self.shake)?;
        validate_warp(&self.warp)?;
        self.validate_temporal()?;
        let aspect = f64::from(self.output.width) / f64::from(self.output.height);
        Ok((aspect, crate::motion::project_safety_scale(self, aspect)?))
    }

    fn validate_temporal(&self) -> Result<(), ConfigError> {
        match self.camera {
            CameraMotion::Breathing {
                amplitude, cycles, ..
            } if amplitude > 0.0 => self.check_harmonic("camera.cycles", 1, cycles)?,
            CameraMotion::FigureEight {
                amplitude, cycles, ..
            } if amplitude[0] > 0.0 || amplitude[1] > 0.0 => {
                self.check_harmonic("camera.cycles", 2, cycles)?;
            }
            CameraMotion::Depth {
                motion,
                amount,
                cycles,
                ..
            } if amount > 0.0 && motion != DepthMotion::DollyOut => {
                self.check_harmonic("camera.cycles", 1, cycles)?;
            }
            _ => {}
        }

        if self.shake.enabled && self.shake.amount > 0.0 {
            let fps = self.timeline.fps;
            let hz = self.shake.profile.max_frequency_hz() * f64::from(self.shake.speed);
            if hz >= fps.nyquist_hz() {
                return Err(nyquist_error(
                    "shake.speed",
                    hz,
                    fps,
                    "lower shake.speed or increase fps",
                ));
            }
        }

        match self.warp {
            WarpSettings::Ripple {
                enabled: true,
                amplitude,
                cycles,
                ..
            } if amplitude > 0.0 => self.check_harmonic("warp.cycles", 1, cycles),
            WarpSettings::HeatShimmer {
                enabled: true,
                amplitude,
                cycles,
                ..
            } if amplitude > 0.0 => self.check_harmonic("warp.cycles", 2, cycles),
            _ => Ok(()),
        }
    }

    /// `harmonic * cycles / duration < fps / 2`, evaluated exactly as
    /// `2 * harmonic * cycles < frames`.
    fn check_harmonic(&self, field: &str, harmonic: u32, cycles: u32) -> Result<(), ConfigError> {
        let periods = 2 * u64::from(harmonic) * u64::from(cycles);
        if periods < u64::from(self.timeline.frames) {
            return Ok(());
        }
        let hz = f64::from(harmonic) * f64::from(cycles) / self.timeline.duration_seconds();
        Err(nyquist_error(
            field,
            hz,
            self.timeline.fps,
            &format!("lower {field}, lengthen the duration, or increase fps"),
        ))
    }

    /// The descriptor that currently governs `id`.
    pub fn active_descriptor(&self, id: &str) -> Result<&'static SettingDescriptor, ConfigError> {
        let mut known = false;
        for descriptor in setting_descriptors() {
            if descriptor.id == id {
                if descriptor.is_visible(self) {
                    return Ok(descriptor);
                }
                known = true;
            }
        }
        if !known {
            return Err(ConfigError::UnknownSetting { id: id.to_owned() });
        }
        let requires = setting_descriptors()
            .filter(|d| d.id == id)
            .map(|d| {
                d.visible_when
                    .iter()
                    .map(|v| format!("{}={}", v.id, v.value))
                    .collect::<Vec<_>>()
                    .join(" and ")
            })
            .collect::<Vec<_>>()
            .join(" or ");
        Err(ConfigError::InactiveSetting {
            id: id.to_owned(),
            requires,
        })
    }

    /// Current value of an active setting; choices are option indices.
    pub fn get_by_id(&self, id: &str) -> Result<SettingValue, ConfigError> {
        let descriptor = self.active_descriptor(id)?;
        if let Some(value) = condition_value(self, id) {
            return descriptor.parse(value);
        }
        if id == "seed" {
            return Ok(SettingValue::Int(i64::from(self.seed)));
        }
        let view_value = |view: &View, field: &str| match field {
            "zoom" => Some(view.zoom),
            "x" => Some(view.x),
            "y" => Some(view.y),
            "roll_degrees" => Some(view.roll_degrees),
            _ => None,
        };
        let value = if let Some(field) = id.strip_prefix("camera.") {
            match &self.camera {
                CameraMotion::Still { view } => field
                    .strip_prefix("view.")
                    .and_then(|field| view_value(view, field)),
                CameraMotion::Transform { start, end, .. } => {
                    if let Some(field) = field.strip_prefix("start.") {
                        view_value(start, field)
                    } else {
                        field
                            .strip_prefix("end.")
                            .and_then(|field| view_value(end, field))
                    }
                }
                CameraMotion::Tilt { amount, .. } => (field == "amount").then_some(*amount),
                CameraMotion::Breathing {
                    base_zoom,
                    amplitude,
                    cycles,
                } => match field {
                    "base_zoom" => Some(*base_zoom),
                    "amplitude" => Some(*amplitude),
                    "cycles" => return Ok(SettingValue::Int(i64::from(*cycles))),
                    _ => None,
                },
                CameraMotion::FigureEight {
                    zoom,
                    amplitude,
                    cycles,
                } => match field {
                    "zoom" => Some(*zoom),
                    "amplitude.0" => Some(amplitude[0]),
                    "amplitude.1" => Some(amplitude[1]),
                    "cycles" => return Ok(SettingValue::Int(i64::from(*cycles))),
                    _ => None,
                },
                CameraMotion::Depth {
                    height,
                    steady,
                    focus,
                    zoom,
                    amount,
                    cycles,
                    ..
                } => match field {
                    "height" => Some(*height),
                    "steady" => Some(*steady),
                    "focus" => Some(*focus),
                    "zoom" => Some(*zoom),
                    "amount" => Some(*amount),
                    "cycles" => return Ok(SettingValue::Int(i64::from(*cycles))),
                    _ => None,
                },
            }
        } else {
            match id {
                "shake.amount" => Some(self.shake.amount),
                "shake.speed" => Some(self.shake.speed),
                _ => match &self.warp {
                    WarpSettings::Ripple {
                        amplitude,
                        wavelength,
                        center,
                        cycles,
                        ..
                    } => match id {
                        "warp.amplitude" => Some(*amplitude),
                        "warp.wavelength" => Some(*wavelength),
                        "warp.center.0" => Some(center[0]),
                        "warp.center.1" => Some(center[1]),
                        "warp.cycles" => return Ok(SettingValue::Int(i64::from(*cycles))),
                        _ => None,
                    },
                    WarpSettings::HeatShimmer {
                        amplitude,
                        spatial_scale,
                        cycles,
                        ..
                    } => match id {
                        "warp.amplitude" => Some(*amplitude),
                        "warp.spatial_scale" => Some(*spatial_scale),
                        "warp.cycles" => return Ok(SettingValue::Int(i64::from(*cycles))),
                        _ => None,
                    },
                },
            }
        };
        value
            .map(SettingValue::Float)
            .ok_or_else(|| ConfigError::UnknownSetting { id: id.to_owned() })
    }

    /// Parses `raw` per the active descriptor (numbers, `true`/`false`, or a
    /// choice's stable value or index) and applies it atomically.
    pub fn set_by_id(&mut self, id: &str, raw: &str) -> Result<(), ConfigError> {
        self.set_many_by_id(std::iter::once((id, raw)))
    }

    /// Applies ordered edits as one transaction. Every value is parsed and
    /// range-checked immediately; cross-field/temporal limits use the final
    /// state. No edit is retained if the completed project is invalid.
    pub fn set_many_by_id<'a>(
        &mut self,
        updates: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<(), ConfigError> {
        let mut candidate = self.clone();
        for (id, raw) in updates {
            let descriptor = candidate.active_descriptor(id)?;
            let value = descriptor.parse(raw)?;
            descriptor.check(value)?;
            let field = field_mut(&mut candidate, id)
                .ok_or_else(|| ConfigError::UnknownSetting { id: id.to_owned() })?;
            if !field.set(value) {
                return Err(ConfigError::invalid(id, "value does not fit this setting"));
            }
        }
        self.commit(candidate)
    }

    /// Applies a typed value atomically: on any error, including temporal
    /// limits, the project is unchanged. Selecting a different `camera.type`
    /// or `warp.type` installs that variant's defaults (keeping
    /// `warp.enabled`); `camera.motion` keeps the shared depth parameters.
    pub fn set_value(&mut self, id: &str, value: SettingValue) -> Result<(), ConfigError> {
        self.active_descriptor(id)?.check(value)?;
        let mut candidate = self.clone();
        let field = field_mut(&mut candidate, id)
            .ok_or_else(|| ConfigError::UnknownSetting { id: id.to_owned() })?;
        if !field.set(value) {
            return Err(ConfigError::invalid(id, "value does not fit this setting"));
        }
        self.commit(candidate)
    }

    /// Replaces camera, shake, and warp with a built-in preset, preserving
    /// output, timeline, seed, and depth adjustments.
    pub fn apply_preset(&mut self, name: &str) -> Result<(), ConfigError> {
        let motion = preset_motion(name)?;
        let mut candidate = self.clone();
        candidate.camera = motion.camera;
        candidate.shake = motion.shake;
        candidate.warp = motion.warp;
        self.commit(candidate)
    }

    pub fn set_output(&mut self, output: Size) -> Result<(), ConfigError> {
        let mut candidate = self.clone();
        candidate.output = output;
        self.commit(candidate)
    }

    /// Replaces timing outright, e.g. with [`Timeline::from_duration`] when
    /// both frame rate and duration are given explicitly.
    pub fn set_timeline(&mut self, timeline: Timeline) -> Result<(), ConfigError> {
        let mut candidate = self.clone();
        candidate.timeline = timeline;
        self.commit(candidate)
    }

    /// Changes the frame rate, preserving the current duration.
    pub fn set_fps(&mut self, fps: FrameRate) -> Result<(), ConfigError> {
        let timeline = self
            .timeline
            .retimed(fps)
            .map_err(|e| e.within("timeline"))?;
        self.set_timeline(timeline)
    }

    /// Changes the duration, preserving the frame rate.
    pub fn set_duration(&mut self, seconds: f64) -> Result<(), ConfigError> {
        let timeline = Timeline::from_duration(self.timeline.fps, seconds)
            .map_err(|e| e.within("timeline"))?;
        self.set_timeline(timeline)
    }

    /// Changes the frame count, preserving the frame rate.
    pub fn set_frames(&mut self, frames: u32) -> Result<(), ConfigError> {
        self.set_timeline(Timeline {
            fps: self.timeline.fps,
            frames,
        })
    }

    fn commit(&mut self, candidate: Self) -> Result<(), ConfigError> {
        candidate.validate()?;
        *self = candidate;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Id-to-field mapping

enum FieldMut<'a> {
    Float(&'a mut f32),
    Int(&'a mut u32),
    Bool(&'a mut bool),
    Choice(ChoiceMut<'a>),
}

enum ChoiceMut<'a> {
    CameraType(&'a mut CameraMotion),
    WarpType(&'a mut WarpSettings),
    Motion(&'a mut DepthMotion),
    Easing(&'a mut Easing),
    Profile(&'a mut ShakeProfile),
}

impl FieldMut<'_> {
    fn get(&self) -> SettingValue {
        match self {
            Self::Float(v) => SettingValue::Float(**v),
            Self::Int(v) => SettingValue::Int(i64::from(**v)),
            Self::Bool(v) => SettingValue::Bool(**v),
            Self::Choice(choice) => SettingValue::Choice(choice.index()),
        }
    }

    /// Stores `value`; `false` if its type or domain does not fit.
    fn set(self, value: SettingValue) -> bool {
        match (self, value) {
            (Self::Float(slot), SettingValue::Float(v)) => *slot = v,
            (Self::Int(slot), SettingValue::Int(v)) => match u32::try_from(v) {
                Ok(v) => *slot = v,
                Err(_) => return false,
            },
            (Self::Bool(slot), SettingValue::Bool(v)) => *slot = v,
            (Self::Choice(choice), SettingValue::Choice(index)) => return choice.select(index),
            _ => return false,
        }
        true
    }
}

impl ChoiceMut<'_> {
    fn index(&self) -> u32 {
        match self {
            Self::CameraType(camera) => camera.kind().index(),
            Self::WarpType(warp) => warp.kind().index(),
            Self::Motion(motion) => motion.index(),
            Self::Easing(easing) => easing.index(),
            Self::Profile(profile) => profile.index(),
        }
    }

    fn index_of(&self, value: &str) -> Option<u32> {
        match self {
            Self::CameraType(_) => CameraKind::from_value(value).map(CameraKind::index),
            Self::WarpType(_) => WarpKind::from_value(value).map(WarpKind::index),
            Self::Motion(_) => DepthMotion::from_value(value).map(DepthMotion::index),
            Self::Easing(_) => Easing::from_value(value).map(Easing::index),
            Self::Profile(_) => ShakeProfile::from_value(value).map(ShakeProfile::index),
        }
    }

    fn select(self, index: u32) -> bool {
        match self {
            Self::CameraType(camera) => {
                let Some(kind) = CameraKind::from_index(index) else {
                    return false;
                };
                if camera.kind() != kind {
                    *camera = CameraMotion::default_for(kind);
                }
            }
            Self::WarpType(warp) => {
                let Some(kind) = WarpKind::from_index(index) else {
                    return false;
                };
                if warp.kind() != kind {
                    *warp = WarpSettings::default_for(kind, warp.enabled());
                }
            }
            Self::Motion(slot) => match DepthMotion::from_index(index) {
                Some(motion) => *slot = motion,
                None => return false,
            },
            Self::Easing(slot) => match Easing::from_index(index) {
                Some(easing) => *slot = easing,
                None => return false,
            },
            Self::Profile(slot) => match ShakeProfile::from_index(index) {
                Some(profile) => *slot = profile,
                None => return false,
            },
        }
        true
    }
}

/// Maps a setting id (the serialized dot path) to its field in the active
/// variant; `None` for unknown ids and fields of inactive variants.
fn field_mut<'a>(project: &'a mut MotionProject, id: &str) -> Option<FieldMut<'a>> {
    let field = match id {
        "seed" => FieldMut::Int(&mut project.seed),
        "depth.invert" => FieldMut::Bool(&mut project.depth.invert),
        "shake.enabled" => FieldMut::Bool(&mut project.shake.enabled),
        "shake.profile" => FieldMut::Choice(ChoiceMut::Profile(&mut project.shake.profile)),
        "shake.amount" => FieldMut::Float(&mut project.shake.amount),
        "shake.speed" => FieldMut::Float(&mut project.shake.speed),
        "camera.type" => FieldMut::Choice(ChoiceMut::CameraType(&mut project.camera)),
        "warp.type" => FieldMut::Choice(ChoiceMut::WarpType(&mut project.warp)),
        _ => {
            if let Some(rest) = id.strip_prefix("camera.") {
                return camera_field(&mut project.camera, rest);
            }
            return warp_field(&mut project.warp, id.strip_prefix("warp.")?);
        }
    };
    Some(field)
}

fn view_field<'a>(view: &'a mut View, field: &str) -> Option<FieldMut<'a>> {
    Some(FieldMut::Float(match field {
        "zoom" => &mut view.zoom,
        "x" => &mut view.x,
        "y" => &mut view.y,
        "roll_degrees" => &mut view.roll_degrees,
        _ => return None,
    }))
}

fn camera_field<'a>(camera: &'a mut CameraMotion, field: &str) -> Option<FieldMut<'a>> {
    let float = FieldMut::Float;
    let easing_field = |easing| Some(FieldMut::Choice(ChoiceMut::Easing(easing)));
    match camera {
        CameraMotion::Still { view } => view_field(view, field.strip_prefix("view.")?),
        CameraMotion::Transform { start, end, easing } => {
            if field == "easing" {
                easing_field(easing)
            } else if let Some(rest) = field.strip_prefix("start.") {
                view_field(start, rest)
            } else {
                view_field(end, field.strip_prefix("end.")?)
            }
        }
        CameraMotion::Tilt { amount, easing } => match field {
            "amount" => Some(float(amount)),
            "easing" => easing_field(easing),
            _ => None,
        },
        CameraMotion::Breathing {
            base_zoom,
            amplitude,
            cycles,
        } => match field {
            "base_zoom" => Some(float(base_zoom)),
            "amplitude" => Some(float(amplitude)),
            "cycles" => Some(FieldMut::Int(cycles)),
            _ => None,
        },
        CameraMotion::FigureEight {
            zoom,
            amplitude,
            cycles,
        } => {
            let [horizontal, vertical] = amplitude;
            match field {
                "zoom" => Some(float(zoom)),
                "amplitude.0" => Some(float(horizontal)),
                "amplitude.1" => Some(float(vertical)),
                "cycles" => Some(FieldMut::Int(cycles)),
                _ => None,
            }
        }
        CameraMotion::Depth {
            motion,
            height,
            steady,
            focus,
            zoom,
            amount,
            cycles,
        } => match field {
            "motion" => Some(FieldMut::Choice(ChoiceMut::Motion(motion))),
            "height" => Some(float(height)),
            "steady" => Some(float(steady)),
            "focus" => Some(float(focus)),
            "zoom" => Some(float(zoom)),
            "amount" => Some(float(amount)),
            "cycles" => Some(FieldMut::Int(cycles)),
            _ => None,
        },
    }
}

fn warp_field<'a>(warp: &'a mut WarpSettings, field: &str) -> Option<FieldMut<'a>> {
    match warp {
        WarpSettings::Ripple {
            enabled,
            amplitude,
            wavelength,
            center,
            cycles,
        } => {
            let [center_x, center_y] = center;
            Some(match field {
                "enabled" => FieldMut::Bool(enabled),
                "amplitude" => FieldMut::Float(amplitude),
                "wavelength" => FieldMut::Float(wavelength),
                "center.0" => FieldMut::Float(center_x),
                "center.1" => FieldMut::Float(center_y),
                "cycles" => FieldMut::Int(cycles),
                _ => return None,
            })
        }
        WarpSettings::HeatShimmer {
            enabled,
            amplitude,
            spatial_scale,
            cycles,
        } => Some(match field {
            "enabled" => FieldMut::Bool(enabled),
            "amplitude" => FieldMut::Float(amplitude),
            "spatial_scale" => FieldMut::Float(spatial_scale),
            "cycles" => FieldMut::Int(cycles),
            _ => return None,
        }),
    }
}

/// Stable string of a choice or boolean setting, for visibility conditions.
fn condition_value(project: &MotionProject, id: &str) -> Option<&'static str> {
    let flag = |on: bool| if on { "true" } else { "false" };
    match id {
        "camera.type" => Some(project.camera.kind().as_str()),
        "warp.type" => Some(project.warp.kind().as_str()),
        "camera.motion" => match project.camera {
            CameraMotion::Depth { motion, .. } => Some(motion.as_str()),
            _ => None,
        },
        "camera.easing" => match project.camera {
            CameraMotion::Transform { easing, .. } | CameraMotion::Tilt { easing, .. } => {
                Some(easing.as_str())
            }
            _ => None,
        },
        "shake.profile" => Some(project.shake.profile.as_str()),
        "shake.enabled" => Some(flag(project.shake.enabled)),
        "warp.enabled" => Some(flag(project.warp.enabled())),
        "depth.invert" => Some(flag(project.depth.invert)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Descriptors

/// Typed setting value; choices are option indices.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "kebab-case")]
pub enum SettingValue {
    Float(f32),
    Int(i64),
    Bool(bool),
    Choice(u32),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChoiceOption {
    /// Stable serialized value.
    pub value: String,
    pub label: String,
}

/// Value domain of a setting; inclusive bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum SettingKind {
    Float { min: f32, max: f32, step: f32 },
    Int { min: i64, max: i64 },
    Bool,
    Choice { options: Vec<ChoiceOption> },
}

/// Visibility condition: setting `id` currently has stable value `value`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Visibility {
    pub id: String,
    pub value: String,
}

/// One editable setting. Several descriptors may share an id when their
/// visibility is disjoint (e.g. `camera.zoom` for figure-eight and depth).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SettingDescriptor {
    /// Serialized dot path, e.g. `camera.start.zoom` or `warp.center.0`.
    pub id: String,
    pub label: String,
    pub unit: String,
    pub help: String,
    pub kind: SettingKind,
    /// Value in this setting's variant defaults.
    pub default: SettingValue,
    /// All conditions must hold for the setting to apply.
    pub visible_when: Vec<Visibility>,
}

impl SettingDescriptor {
    pub fn is_visible(&self, project: &MotionProject) -> bool {
        self.visible_when.iter().all(|condition| {
            condition_value(project, &condition.id) == Some(condition.value.as_str())
        })
    }

    /// Parses user text: a number, `true`/`false`, or a choice's stable value
    /// or index. Range checks happen in [`SettingDescriptor::check`].
    pub fn parse(&self, raw: &str) -> Result<SettingValue, ConfigError> {
        let raw = raw.trim();
        let invalid = |expected: String| {
            ConfigError::invalid(
                self.id.as_str(),
                format!("expected {expected}, got `{raw}`"),
            )
        };
        match &self.kind {
            SettingKind::Float { .. } => raw
                .parse()
                .map(SettingValue::Float)
                .map_err(|_| invalid("a number".to_owned())),
            SettingKind::Int { .. } => raw
                .parse()
                .map(SettingValue::Int)
                .map_err(|_| invalid("an integer".to_owned())),
            SettingKind::Bool => match raw {
                "true" => Ok(SettingValue::Bool(true)),
                "false" => Ok(SettingValue::Bool(false)),
                _ => Err(invalid("`true` or `false`".to_owned())),
            },
            SettingKind::Choice { options } => options
                .iter()
                .position(|option| option.value == raw)
                .or_else(|| raw.parse::<usize>().ok().filter(|i| *i < options.len()))
                .map(|index| SettingValue::Choice(index as u32))
                .ok_or_else(|| {
                    let values: Vec<&str> = options.iter().map(|o| o.value.as_str()).collect();
                    invalid(format!(
                        "one of {} or an index 0..={}",
                        values.join(", "),
                        options.len().saturating_sub(1)
                    ))
                }),
        }
    }

    /// Type, finiteness, and range of `value` for this setting.
    pub fn check(&self, value: SettingValue) -> Result<(), ConfigError> {
        match (&self.kind, value) {
            (SettingKind::Float { min, max, .. }, SettingValue::Float(v)) => {
                check_float(&self.id, v, *min, *max)
            }
            (SettingKind::Int { min, max }, SettingValue::Int(v)) => {
                check_int(&self.id, v, *min, *max)
            }
            (SettingKind::Bool, SettingValue::Bool(_)) => Ok(()),
            (SettingKind::Choice { options }, SettingValue::Choice(index)) => {
                if (index as usize) < options.len() {
                    Ok(())
                } else {
                    Err(ConfigError::invalid(
                        self.id.as_str(),
                        format!(
                            "choice index {index} is out of range 0..={}",
                            options.len().saturating_sub(1)
                        ),
                    ))
                }
            }
            (kind, _) => {
                let expected = match kind {
                    SettingKind::Float { .. } => "a float",
                    SettingKind::Int { .. } => "an integer",
                    SettingKind::Bool => "a boolean",
                    SettingKind::Choice { .. } => "a choice index",
                };
                Err(ConfigError::invalid(
                    self.id.as_str(),
                    format!("expected {expected} value"),
                ))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SettingGroup {
    pub id: String,
    pub label: String,
    pub help: String,
    pub settings: Vec<SettingDescriptor>,
}

/// Everything a front end needs to build its controls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SettingsSchema {
    pub version: u32,
    pub defaults: MotionProject,
    pub groups: Vec<SettingGroup>,
}

/// Owned schema for serialization (e.g. to WASM front ends).
pub fn settings_schema() -> SettingsSchema {
    SettingsSchema {
        version: PRESET_VERSION,
        defaults: MotionProject::default(),
        groups: REGISTRY.clone(),
    }
}

/// The registry, built once.
pub fn setting_groups() -> &'static [SettingGroup] {
    &REGISTRY
}

pub fn setting_descriptors() -> impl Iterator<Item = &'static SettingDescriptor> {
    REGISTRY.iter().flat_map(|group| group.settings.iter())
}

static REGISTRY: LazyLock<Vec<SettingGroup>> = LazyLock::new(build_registry);

type When = &'static [(&'static str, &'static str)];

const ALWAYS: When = &[];
const STILL: When = &[("camera.type", "still")];
const TRANSFORM: When = &[("camera.type", "transform")];
const TILT: When = &[("camera.type", "tilt")];
const BREATHING: When = &[("camera.type", "breathing")];
const FIGURE_EIGHT: When = &[("camera.type", "figure-eight")];
const DEPTH: When = &[("camera.type", "depth")];
const RIPPLE: When = &[("warp.type", "ripple")];
const HEAT_SHIMMER: When = &[("warp.type", "heat-shimmer")];

/// Value of `id` in the default project after installing the variants named
/// by `when`. Panics on registry/field mismatches, which are static bugs.
fn default_value(id: &str, when: When) -> SettingValue {
    let mut project = MotionProject::default();
    for (condition, value) in when {
        let Some(FieldMut::Choice(choice)) = field_mut(&mut project, condition) else {
            panic!("visibility condition `{condition}` of `{id}` is not a choice setting");
        };
        let index = choice
            .index_of(value)
            .unwrap_or_else(|| panic!("`{value}` is not an option of `{condition}`"));
        assert!(choice.select(index), "cannot select `{condition}={value}`");
    }
    field_mut(&mut project, id)
        .unwrap_or_else(|| panic!("setting `{id}` has no field in its variant"))
        .get()
}

fn setting(
    id: &str,
    label: &str,
    unit: &str,
    help: &str,
    kind: SettingKind,
    when: When,
) -> SettingDescriptor {
    SettingDescriptor {
        id: id.to_owned(),
        label: label.to_owned(),
        unit: unit.to_owned(),
        help: help.to_owned(),
        kind,
        default: default_value(id, when),
        visible_when: when
            .iter()
            .map(|(id, value)| Visibility {
                id: (*id).to_owned(),
                value: (*value).to_owned(),
            })
            .collect(),
    }
}

fn view_settings(ids: &ViewIds, labels: [&str; 4], when: When) -> [SettingDescriptor; 4] {
    let [zoom, x, y, roll] = labels;
    [
        setting(
            ids.zoom,
            zoom,
            "×",
            "Magnification of the cover-cropped frame; 1 shows the whole crop.",
            ZOOM.kind(),
            when,
        ),
        setting(
            ids.x,
            x,
            "frame",
            "Fraction of available horizontal crop travel: 0 left, 1 right.",
            POSITION.kind(),
            when,
        ),
        setting(
            ids.y,
            y,
            "frame",
            "Fraction of available vertical crop travel: 0 top, 1 bottom.",
            POSITION.kind(),
            when,
        ),
        setting(
            ids.roll,
            roll,
            "°",
            "Rotation of the view.",
            ROLL.kind(),
            when,
        ),
    ]
}

fn cycles_setting(id: &str, help: &str, when: When) -> SettingDescriptor {
    setting(id, "Cycles", "cycles", help, CYCLES.kind(), when)
}

fn build_registry() -> Vec<SettingGroup> {
    let mut camera = vec![setting(
        "camera.type",
        "Motion",
        "",
        "Camera motion family. Changing it installs that family's default parameters.",
        SettingKind::Choice {
            options: CameraKind::choice_options(),
        },
        ALWAYS,
    )];
    camera.extend(view_settings(
        &STILL_VIEW,
        ["Zoom", "X", "Y", "Roll"],
        STILL,
    ));
    camera.extend(view_settings(
        &START_VIEW,
        ["Start zoom", "Start X", "Start Y", "Start roll"],
        TRANSFORM,
    ));
    camera.extend(view_settings(
        &END_VIEW,
        ["End zoom", "End X", "End Y", "End roll"],
        TRANSFORM,
    ));
    let easing = |when| {
        setting(
            "camera.easing",
            "Easing",
            "",
            "Interpolation curve across the timeline.",
            SettingKind::Choice {
                options: Easing::choice_options(),
            },
            when,
        )
    };
    camera.extend([
        easing(TRANSFORM),
        setting(
            "camera.amount",
            "Amount",
            "",
            "Strength of the tilt sweep; 0 holds the frame still.",
            AMOUNT.kind(),
            TILT,
        ),
        easing(TILT),
        setting(
            "camera.base_zoom",
            "Base zoom",
            "×",
            "Minimum zoom; breathing adds up to the zoom swing.",
            ZOOM.kind(),
            BREATHING,
        ),
        setting(
            "camera.amplitude",
            "Zoom swing",
            "×",
            "Peak zoom deviation from the base zoom.",
            BREATHING_AMPLITUDE.kind(),
            BREATHING,
        ),
        cycles_setting(
            "camera.cycles",
            "Full breaths over the timeline.",
            BREATHING,
        ),
        setting(
            "camera.zoom",
            "Zoom",
            "×",
            "Magnification while tracing the figure eight.",
            ZOOM.kind(),
            FIGURE_EIGHT,
        ),
        setting(
            "camera.amplitude.0",
            "Horizontal swing",
            "frame",
            "Horizontal extent of the figure eight.",
            FIGURE_EIGHT_AMPLITUDE.kind(),
            FIGURE_EIGHT,
        ),
        setting(
            "camera.amplitude.1",
            "Vertical swing",
            "frame",
            "Vertical extent of the figure eight.",
            FIGURE_EIGHT_AMPLITUDE.kind(),
            FIGURE_EIGHT,
        ),
        cycles_setting(
            "camera.cycles",
            "Complete figure-eight loops over the timeline.",
            FIGURE_EIGHT,
        ),
        setting(
            "camera.motion",
            "Depth motion",
            "",
            "Parallax camera path. Changing it keeps the parameters below.",
            SettingKind::Choice {
                options: DepthMotion::choice_options(),
            },
            DEPTH,
        ),
        setting(
            "camera.height",
            "Height",
            "frame",
            "Parallax displacement of the nearest depth.",
            DEPTH_HEIGHT.kind(),
            DEPTH,
        ),
        setting(
            "camera.steady",
            "Steady depth",
            "depth",
            "Depth plane that stays fixed while the camera moves.",
            DEPTH_PLANE.kind(),
            DEPTH,
        ),
        setting(
            "camera.focus",
            "Focus depth",
            "depth",
            "Depth plane the camera aims at.",
            DEPTH_PLANE.kind(),
            DEPTH,
        ),
        setting(
            "camera.zoom",
            "Zoom",
            "×",
            "Depth projection scale; values below 1 magnify the covered image.",
            DEPTH_ZOOM.kind(),
            DEPTH,
        ),
        setting(
            "camera.amount",
            "Amount",
            "",
            "Overall motion strength; 0 freezes the camera.",
            AMOUNT.kind(),
            DEPTH,
        ),
        cycles_setting(
            "camera.cycles",
            "Motion repetitions over the timeline; dolly out ignores it.",
            DEPTH,
        ),
    ]);

    let depth = vec![setting(
        "depth.invert",
        "Invert depth",
        "",
        "Treat dark depth values as near instead of white.",
        SettingKind::Bool,
        ALWAYS,
    )];

    let shake = vec![
        setting(
            "shake.enabled",
            "Enable shake",
            "",
            "Layer procedural shake over the camera path.",
            SettingKind::Bool,
            ALWAYS,
        ),
        setting(
            "shake.profile",
            "Profile",
            "",
            "Shake character; vibrate profiles use higher frequencies.",
            SettingKind::Choice {
                options: ShakeProfile::choice_options(),
            },
            ALWAYS,
        ),
        setting(
            "shake.amount",
            "Amount",
            "×",
            "Shake strength multiplier.",
            SHAKE_AMOUNT.kind(),
            ALWAYS,
        ),
        setting(
            "shake.speed",
            "Speed",
            "×",
            "Frequency multiplier; the fastest component must stay below half the frame rate.",
            SHAKE_SPEED.kind(),
            ALWAYS,
        ),
    ];

    let warp = vec![
        setting(
            "warp.enabled",
            "Enable warp",
            "",
            "Layer an image-space warp over the camera path.",
            SettingKind::Bool,
            ALWAYS,
        ),
        setting(
            "warp.type",
            "Warp",
            "",
            "Warp family. Changing it installs that family's defaults and keeps it enabled or disabled.",
            SettingKind::Choice {
                options: WarpKind::choice_options(),
            },
            ALWAYS,
        ),
        setting(
            "warp.amplitude",
            "Amplitude",
            "frame",
            "Peak ripple displacement.",
            WARP_AMPLITUDE.kind(),
            RIPPLE,
        ),
        setting(
            "warp.wavelength",
            "Wavelength",
            "frame",
            "Distance between ripple crests.",
            WAVELENGTH.kind(),
            RIPPLE,
        ),
        setting(
            "warp.center.0",
            "Center X",
            "frame",
            "Horizontal ripple origin: 0 is the left edge, 1 the right edge.",
            POSITION.kind(),
            RIPPLE,
        ),
        setting(
            "warp.center.1",
            "Center Y",
            "frame",
            "Vertical ripple origin: 0 is the top edge, 1 the bottom edge.",
            POSITION.kind(),
            RIPPLE,
        ),
        cycles_setting("warp.cycles", "Wave periods over the timeline.", RIPPLE),
        setting(
            "warp.amplitude",
            "Amplitude",
            "frame",
            "Peak shimmer displacement.",
            WARP_AMPLITUDE.kind(),
            HEAT_SHIMMER,
        ),
        setting(
            "warp.spatial_scale",
            "Spatial scale",
            "×",
            "Spatial frequency of shimmer features; larger values give finer waves.",
            SPATIAL_SCALE.kind(),
            HEAT_SHIMMER,
        ),
        cycles_setting(
            "warp.cycles",
            "Shimmer loops over the timeline.",
            HEAT_SHIMMER,
        ),
    ];

    let project = vec![setting(
        "seed",
        "Seed",
        "",
        "Seed for deterministic procedural noise.",
        SEED.kind(),
        ALWAYS,
    )];

    let group = |id: &str, label: &str, help: &str, settings| SettingGroup {
        id: id.to_owned(),
        label: label.to_owned(),
        help: help.to_owned(),
        settings,
    };
    vec![
        group("camera", "Camera", "Camera path over the timeline.", camera),
        group("depth", "Depth", "Depth map interpretation.", depth),
        group("shake", "Shake", "Procedural camera shake.", shake),
        group("warp", "Warp", "Image-space warp effects.", warp),
        group("project", "Project", "Project-wide settings.", project),
    ]
}
