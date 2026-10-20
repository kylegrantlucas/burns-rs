//! Deterministic motion for one still image.
//!
//! Projects contain only portable settings. Asset decoding, timeline evaluation, and
//! descriptors work without a GPU; the default `gpu` feature adds shared native/WebGPU rendering.

pub mod image;
pub mod motion;
pub mod png_output;
pub mod presets;
#[cfg(feature = "gpu")]
pub mod renderer;
pub mod settings;
pub mod timeline;

pub use image::{CoverMap, DepthMap, ImageError, SourceImage};
pub use motion::{CameraPose, EvaluatedMotion, MotionEvaluator, ShakePose, WarpField};
pub use png_output::{PngError, write_rgba8_png};
pub use presets::{PRESET_VERSION, PresetDescriptor, preset, presets};
#[cfg(feature = "gpu")]
pub use renderer::{GpuContext, RenderError, RenderQuality, RenderView, Renderer};
pub use settings::{
    CameraKind, CameraMotion, ChoiceOption, ConfigError, DepthAdjustments, DepthMotion, Easing,
    MotionProject, SettingDescriptor, SettingGroup, SettingKind, SettingValue, SettingsSchema,
    ShakeProfile, ShakeSettings, Size, View, Visibility, WarpKind, WarpSettings,
    setting_descriptors, setting_groups, settings_schema,
};
pub use timeline::{FrameRate, FrameTime, Timeline};
