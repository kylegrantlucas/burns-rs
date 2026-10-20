//! Command-line syntax.

use std::path::PathBuf;

use burns::FrameRate;
use burns_media::{DEFAULT_CRF, MAX_CRF};
use clap::{Args, Parser, Subcommand};

use crate::project::{Assignment, Overrides, parse_fps};

/// Deterministic single-image motion: render clips, frames, and depth maps headlessly.
#[derive(Debug, Parser)]
#[command(name = "burns", version, about, propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// List the built-in presets and whether they need depth or loop.
    Presets {
        /// Print a JSON array of preset metadata.
        #[arg(long)]
        json: bool,
    },
    /// Print the settings schema (ids, ranges, defaults, choices, visibility) as JSON.
    Schema,
    /// Print a built-in preset as a complete project JSON document.
    Preset {
        /// Preset name, as listed by `burns presets`.
        name: String,
    },
    /// Render an H.264 MP4 clip with FFmpeg.
    Render(RenderArgs),
    /// Render one final-quality frame as an RGBA PNG.
    Frame(FrameArgs),
    /// Generate (or reuse the cached) depth map for an image and save it as a 16-bit PNG.
    ///
    /// The first run downloads the pinned Depth Anything V2 Small model (about 99 MB) and needs
    /// ONNX Runtime 1.27+ (ORT_DYLIB_PATH). The raw map is saved; a project's depth inversion is
    /// applied at render time, not baked into the file.
    Depth(DepthArgs),
}

#[derive(Debug, Args)]
pub struct RenderArgs {
    /// Source image (PNG, JPEG, or WebP).
    #[arg(short = 'i', long = "input", value_name = "IMAGE")]
    pub input: PathBuf,
    /// Output MP4 path.
    #[arg(short = 'o', long = "output", value_name = "CLIP.mp4")]
    pub output: PathBuf,
    #[command(flatten)]
    pub project: ProjectArgs,
    #[command(flatten)]
    pub depth: DepthSource,
    /// H.264 constant rate factor (lower is higher quality).
    #[arg(long, value_name = "0..51", default_value_t = DEFAULT_CRF,
          value_parser = clap::value_parser!(u8).range(0..=i64::from(MAX_CRF)))]
    pub crf: u8,
    /// Replace an existing output file once the export succeeds.
    #[arg(long)]
    pub overwrite: bool,
}

#[derive(Debug, Args)]
pub struct FrameArgs {
    /// Source image (PNG, JPEG, or WebP).
    #[arg(short = 'i', long = "input", value_name = "IMAGE")]
    pub input: PathBuf,
    /// Output PNG path.
    #[arg(short = 'o', long = "output", value_name = "FRAME.png")]
    pub output: PathBuf,
    /// Zero-based frame index within the timeline.
    #[arg(long, value_name = "N")]
    pub frame: u32,
    #[command(flatten)]
    pub project: ProjectArgs,
    #[command(flatten)]
    pub depth: DepthSource,
    /// Replace an existing output file.
    #[arg(long)]
    pub overwrite: bool,
}

#[derive(Debug, Args)]
pub struct DepthArgs {
    /// Source image (PNG, JPEG, or WebP).
    #[arg(short = 'i', long = "input", value_name = "IMAGE")]
    pub input: PathBuf,
    /// Output 16-bit grayscale PNG path (white is near).
    #[arg(short = 'o', long = "output", value_name = "DEPTH.png")]
    pub output: PathBuf,
    /// Replace an existing output file.
    #[arg(long)]
    pub overwrite: bool,
}

/// Which project to start from and how to edit it.
#[derive(Debug, Args)]
pub struct ProjectArgs {
    /// Built-in preset to start from [default: kenburns].
    #[arg(long, value_name = "NAME", conflicts_with = "config")]
    pub preset: Option<String>,
    /// Complete project JSON file to start from (strict: no unknown or missing fields).
    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,
    /// Clip duration; converted to frames at the final frame rate.
    #[arg(long, value_name = "SECONDS", allow_negative_numbers = true)]
    pub duration: Option<f64>,
    /// Frame rate; without --duration the project's duration is kept.
    #[arg(long, value_name = "NUM[/DEN]", value_parser = parse_fps)]
    pub fps: Option<FrameRate>,
    /// Output width in pixels.
    #[arg(long, value_name = "W", value_parser = clap::value_parser!(u32).range(1..))]
    pub width: Option<u32>,
    /// Output height in pixels.
    #[arg(long, value_name = "H", value_parser = clap::value_parser!(u32).range(1..))]
    pub height: Option<u32>,
    /// Set a motion setting or seed by schema id; repeatable, applied in order.
    #[arg(long = "set", value_name = "ID=VALUE")]
    pub set: Vec<Assignment>,
    /// Random seed for shake and warp phases; overrides `--set seed=...`.
    #[arg(long, value_name = "U32")]
    pub seed: Option<u32>,
}

impl ProjectArgs {
    pub fn overrides(&self) -> Overrides {
        Overrides {
            set: self.set.clone(),
            width: self.width,
            height: self.height,
            fps: self.fps,
            duration: self.duration,
            seed: self.seed,
        }
    }
}

/// Where a depth camera gets its map. Without either flag, a previously generated map cached
/// for this image is used; other cameras ignore depth entirely.
#[derive(Debug, Args)]
pub struct DepthSource {
    /// 8- or 16-bit grayscale depth PNG (white is near), any resolution, covering the full image.
    #[arg(long, value_name = "MAP.png", conflicts_with = "auto_depth")]
    pub depth: Option<PathBuf>,
    /// Generate the depth map if it is not cached (depth motion only; may download the ~99 MB
    /// model and needs ONNX Runtime).
    #[arg(long)]
    pub auto_depth: bool,
}
