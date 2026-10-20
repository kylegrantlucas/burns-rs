use std::io;
use std::path::{Path, PathBuf};

use burns::{ConfigError, ImageError, PngError, RenderError};

/// Errors from loading assets, preflighting, rendering, encoding, and publishing output files.
#[derive(Debug, thiserror::Error)]
pub enum MediaError {
    /// The project or a requested frame is invalid.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// An in-memory image or depth map could not be built or encoded.
    #[error(transparent)]
    Image(#[from] ImageError),
    /// GPU setup, upload, rendering, or readback failed.
    #[error(transparent)]
    Render(#[from] RenderError),
    /// A file was read but its contents could not be decoded.
    #[error("`{}`: {source}", path.display())]
    Asset {
        path: PathBuf,
        #[source]
        source: ImageError,
    },
    /// A filesystem operation failed.
    #[error("could not {action} `{}`: {source}", path.display())]
    Io {
        action: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    /// The output path cannot name a file.
    #[error("output path `{}` is invalid: {reason}", path.display())]
    InvalidOutputPath { path: PathBuf, reason: &'static str },
    /// The directory that would contain the output is missing or not a directory.
    #[error(
        "cannot write `{}`: the output directory `{}` {reason}",
        path.display(),
        directory.display()
    )]
    OutputDirectory {
        path: PathBuf,
        directory: PathBuf,
        reason: &'static str,
    },
    /// The output entry (a file, directory, or symlink, even a dangling one) already exists and
    /// overwriting was not requested, or it appeared while the output was being written.
    #[error(
        "`{}` already exists; choose another output path or enable overwrite",
        path.display()
    )]
    OutputExists { path: PathBuf },
    /// Overwriting was requested but the output path is a directory.
    #[error("`{}` is a directory; choose a file path for the output", path.display())]
    OutputIsDirectory { path: PathBuf },
    /// The x264 constant rate factor is out of range.
    #[error(
        "CRF {crf} is out of range: H.264 CRF must be within 0..=51 (lower is higher quality; \
         the default is 20)"
    )]
    InvalidCrf { crf: u8 },
    /// H.264 4:2:0 output needs even dimensions; frames are never padded or cropped.
    #[error(
        "MP4 export needs an even width and height, but the project output is {width}x{height}; \
         set the output to an even size such as {}x{} (PNG frames accept odd sizes)",
        even(*width),
        even(*height)
    )]
    OddDimensions { width: u32, height: u32 },
    /// The project uses the depth camera but no depth map was supplied.
    #[error(
        "the depth camera needs a depth map: load or estimate one for this image, or choose a \
         preset that does not require depth"
    )]
    MissingDepth,
    /// A single-frame or depth-map export was cancelled before publication.
    #[error("media export cancelled before output publication")]
    Cancelled,
    /// FFmpeg is missing, cannot run, or lacks the libx264 encoder.
    #[error("the FFmpeg encoder `{}` is unavailable: {reason}", program.display())]
    EncoderUnavailable { program: PathBuf, reason: String },
    /// FFmpeg stopped accepting frames or exited unsuccessfully.
    #[error("FFmpeg failed: {reason}\nffmpeg stderr:\n{}", stderr_text(stderr))]
    EncoderFailed { reason: String, stderr: String },
    /// A PNG frame could not be encoded.
    #[error("failed to encode PNG frame: {0}")]
    FrameEncode(#[source] PngError),
}

impl MediaError {
    pub(crate) fn io(action: &'static str, path: &Path, source: io::Error) -> Self {
        Self::Io {
            action,
            path: path.to_owned(),
            source,
        }
    }
}

/// Nearest even size at or below `edge`, but at least 2.
fn even(edge: u32) -> u32 {
    (edge & !1).max(2)
}

fn stderr_text(stderr: &str) -> &str {
    if stderr.trim().is_empty() {
        "(no output)"
    } else {
        stderr.trim_end()
    }
}
