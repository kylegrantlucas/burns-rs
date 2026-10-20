use std::{io, path::PathBuf};

use burns::ImageError;

/// Errors from depth preparation, inference, download, and caching.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DepthError {
    /// A depth map or source image could not be constructed, decoded, or encoded.
    #[error(transparent)]
    Image(#[from] ImageError),
    /// The source cannot be resized to a model input.
    #[error(
        "source image {width}x{height} cannot be prepared for depth estimation: {reason}; \
         crop or resize it, or supply a depth map"
    )]
    InputSize {
        width: u32,
        height: u32,
        reason: &'static str,
    },
    /// Raw model output dimensions are zero or overflow.
    #[error("depth model output size {width}x{height} is invalid: {reason}")]
    OutputSize {
        width: u32,
        height: u32,
        reason: &'static str,
    },
    /// Raw model output does not hold one sample per pixel.
    #[error(
        "depth model output for {width}x{height} must contain exactly {expected} samples, got {actual}"
    )]
    OutputLength {
        width: u32,
        height: u32,
        expected: usize,
        actual: usize,
    },
    /// Raw model output contains NaN or infinity.
    #[error(
        "depth model output at index {index} (x {x}, y {y}) is {value}; expected a finite value"
    )]
    NonFinite {
        index: usize,
        x: u32,
        y: u32,
        value: f32,
    },
    /// No ONNX Runtime shared library was found. Only depth generation needs it.
    #[error(
        "ONNX Runtime 1.27 or newer is required to generate depth, but it was not found ({tried}); \
         set ORT_DYLIB_PATH to its shared library (for example run inside `nix develop`), or \
         supply a depth map instead. 2D motion and imported depth maps do not need it"
    )]
    RuntimeMissing { tried: String },
    /// ONNX Runtime was found but could not be loaded or initialized.
    #[error(
        "failed to initialize ONNX Runtime from `{path}`: {message}; it must be ONNX Runtime \
         1.27 or newer (set ORT_DYLIB_PATH)"
    )]
    RuntimeInit { path: PathBuf, message: String },
    /// A different ONNX Runtime library is already loaded in this process.
    #[error(
        "ONNX Runtime is already loaded from `{loaded}` in this process and cannot be switched \
         to `{requested}`; restart to use a different runtime"
    )]
    RuntimeMismatch { loaded: PathBuf, requested: PathBuf },
    /// The model file could not be loaded into an inference session.
    #[error("failed to load depth model `{path}`: {message}")]
    ModelLoad { path: PathBuf, message: String },
    /// Session metadata or a run result violates the pinned tensor contract.
    #[error(
        "depth model does not match the pinned {} contract: {message}",
        crate::MODEL_NAME
    )]
    ModelContract { message: String },
    /// Running the model failed.
    #[error("depth inference failed: {message}")]
    Inference { message: String },
    /// The HTTP request or transfer failed.
    #[error("failed to download the depth model from {url}: {message}")]
    Download { url: String, message: String },
    /// A downloaded or existing model file has the wrong length.
    #[error("depth model from {origin} is {actual} bytes, expected exactly {expected}")]
    ModelLength {
        origin: String,
        expected: u64,
        actual: u64,
    },
    /// A download delivered more bytes than the pinned model has.
    #[error("depth model from {origin} is longer than the expected {expected} bytes")]
    ModelTooLong { origin: String, expected: u64 },
    /// A downloaded or existing model file has the wrong SHA-256.
    #[error(
        "depth model from {origin} has SHA-256 {actual}, expected {expected}; the file is corrupt \
         or the server returned a different file"
    )]
    ModelHash {
        origin: String,
        expected: String,
        actual: String,
    },
    /// The operation was cancelled; nothing was published.
    #[error("depth generation was cancelled")]
    Cancelled,
    /// A filesystem operation failed.
    #[error("failed to {action} `{path}`: {source}")]
    Io {
        action: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    /// The platform has no user cache directory.
    #[error("cannot determine the platform cache directory (no home directory is set)")]
    NoCacheDir,
    /// A cached depth map exists but cannot be used.
    #[error(
        "cached depth map `{path}` is unusable: {reason}; delete it or generate depth again to \
         replace it"
    )]
    CacheCorrupt { path: PathBuf, reason: String },
}

impl DepthError {
    #[cfg_attr(
        not(all(feature = "native", not(target_arch = "wasm32"))),
        allow(dead_code)
    )]
    pub(crate) fn io(action: &'static str, path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            action,
            path: path.into(),
            source,
        }
    }
}
