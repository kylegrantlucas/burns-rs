//! One message-carrying error for every binding; JavaScript receives it as an `Error`.

use std::fmt;

use burns::{ConfigError, ImageError, RenderError};
use burns_depth::DepthError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineError(String);

impl EngineError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EngineError {}

impl From<RenderError> for EngineError {
    fn from(error: RenderError) -> Self {
        match error {
            // The core remedy names native types; the browser recovers through the engine.
            RenderError::DeviceLost { reason } => Self(format!(
                "the WebGPU device was lost ({reason}); await initCanvas(canvas) again to \
                 recreate it (the project and loaded assets are kept)"
            )),
            other => Self(other.to_string()),
        }
    }
}

impl From<ConfigError> for EngineError {
    fn from(error: ConfigError) -> Self {
        Self(error.to_string())
    }
}

impl From<ImageError> for EngineError {
    fn from(error: ImageError) -> Self {
        Self(error.to_string())
    }
}

impl From<DepthError> for EngineError {
    fn from(error: DepthError) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for EngineError {
    fn from(error: serde_json::Error) -> Self {
        Self(format!("JSON serialization failed: {error}"))
    }
}

pub type Result<T, E = EngineError> = std::result::Result<T, E>;
