use std::io;
use std::path::PathBuf;

use burns::{ConfigError, RenderError};
use burns_depth::DepthError;
use burns_media::MediaError;

/// Every way a command can fail. The message is printed to stderr as `error: ...`.
#[derive(Debug, thiserror::Error)]
pub enum CliError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error("project file `{}`: {source}", path.display())]
    ProjectFile {
        path: PathBuf,
        #[source]
        source: ConfigError,
    },
    #[error("could not read project file `{}`: {source}", path.display())]
    ReadProject {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(transparent)]
    Media(#[from] MediaError),
    #[error(transparent)]
    Depth(#[from] DepthError),
    #[error(transparent)]
    Render(#[from] RenderError),
    #[error(
        "the depth camera needs a depth map, and none is cached for `{}`; pass --depth MAP.png \
         to use an existing 8/16-bit grayscale map, or --auto-depth to generate one (the first \
         generation downloads the ~99 MB Depth Anything V2 Small model and needs ONNX Runtime)",
        image.display()
    )]
    DepthUnavailable { image: PathBuf },
    #[error("export failed: {0}")]
    Export(String),
    #[error("could not install the Ctrl-C handler: {0}")]
    Signal(String),
    #[error("could not write to stdout: {0}")]
    Stdout(#[source] io::Error),
    #[error("interrupted by Ctrl-C; no output was written")]
    Interrupted,
}

impl CliError {
    /// Process exit status: 130 after Ctrl-C, 1 for every other failure.
    pub fn exit_code(&self) -> u8 {
        match self {
            Self::Interrupted
            | Self::Depth(DepthError::Cancelled)
            | Self::Media(MediaError::Cancelled) => 130,
            _ => 1,
        }
    }
}
