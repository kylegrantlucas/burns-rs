//! Native media I/O shared by the burns CLI and GUI.
//!
//! * [`load_image`] and [`load_depth`] read a file once and decode it with the core decoders.
//! * [`safe_write`] publishes any file through an owned temporary in the destination directory:
//!   existing entries (including dangling symlinks) are refused unless `overwrite` is set, a
//!   destination created during the write is never clobbered, and failures leave no partial file.
//! * [`save_frame`] renders one final-quality motion frame and streams it to PNG.
//! * [`RenderJob`] exports a project to H.264 MP4 through an `ffmpeg` child process on its own
//!   worker thread, with progress, cancellation, and the same publishing guarantees.
//!
//! Encoder settings live in [`RenderRequest`], never in the portable project.

mod assets;
mod encoder;
mod error;
mod frame;
mod job;
mod output;
mod prepare;

pub use assets::{load_depth, load_image, write_depth_png};
pub use error::MediaError;
pub use frame::{FrameRequest, save_frame};
pub use job::{DEFAULT_CRF, MAX_CRF, RenderJob, RenderRequest, RenderStatus};
pub use output::safe_write;
