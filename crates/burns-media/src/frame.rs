//! Single final-quality PNG frames.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use burns::{
    DepthMap, GpuContext, MotionProject, PngError, RenderView, Size, SourceImage, write_rgba8_png,
};

use crate::MediaError;
use crate::output::Destination;
use crate::prepare::{check_inputs, renderer};

/// Immutable inputs and optional cancellation for a single motion PNG.
#[derive(Debug, Clone, Copy)]
pub struct FrameRequest<'a> {
    pub project: &'a MotionProject,
    pub image: &'a SourceImage,
    pub depth: Option<&'a DepthMap>,
    pub frame: u32,
    pub output: &'a Path,
    pub overwrite: bool,
    pub cancel: Option<&'a AtomicBool>,
}

pub(crate) fn check_cancel(cancel: Option<&AtomicBool>) -> Result<(), MediaError> {
    if cancel.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        Err(MediaError::Cancelled)
    } else {
        Ok(())
    }
}

/// Saves one final-quality motion PNG.
///
/// Always renders [`RenderView::Motion`] at exactly `project.output`; odd sizes are fine. The
/// rows are streamed from the mapped GPU readback into the encoder without a full-frame copy,
/// and the file is published like [`crate::safe_write`]. Cancellation before
/// publication removes only the owned temporary; publication wins a late signal.
pub fn save_frame(context: GpuContext, request: FrameRequest<'_>) -> Result<(), MediaError> {
    check_cancel(request.cancel)?;
    let depth = check_inputs(request.project, request.depth)?;
    request.project.timeline.frame_time(request.frame)?;
    let destination = Destination::check(request.output, request.overwrite)?;
    let mut renderer = renderer(context, request.project, request.image, depth)?;
    renderer.render(request.project, request.frame, RenderView::Motion)?;
    check_cancel(request.cancel)?;
    let size = renderer.output_size();
    let mut temp = destination.temp_file()?;
    let file = temp.as_file_mut();
    pollster::block_on(
        renderer.with_rgba8(|bytes, pitch| write_png(file, size, bytes, pitch, &request)),
    )??;
    check_cancel(request.cancel)?;
    destination.publish(temp)
}

/// Streams `height` rows of `width * 4` bytes, each starting at `y * pitch`, as an RGBA8 PNG.
fn write_png(
    file: &mut File,
    size: Size,
    bytes: &[u8],
    pitch: usize,
    request: &FrameRequest<'_>,
) -> Result<(), MediaError> {
    write_rgba8_png(&mut BufWriter::new(file), size, bytes, pitch, || {
        request
            .cancel
            .is_some_and(|flag| flag.load(Ordering::Acquire))
    })
    .map_err(|error| match error {
        PngError::Cancelled => MediaError::Cancelled,
        PngError::Io(error) => MediaError::io("write PNG frame", request.output, error),
        other => MediaError::FrameEncode(other),
    })
}
