//! Preflight and renderer setup shared by PNG frames and MP4 export.

use burns::{
    CameraMotion, DepthMap, GpuContext, MotionProject, RenderQuality, Renderer, SourceImage,
};

use crate::MediaError;

/// Validates `project` and returns the depth map the render needs: `Some` only for the depth
/// camera. Any grid resolution is aligned across the full source UV extent.
pub(crate) fn check_inputs<'a>(
    project: &MotionProject,
    depth: Option<&'a DepthMap>,
) -> Result<Option<&'a DepthMap>, MediaError> {
    project.validate()?;
    if !matches!(project.camera, CameraMotion::Depth { .. }) {
        return Ok(None);
    }
    let depth = depth.ok_or(MediaError::MissingDepth)?;
    Ok(Some(depth))
}

/// A final-quality renderer at exactly `project.output` with the assets uploaded once.
///
/// Creating it checks the 2x supersampled target, readback, source, and depth against the
/// device limits.
pub(crate) fn renderer(
    context: GpuContext,
    project: &MotionProject,
    image: &SourceImage,
    depth: Option<&DepthMap>,
) -> Result<Renderer, MediaError> {
    let mut renderer = Renderer::new(context, project.output, RenderQuality::Final)?;
    renderer.set_source(image)?;
    if depth.is_some() {
        renderer.set_depth(depth)?;
    }
    Ok(renderer)
}
