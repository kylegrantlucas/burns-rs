//! Live preview: one burns [`Renderer`] at preview quality drawing into its own output texture,
//! which egui samples directly as a registered native texture. Nothing is read back from the
//! GPU or uploaded per frame; assets are uploaded once when they change and a frame is only
//! rendered when its inputs differ from the last render.

use burns::{
    DepthMap, GpuContext, MotionProject, RenderError, RenderQuality, RenderView, Renderer, Size,
    SourceImage,
};
use eframe::egui::TextureId;
use egui_wgpu::{RenderState, wgpu};

/// Everything that determines the pixels of the output texture.
#[derive(Clone, PartialEq)]
struct RenderKey {
    project: MotionProject,
    frame: u32,
    view: RenderView,
    source: u64,
    depth: Option<u64>,
}

pub struct Preview {
    renderer: Renderer,
    render_state: RenderState,
    texture: TextureId,
    /// Generation id of the uploaded source and depth map.
    source: Option<u64>,
    depth: Option<u64>,
    last: Option<RenderKey>,
}

impl Preview {
    /// Compiles the renderer pipelines on egui's device and registers the output texture.
    pub fn new(render_state: &RenderState, output: Size) -> Result<Self, RenderError> {
        let context = GpuContext {
            device: render_state.device.clone(),
            queue: render_state.queue.clone(),
        };
        let renderer = Renderer::new(context, output, RenderQuality::Preview)?;
        let texture = render_state.renderer.write().register_native_texture(
            &render_state.device,
            renderer.output_view(),
            wgpu::FilterMode::Linear,
        );
        Ok(Self {
            renderer,
            render_state: render_state.clone(),
            texture,
            source: None,
            depth: None,
            last: None,
        })
    }

    /// The shared device and queue, for exports that run on the same GPU.
    pub fn context(&self) -> &GpuContext {
        self.renderer.context()
    }

    pub fn texture(&self) -> TextureId {
        self.texture
    }

    /// Size of the preview texture (the project output, long edge capped at 960 pixels).
    pub fn output_size(&self) -> Size {
        self.renderer.output_size()
    }

    /// Uploads a new source image once. `id` identifies it for render caching. On failure the
    /// previous source stays uploaded.
    pub fn set_source(&mut self, id: u64, image: &SourceImage) -> Result<(), RenderError> {
        self.last = None;
        self.renderer.set_source(image)?;
        self.source = Some(id);
        Ok(())
    }

    /// Uploads a depth map once, or clears it. On failure the previous map stays uploaded.
    pub fn set_depth(&mut self, depth: Option<(u64, &DepthMap)>) -> Result<(), RenderError> {
        self.last = None;
        self.renderer.set_depth(depth.map(|(_, map)| map))?;
        self.depth = depth.map(|(id, _)| id);
        Ok(())
    }

    /// Renders `view` of `project` at `frame` unless the texture already shows exactly that.
    /// Resizes the target (and re-registers it with egui) when the output size changes.
    pub fn render(
        &mut self,
        project: &MotionProject,
        frame: u32,
        view: RenderView,
    ) -> Result<(), RenderError> {
        let Some(source) = self.source else {
            return Err(RenderError::MissingSource);
        };
        let key = RenderKey {
            project: project.clone(),
            frame,
            view,
            source,
            depth: self.depth,
        };
        if self.last.as_ref() == Some(&key) {
            return Ok(());
        }
        self.last = None;
        if self.renderer.requested_size() != project.output {
            self.renderer
                .resize(project.output, RenderQuality::Preview)?;
            self.render_state
                .renderer
                .write()
                .update_egui_texture_from_wgpu_texture(
                    &self.render_state.device,
                    self.renderer.output_view(),
                    wgpu::FilterMode::Linear,
                    self.texture,
                );
        }
        self.renderer.render(project, frame, view)?;
        self.last = Some(key);
        Ok(())
    }
}

impl Drop for Preview {
    fn drop(&mut self) {
        self.render_state
            .renderer
            .write()
            .free_texture(&self.texture);
    }
}
