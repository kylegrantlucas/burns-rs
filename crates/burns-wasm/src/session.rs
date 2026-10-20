//! Engine state independent of JavaScript: the project, retained assets, the
//! preview target, and the GPU slot.
//!
//! Every method is synchronous and borrows only for its own duration. The two
//! asynchronous operations (canvas initialization and frame capture) are split
//! into a `begin_*` step that reserves the GPU slot, a future that owns what it
//! needs, and a `finish_*` step that reinstalls state. Project controls remain
//! usable during initialization; capture holds all edits until restoration.

use burns::renderer::preview_size;
use burns::{
    DepthMap, FrameRate, GpuContext, MotionProject, RenderQuality, RenderView, Renderer, Size,
    SourceImage, Timeline,
};

use crate::error::{EngineError, Result};

/// Requested preview size and quality, retained across GPU (re)initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub size: Size,
    pub quality: RenderQuality,
}

impl Target {
    /// Output texture size the renderer allocates for this target.
    pub fn output_size(self) -> Size {
        match self.quality {
            RenderQuality::Preview => preview_size(self.size),
            RenderQuality::Final => self.size,
        }
    }
}

/// The renderer plus, in the browser, the canvas it presents to.
pub struct Gpu {
    renderer: Renderer,
    #[cfg(target_arch = "wasm32")]
    canvas: crate::canvas::CanvasSurface,
}

impl Gpu {
    /// Renders one view and presents it within the same task, so the canvas
    /// shows exactly this frame when control returns to JavaScript.
    fn draw(&mut self, project: &MotionProject, frame: u32, view: RenderView) -> Result<()> {
        self.renderer.render(project, frame, view)?;
        self.present()
    }

    #[cfg(target_arch = "wasm32")]
    fn present(&mut self) -> Result<()> {
        self.canvas.present(&mut self.renderer)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn present(&mut self) -> Result<()> {
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    fn configure(&mut self) {
        self.canvas.configure(&self.renderer);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn configure(&mut self) {}
}

enum GpuState {
    Uninitialized,
    Ready(Box<Gpu>),
    /// `framePng` owns the GPU until it resolves.
    Capturing,
}

struct GpuSlot {
    state: GpuState,
    /// An `initCanvas` is pending; a previous `Ready` GPU stays usable meanwhile.
    initializing: bool,
    /// Why the last `initCanvas` failed, for later "not initialized" errors.
    init_error: Option<String>,
}

impl GpuSlot {
    fn ready(&mut self) -> Result<&mut Gpu> {
        match &mut self.state {
            GpuState::Ready(gpu) => Ok(gpu),
            GpuState::Capturing => Err(capture_busy()),
            GpuState::Uninitialized => Err(not_initialized(
                self.initializing,
                self.init_error.as_deref(),
            )),
        }
    }

    /// The GPU to upload a changed asset or target to, `None` while there is
    /// none yet (initialization uploads the retained state), or busy.
    fn for_update(&mut self) -> Result<Option<&mut Gpu>> {
        match &mut self.state {
            GpuState::Ready(gpu) => Ok(Some(gpu)),
            GpuState::Capturing => Err(capture_busy()),
            GpuState::Uninitialized => Ok(None),
        }
    }
}

fn not_initialized(initializing: bool, init_error: Option<&str>) -> EngineError {
    let mut message = "WebGPU is not initialized: await initCanvas(canvas) first".to_owned();
    if initializing {
        message.push_str("; initCanvas is still in progress");
    } else if let Some(error) = init_error {
        message.push_str("; the last initCanvas failed: ");
        message.push_str(error);
    }
    EngineError::new(message)
}

fn capture_busy() -> EngineError {
    EngineError::new(
        "a framePng capture is in progress; wait for it to settle before editing, rendering, \
         loading assets, resizing, or initializing the canvas",
    )
}

/// What a frame capture owns while it awaits the readback.
pub struct Capture {
    gpu: Box<Gpu>,
    /// Settings snapshot: edits made while the capture is pending do not affect it.
    project: MotionProject,
    index: u32,
    previous: Target,
}

impl Capture {
    /// Renders and encodes the snapshot; the GPU stays owned by this capture.
    pub async fn run(&mut self) -> Result<Vec<u8>> {
        capture_png(&mut self.gpu.renderer, &self.project, self.index).await
    }
}

/// Pure session state plus the GPU slot.
pub struct Session {
    project: MotionProject,
    source: Option<SourceImage>,
    depth: Option<DepthMap>,
    target: Target,
    /// Last successfully presented frame and view, re-presented after a capture.
    shown: Option<(u32, RenderView)>,
    gpu: GpuSlot,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    /// Default project, no assets, no GPU; a preview target at the project output size.
    pub fn new() -> Self {
        let project = MotionProject::default();
        Self {
            target: Target {
                size: project.output,
                quality: RenderQuality::Preview,
            },
            project,
            source: None,
            depth: None,
            shown: None,
            gpu: GpuSlot {
                state: GpuState::Uninitialized,
                initializing: false,
                init_error: None,
            },
        }
    }

    // ---------------------------------------------------------------------
    // Project controls (never touch the GPU)

    fn check_editable(&self) -> Result<()> {
        if matches!(self.gpu.state, GpuState::Capturing) {
            Err(capture_busy())
        } else {
            Ok(())
        }
    }

    pub fn project_json(&self) -> Result<String> {
        Ok(self.project.to_json()?)
    }

    pub fn set_project_json(&mut self, json: &str) -> Result<String> {
        self.check_editable()?;
        self.project = MotionProject::from_json(json)?;
        self.project_json()
    }

    pub fn set_setting(&mut self, id: &str, raw: &str) -> Result<String> {
        self.check_editable()?;
        self.project.set_by_id(id, raw)?;
        self.project_json()
    }

    pub fn setting(&self, id: &str) -> Result<burns::SettingValue> {
        Ok(self.project.get_by_id(id)?)
    }

    pub fn set_timing(&mut self, fps_num: u32, fps_den: u32, seconds: f64) -> Result<String> {
        self.check_editable()?;
        let timeline = Timeline::from_duration(FrameRate::new(fps_num, fps_den), seconds)
            .map_err(|error| error.within("timeline"))?;
        self.project.set_timeline(timeline)?;
        self.project_json()
    }

    pub fn set_output_size(&mut self, width: u32, height: u32) -> Result<String> {
        self.check_editable()?;
        self.project.set_output(Size::new(width, height))?;
        self.project_json()
    }

    pub fn apply_preset(&mut self, name: &str) -> Result<String> {
        self.check_editable()?;
        self.project.apply_preset(name)?;
        self.project_json()
    }

    // ---------------------------------------------------------------------
    // Assets

    /// Decodes a still and makes it the source, clearing the depth map. Uploads
    /// it once when a GPU exists; on any error nothing changes.
    pub fn load_image(&mut self, bytes: &[u8]) -> Result<String> {
        let image = SourceImage::decode(bytes)?;
        if let Some(gpu) = self.gpu.for_update()? {
            gpu.renderer.set_source(&image)?;
            gpu.renderer.set_depth(None)?;
        }
        self.source = Some(image);
        self.depth = None;
        self.shown = None;
        self.project_json()
    }

    /// Decodes a raw 8/16-bit grayscale depth PNG exactly as stored.
    pub fn load_depth(&mut self, bytes: &[u8]) -> Result<String> {
        let map = DepthMap::decode_png(bytes)?;
        if let Some(gpu) = self.gpu.for_update()? {
            gpu.renderer.set_depth(Some(&map))?;
        }
        self.depth = Some(map);
        self.project_json()
    }

    pub fn source(&self) -> Option<&SourceImage> {
        self.source.as_ref()
    }

    pub fn depth(&self) -> Option<&DepthMap> {
        self.depth.as_ref()
    }

    // ---------------------------------------------------------------------
    // Preview target and rendering

    pub fn target(&self) -> Target {
        self.target
    }

    pub fn gpu_ready(&self) -> bool {
        matches!(self.gpu.state, GpuState::Ready(_))
    }

    /// Changes the preview target, keeping assets. With a GPU the renderer
    /// targets and the canvas are resized (which clears the canvas; render
    /// again); otherwise the target applies at initialization.
    pub fn resize_target(&mut self, width: u32, height: u32, final_quality: bool) -> Result<()> {
        let size = Size::new(width, height);
        size.validate().map_err(|error| error.within("target"))?;
        let target = Target {
            size,
            quality: if final_quality {
                RenderQuality::Final
            } else {
                RenderQuality::Preview
            },
        };
        if let Some(gpu) = self.gpu.for_update()? {
            gpu.renderer.resize(target.size, target.quality)?;
            gpu.configure();
        }
        self.target = target;
        Ok(())
    }

    pub fn render_frame(&mut self, index: u32, view: RenderView) -> Result<()> {
        self.gpu.ready()?.draw(&self.project, index, view)?;
        self.shown = Some((index, view));
        Ok(())
    }

    // ---------------------------------------------------------------------
    // GPU initialization

    /// Reserves initialization; fails while another initialization or a capture is pending.
    pub fn begin_init(&mut self) -> Result<()> {
        if self.gpu.initializing {
            return Err(EngineError::new(
                "initCanvas is already in progress; await the pending call",
            ));
        }
        if matches!(self.gpu.state, GpuState::Capturing) {
            return Err(capture_busy());
        }
        self.gpu.initializing = true;
        Ok(())
    }

    /// Builds the renderer for the current target and uploads the retained
    /// assets. On failure the previous GPU (if any), project, and assets stay.
    /// `device` comes from the browser canvas request (always an error natively).
    pub fn finish_init(&mut self, device: Result<(GpuContext, CanvasHandle)>) -> Result<()> {
        self.gpu.initializing = false;
        let result = device.and_then(|(context, canvas)| self.build_gpu(context, canvas));
        match result {
            Ok(gpu) => {
                self.gpu.state = GpuState::Ready(gpu);
                self.gpu.init_error = None;
                self.shown = None;
                Ok(())
            }
            Err(error) => {
                self.gpu.init_error = Some(error.to_string());
                Err(error)
            }
        }
    }

    /// The canvas surface goes unused natively, where `device` is never `Ok`.
    fn build_gpu(
        &self,
        context: GpuContext,
        #[cfg_attr(not(target_arch = "wasm32"), allow(unused_variables))] canvas: CanvasHandle,
    ) -> Result<Box<Gpu>> {
        let mut renderer = Renderer::new(context, self.target.size, self.target.quality)?;
        if let Some(source) = &self.source {
            renderer.set_source(source)?;
        }
        if let Some(depth) = &self.depth {
            renderer.set_depth(Some(depth))?;
        }
        let mut gpu = Box::new(Gpu {
            renderer,
            #[cfg(target_arch = "wasm32")]
            canvas,
        });
        gpu.configure();
        Ok(gpu)
    }

    // ---------------------------------------------------------------------
    // Frame capture

    /// Takes the GPU and snapshots the settings for one capture.
    pub fn begin_capture(&mut self, index: u32) -> Result<Capture> {
        if self.gpu.initializing {
            return Err(EngineError::new(
                "initCanvas is in progress; await it before calling framePng",
            ));
        }
        self.gpu.ready()?;
        let gpu = match std::mem::replace(&mut self.gpu.state, GpuState::Capturing) {
            GpuState::Ready(gpu) => gpu,
            other => {
                self.gpu.state = other;
                return Err(not_initialized(
                    self.gpu.initializing,
                    self.gpu.init_error.as_deref(),
                ));
            }
        };
        Ok(Capture {
            previous: Target {
                size: gpu.renderer.requested_size(),
                quality: gpu.renderer.quality(),
            },
            gpu,
            project: self.project.clone(),
            index,
        })
    }

    /// Restores the preview target and the shown frame/view, reinstalls the
    /// GPU, and returns the capture result. A restore failure is reported
    /// alongside (never instead of) a capture failure.
    pub fn finish_capture(
        &mut self,
        mut capture: Capture,
        captured: Result<Vec<u8>>,
    ) -> Result<Vec<u8>> {
        let restored = self.restore(&mut capture.gpu, capture.previous);
        self.gpu.state = GpuState::Ready(capture.gpu);
        match (captured, restored) {
            (Ok(png), Ok(())) => Ok(png),
            (Err(error), Ok(())) => Err(error),
            (Err(error), Err(restore)) => Err(EngineError::new(format!(
                "{error}; restoring the preview afterwards also failed: {restore}"
            ))),
            (Ok(_), Err(restore)) => Err(EngineError::new(format!(
                "frame {} was captured, but restoring the preview failed: {restore}; call \
                 resizeTarget and renderFrame again",
                capture.index
            ))),
        }
    }

    fn restore(&self, gpu: &mut Gpu, previous: Target) -> Result<()> {
        gpu.renderer.resize(previous.size, previous.quality)?;
        if let Some((frame, view)) = self.shown {
            gpu.draw(&self.project, frame, view)?;
        }
        Ok(())
    }
}

/// Canvas surface handed to [`Session::finish_init`]; nothing natively.
#[cfg(target_arch = "wasm32")]
pub type CanvasHandle = crate::canvas::CanvasSurface;
#[cfg(not(target_arch = "wasm32"))]
pub type CanvasHandle = ();

/// Renders frame `index` of `project` as Motion at the exact output size with
/// 2x supersampling, streaming the mapped rows straight into a PNG encoder.
/// GPU validation and out-of-memory errors raised meanwhile are reported.
pub async fn capture_png(
    renderer: &mut Renderer,
    project: &MotionProject,
    index: u32,
) -> Result<Vec<u8>> {
    let device = renderer.context().device.clone();
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let result = render_png(renderer, project, index).await;
    let memory = memory.pop().await;
    let validation = validation.pop().await;
    let gpu_error = memory
        .map(|error| format!("GPU out of memory during frame capture: {error}"))
        .or_else(|| {
            validation.map(|error| format!("GPU validation failed during frame capture: {error}"))
        });
    match (gpu_error, result) {
        (None, result) => result,
        (Some(gpu), Ok(_)) => Err(EngineError::new(gpu)),
        (Some(gpu), Err(error)) => Err(EngineError::new(format!("{error} ({gpu})"))),
    }
}

async fn render_png(
    renderer: &mut Renderer,
    project: &MotionProject,
    index: u32,
) -> Result<Vec<u8>> {
    renderer.resize(project.output, RenderQuality::Final)?;
    renderer.render(project, index, RenderView::Motion)?;
    let size = renderer.output_size();
    renderer
        .with_rgba8(|rows, pitch| encode_png(size, rows, pitch))
        .await?
}

fn encode_png(size: Size, rows: &[u8], pitch: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    burns::write_rgba8_png(&mut out, size, rows, pitch, || false)
        .map_err(|error| EngineError::new(error.to_string()))?;
    Ok(out)
}
