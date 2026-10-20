//! Browser WebGPU device acquisition and canvas presentation (wasm32 only).
//!
//! The canvas is configured `Rgba8Unorm` with `COPY_DST`, and each frame is one
//! texture copy from the renderer output. Should a browser refuse that format,
//! the shared renderer's own WGSL blit (`Renderer::present_to`) writes the
//! canvas format instead. Neither path reads pixels back to the CPU.

use burns::renderer::COLOR_FORMAT;
use burns::{GpuContext, Renderer, Size};
use wasm_bindgen::JsValue;
use web_sys::HtmlCanvasElement;

use crate::error::{EngineError, Result};

pub struct CanvasSurface {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    /// `true`: copy the `Rgba8Unorm` output; `false`: blit into another format.
    copy: bool,
}

/// Requests a WebGPU adapter compatible with `canvas` and a device with the
/// adapter's texture-size and buffer limits (the same limits as native).
pub async fn request(canvas: HtmlCanvasElement) -> Result<(GpuContext, CanvasSurface)> {
    check_environment()?;
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(descriptor);
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
        .map_err(|error| {
            EngineError::new(format!(
                "could not create a WebGPU context on the canvas: {error}; use a canvas that \
                 has no 2D or WebGL context"
            ))
        })?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            apply_limit_buckets: false,
        })
        .await
        .map_err(|error| {
            EngineError::new(format!(
                "the browser exposes WebGPU but provided no GPU adapter ({error}); enable \
                 hardware acceleration, update the GPU driver, or check the browser's GPU \
                 status page"
            ))
        })?;
    let info = adapter.get_info();
    let supported = adapter.limits();
    let mut limits = wgpu::Limits::default().using_resolution(supported.clone());
    limits.max_buffer_size = supported.max_buffer_size;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("burns"),
            required_limits: limits,
            ..Default::default()
        })
        .await
        .map_err(|error| {
            EngineError::new(format!(
                "the WebGPU adapter {} could not create a device: {error}",
                info.name
            ))
        })?;

    let capabilities = surface.get_capabilities(&adapter);
    let copy = capabilities.formats.contains(&COLOR_FORMAT);
    let (format, usage) = if copy {
        (
            COLOR_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
        )
    } else {
        let format = *capabilities.formats.first().ok_or_else(|| {
            EngineError::new("the canvas reports no supported WebGPU texture format")
        })?;
        (format, wgpu::TextureUsages::RENDER_ATTACHMENT)
    };
    let config = wgpu::SurfaceConfiguration {
        usage,
        format,
        color_space: wgpu::SurfaceColorSpace::Auto,
        // Replaced by the renderer output size in `configure`.
        width: 1,
        height: 1,
        desired_maximum_frame_latency: 2,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: Vec::new(),
    };
    Ok((
        GpuContext { device, queue },
        CanvasSurface {
            surface,
            config,
            copy,
        },
    ))
}

/// WebGPU only exists in secure contexts on browsers that ship it.
fn check_environment() -> Result<()> {
    let window = web_sys::window().ok_or_else(|| {
        EngineError::new("initCanvas needs a page window; call it from the main thread")
    })?;
    if !window.is_secure_context() {
        return Err(EngineError::new(
            "WebGPU requires a secure context: serve the page over HTTPS or from \
             http://localhost or http://127.0.0.1",
        ));
    }
    let gpu = js_sys::Reflect::get(&window, &JsValue::from_str("navigator"))
        .and_then(|navigator| js_sys::Reflect::get(&navigator, &JsValue::from_str("gpu")))
        .unwrap_or(JsValue::UNDEFINED);
    if gpu.is_undefined() || gpu.is_null() {
        return Err(EngineError::new(
            "this browser does not expose WebGPU (navigator.gpu is missing); use a current \
             Chrome, Edge, or Safari, or enable WebGPU in Firefox",
        ));
    }
    Ok(())
}

impl CanvasSurface {
    /// Sizes the canvas drawing buffer to the renderer output and configures it.
    pub fn configure(&mut self, renderer: &Renderer) {
        let Size { width, height } = renderer.output_size();
        self.config.width = width;
        self.config.height = height;
        self.surface
            .configure(&renderer.context().device, &self.config);
    }

    /// Copies (or blits) the rendered output into the canvas and presents it.
    pub fn present(&mut self, renderer: &mut Renderer) -> Result<()> {
        let size = renderer.output_size();
        if (self.config.width, self.config.height) != (size.width, size.height) {
            self.configure(renderer);
        }
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Timeout => {
                return Err(EngineError::new(
                    "timed out acquiring the canvas texture; call renderFrame again",
                ));
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                return Err(EngineError::new(
                    "the canvas is occluded; call renderFrame again once it is visible",
                ));
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.configure(renderer);
                return Err(EngineError::new(
                    "the canvas configuration was outdated and has been reapplied; call \
                     renderFrame again",
                ));
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                return Err(EngineError::new(
                    "the WebGPU canvas context was lost or rejected its configuration; await \
                     initCanvas(canvas) again",
                ));
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(EngineError::new(
                    "acquiring the canvas texture raised a WebGPU validation error; see the \
                     browser console",
                ));
            }
        };
        if self.copy {
            let GpuContext { device, queue } = renderer.context();
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burns canvas copy"),
            });
            encoder.copy_texture_to_texture(
                renderer.output_texture().as_image_copy(),
                frame.texture.as_image_copy(),
                wgpu::Extent3d {
                    width: size.width,
                    height: size.height,
                    depth_or_array_layers: 1,
                },
            );
            queue.submit([encoder.finish()]);
        } else {
            let view = frame.texture.create_view(&Default::default());
            renderer.present_to(&view, self.config.format)?;
        }
        renderer.context().queue.present(frame);
        Ok(())
    }
}
