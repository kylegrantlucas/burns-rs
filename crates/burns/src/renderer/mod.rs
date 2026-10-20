//! Shared native/WebGPU renderer.
//!
//! One [`Renderer`] owns its pipelines, render targets, uniforms, bind groups,
//! asset textures, and readback buffer. Assets are uploaded once by
//! [`Renderer::set_source`]/[`Renderer::set_depth`]; targets are recreated only by
//! [`Renderer::resize`]. A frame writes one small uniform block and records one
//! or two fullscreen passes: no shader compilation, pixel allocation, or asset
//! upload happens per frame.
//!
//! Colors are sRGB-encoded bytes stored in `Rgba8Unorm` textures and are never
//! passed through an extra transfer function. Depth maps are `R32Float` and
//! filtered manually with four clamped texel loads, so no float32-filtering
//! feature is required.

use std::borrow::Cow;
use std::fmt;

use bytemuck::{Pod, Zeroable};
use futures_channel::oneshot;

use crate::image::{CoverMap, DepthMap, SourceImage};
use crate::motion::{CameraPose, EvaluatedMotion, MotionEvaluator, WarpField};
use crate::settings::{ConfigError, MotionProject, Size};

mod loss;
use loss::DeviceLoss;

/// Longest output edge in [`RenderQuality::Preview`].
pub const PREVIEW_MAX_EDGE: u32 = 960;

/// Per-axis supersampling factor of [`RenderQuality::Final`].
pub const FINAL_SUPERSAMPLE: u32 = 2;

/// Format of the source, intermediate, and output textures.
pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Backends [`GpuContext::headless`] may use: Metal, Vulkan, and DirectX 12
/// natively; WebGPU in the browser. Never a no-op or GL fallback.
pub const HEADLESS_BACKENDS: wgpu::Backends = if cfg!(target_arch = "wasm32") {
    wgpu::Backends::BROWSER_WEBGPU
} else {
    wgpu::Backends::METAL
        .union(wgpu::Backends::VULKAN)
        .union(wgpu::Backends::DX12)
};

const MODE_PLANAR: u32 = 0;
const MODE_DEPTH: u32 = 1;
const MODE_SOURCE: u32 = 2;
const MODE_DEPTH_VIEW: u32 = 3;

const WARP_NONE: u32 = 0;
const WARP_RIPPLE: u32 = 1;
const WARP_SHIMMER: u32 = 2;

/// Errors from GPU setup, asset upload, rendering, and readback.
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    /// No hardware adapter on the allowed backends.
    #[error(
        "no GPU adapter is available on {backends}: {message}; install or update a Metal, Vulkan, \
         or DirectX 12 driver, or use a browser with WebGPU enabled"
    )]
    NoAdapter { backends: String, message: String },
    /// The adapter refused to create a device.
    #[error("the GPU adapter {adapter} could not create a device: {message}")]
    RequestDevice { adapter: String, message: String },
    /// The device was lost (driver reset, GPU removal, or destruction).
    #[error("the GPU device was lost ({reason}); create a new GpuContext and Renderer")]
    DeviceLost { reason: String },
    /// A texture would exceed the device's maximum 2D texture dimension.
    #[error(
        "{what} of {width}x{height} pixels exceeds this GPU's {max}-pixel texture limit; {remedy}"
    )]
    TextureLimit {
        what: &'static str,
        width: u64,
        height: u64,
        max: u32,
        remedy: &'static str,
    },
    /// The readback buffer would exceed the device's maximum buffer size.
    #[error(
        "reading back a {width}x{height} output needs {bytes} bytes, above this GPU's {max}-byte \
         buffer limit; resize the render target smaller"
    )]
    BufferLimit {
        width: u32,
        height: u32,
        bytes: u64,
        max: u64,
    },
    /// Rendering was requested before a source image was set.
    #[error("no source image is loaded; call Renderer::set_source before rendering")]
    MissingSource,
    /// A depth camera or depth inspection was requested without a depth map.
    #[error(
        "no depth map is loaded: load or estimate a depth map and pass it to Renderer::set_depth; \
         render RenderView::Source until one exists"
    )]
    MissingDepth,
    /// The project, frame, or requested size is invalid.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// The readback destination stride is shorter than one pixel row.
    #[error("readback stride {stride} is shorter than one {width}-pixel RGBA row of {row} bytes")]
    DestinationStride {
        stride: usize,
        width: u32,
        row: usize,
    },
    /// The readback destination cannot hold every row at the given stride.
    #[error(
        "readback destination holds {len} bytes, but {height} rows of {width} pixels at stride \
         {stride} need {required}"
    )]
    DestinationTooShort {
        len: usize,
        required: usize,
        width: u32,
        height: u32,
        stride: usize,
    },
    /// Mapping the readback buffer failed.
    #[error("GPU readback failed: {message}")]
    Readback { message: String },
}

/// Shared device and queue. Cheap to clone; every clone refers to the same GPU.
#[derive(Clone)]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let info = self.device.adapter_info();
        f.debug_struct("GpuContext")
            .field("adapter", &info.name)
            .field("backend", &info.backend)
            .finish()
    }
}

impl GpuContext {
    /// Requests a high-performance adapter on [`HEADLESS_BACKENDS`] and a device
    /// with the adapter's texture-size limits.
    pub async fn headless() -> Result<Self, RenderError> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = HEADLESS_BACKENDS;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|error| RenderError::NoAdapter {
                backends: format!("{HEADLESS_BACKENDS:?}"),
                message: error.to_string(),
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
            .map_err(|error| RenderError::RequestDevice {
                adapter: format!("{} ({:?})", info.name, info.backend),
                message: error.to_string(),
            })?;
        Ok(Self { device, queue })
    }
}

/// Render resolution policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderQuality {
    /// Long edge capped at [`PREVIEW_MAX_EDGE`], keeping the requested aspect.
    Preview,
    /// Rendered at [`FINAL_SUPERSAMPLE`]x per axis, box-resolved to the exact size.
    Final,
}

/// What a render shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderView {
    /// The animated project; exports always use this.
    Motion,
    /// The unanimated cover-cropped source. Works without a depth map.
    Source,
    /// The unwarped cover-cropped effective depth map (`1 - raw` when inverted).
    Depth,
}

/// Uniform block mirrored by `Params` in `motion.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    frame_map: [[f32; 4]; 3],
    cover: [f32; 4],
    warp_a: [f32; 4],
    warp_b: [f32; 4],
    depth_a: [f32; 4],
    depth_b: [f32; 4],
    modes: [u32; 4],
}

impl Params {
    fn motion(motion: &EvaluatedMotion, params: &mut Self) {
        let matrix = motion.frame_matrix();
        for (row, values) in params.frame_map.iter_mut().zip(matrix) {
            *row = [values[0] as f32, values[1] as f32, values[2] as f32, 0.0];
        }
        match motion.warp {
            WarpField::None => params.modes[1] = WARP_NONE,
            WarpField::Ripple {
                center,
                amplitude,
                wavelength,
                phase,
            } => {
                params.modes[1] = WARP_RIPPLE;
                params.warp_a = [
                    center[0] as f32,
                    center[1] as f32,
                    amplitude as f32,
                    (1.0 / wavelength) as f32,
                ];
                params.warp_b = [(wavelength / 4.0) as f32, phase as f32, 0.0, 0.0];
            }
            WarpField::HeatShimmer {
                amplitude,
                spatial_scale,
                phases,
            } => {
                let tau = std::f64::consts::TAU * spatial_scale;
                params.modes[1] = WARP_SHIMMER;
                params.warp_a = [
                    amplitude as f32,
                    (6.0 * tau) as f32,
                    (11.0 * tau) as f32,
                    (4.0 * tau) as f32,
                ];
                params.warp_b = [phases[0] as f32, phases[1] as f32, phases[2] as f32, 0.0];
            }
        }
        match motion.camera {
            CameraPose::Depth(depth) => {
                let k = depth.coefficients();
                params.modes[0] = MODE_DEPTH;
                params.depth_a = [
                    depth.height as f32,
                    k.far_scale as f32,
                    k.origin_scale as f32,
                    k.offset_far as f32,
                ];
                params.depth_b = [depth.offset[0] as f32, depth.offset[1] as f32, 0.0, 0.0];
            }
            CameraPose::View(_) | CameraPose::Tilt { .. } => params.modes[0] = MODE_PLANAR,
        }
    }
}

/// Render targets and readback storage for one requested size and quality.
struct Targets {
    requested: Size,
    quality: RenderQuality,
    output_size: Size,
    output: wgpu::Texture,
    output_view: wgpu::TextureView,
    /// Final quality: the 2x motion target and its resolve bind group.
    supersample: Option<(wgpu::TextureView, wgpu::BindGroup)>,
    present_group: wgpu::BindGroup,
    readback: wgpu::Buffer,
    padded_row: u32,
}

struct SourceTexture {
    size: Size,
    view: wgpu::TextureView,
}

/// Unmaps the readback buffer when dropped, including during a consumer panic.
struct MappedReadback<'a>(&'a wgpu::Buffer);

impl Drop for MappedReadback<'_> {
    fn drop(&mut self) {
        self.0.unmap();
    }
}

/// Reusable GPU renderer for one source, one optional depth map, and one target size.
pub struct Renderer {
    context: GpuContext,
    lost: DeviceLoss,
    motion_pipeline: wgpu::RenderPipeline,
    resolve_pipeline: wgpu::RenderPipeline,
    image_module: wgpu::ShaderModule,
    image_layout: wgpu::BindGroupLayout,
    image_pipeline_layout: wgpu::PipelineLayout,
    motion_layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    sampler: wgpu::Sampler,
    placeholder_source: wgpu::TextureView,
    placeholder_depth: wgpu::TextureView,
    source: Option<SourceTexture>,
    depth: Option<wgpu::TextureView>,
    motion_group: wgpu::BindGroup,
    targets: Targets,
    evaluator: Option<MotionEvaluator>,
    present_pipelines: Vec<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
}

impl fmt::Debug for Renderer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Renderer")
            .field("requested", &self.targets.requested)
            .field("quality", &self.targets.quality)
            .field("output_size", &self.targets.output_size)
            .field("source", &self.source.as_ref().map(|s| s.size))
            .field("depth", &self.depth.is_some())
            .finish()
    }
}

impl Renderer {
    /// Compiles the pipelines and allocates targets for `size` at `quality`.
    ///
    /// Installs one shared device-loss observation for live burns renderers on
    /// `context.device`, so preview and export both report [`RenderError::DeviceLost`].
    pub fn new(
        context: GpuContext,
        size: Size,
        quality: RenderQuality,
    ) -> Result<Self, RenderError> {
        size.validate().map_err(|e| e.within("size"))?;
        let device = &context.device;

        let lost = DeviceLoss::new(device)?;

        let motion_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("burns motion"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("motion.wgsl"))),
        });
        let image_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("burns resolve"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("resolve.wgsl"))),
        });

        let fragment = wgpu::ShaderStages::FRAGMENT;
        let texture = |binding, filterable| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: fragment,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let sampler_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: fragment,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let motion_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("burns motion"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: fragment,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(size_of::<Params>() as u64),
                    },
                    count: None,
                },
                texture(1, true),
                sampler_entry(2),
                texture(3, false),
            ],
        });
        let image_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("burns image"),
            entries: &[texture(0, true), sampler_entry(1)],
        });
        let motion_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("burns motion"),
                bind_group_layouts: &[Some(&motion_layout)],
                immediate_size: 0,
            });
        let image_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("burns image"),
                bind_group_layouts: &[Some(&image_layout)],
                immediate_size: 0,
            });
        let motion_pipeline = fullscreen_pipeline(
            device,
            "burns motion",
            &motion_pipeline_layout,
            &motion_module,
            "fs_main",
            COLOR_FORMAT,
        );
        let resolve_pipeline = fullscreen_pipeline(
            device,
            "burns resolve",
            &image_pipeline_layout,
            &image_module,
            "fs_resolve",
            COLOR_FORMAT,
        );

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("burns params"),
            size: size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("burns bilinear"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let one = Size::new(1, 1);
        let placeholder_source = upload(
            &context,
            "burns placeholder source",
            one,
            COLOR_FORMAT,
            &[0, 0, 0, 255],
        );
        let placeholder_depth = upload(
            &context,
            "burns placeholder depth",
            one,
            DEPTH_FORMAT,
            bytemuck::bytes_of(&0.0f32),
        );
        let motion_group = motion_group(
            device,
            &motion_layout,
            &params,
            &placeholder_source,
            &sampler,
            &placeholder_depth,
        );
        let targets = Targets::new(device, size, quality, &image_layout, &sampler)?;

        Ok(Self {
            lost,
            motion_pipeline,
            resolve_pipeline,
            image_module,
            image_layout,
            image_pipeline_layout,
            motion_layout,
            params,
            sampler,
            placeholder_source,
            placeholder_depth,
            source: None,
            depth: None,
            motion_group,
            targets,
            evaluator: None,
            present_pipelines: Vec::new(),
            context,
        })
    }

    /// Recreates the targets when the requested size or quality changes.
    pub fn resize(&mut self, size: Size, quality: RenderQuality) -> Result<(), RenderError> {
        self.check_device()?;
        size.validate().map_err(|e| e.within("size"))?;
        if self.targets.requested == size && self.targets.quality == quality {
            return Ok(());
        }
        self.targets = Targets::new(
            &self.context.device,
            size,
            quality,
            &self.image_layout,
            &self.sampler,
        )?;
        Ok(())
    }

    /// Uploads the source image once; it stays resident until replaced.
    pub fn set_source(&mut self, source: &SourceImage) -> Result<(), RenderError> {
        self.check_device()?;
        let size = source.size();
        check_extent(
            &self.context.device,
            "source image",
            u64::from(size.width),
            u64::from(size.height),
            "downscale the source image before loading it",
        )?;
        let view = upload(
            &self.context,
            "burns source",
            size,
            COLOR_FORMAT,
            source.pixels(),
        );
        self.source = Some(SourceTexture { size, view });
        self.rebind();
        Ok(())
    }

    /// Uploads a depth map once, or clears it with `None`.
    pub fn set_depth(&mut self, depth: Option<&DepthMap>) -> Result<(), RenderError> {
        self.check_device()?;
        self.depth = match depth {
            Some(map) => {
                let size = map.size();
                check_extent(
                    &self.context.device,
                    "depth map",
                    u64::from(size.width),
                    u64::from(size.height),
                    "downscale the depth map before loading it",
                )?;
                Some(upload(
                    &self.context,
                    "burns depth",
                    size,
                    DEPTH_FORMAT,
                    bytemuck::cast_slice(map.values()),
                ))
            }
            None => None,
        };
        self.rebind();
        Ok(())
    }

    /// Renders `view` of `project` at `frame` into the output texture.
    ///
    /// Geometry uses the project output aspect regardless of the target size.
    /// [`RenderView::Source`] and [`RenderView::Depth`] ignore `frame` and motion.
    pub fn render(
        &mut self,
        project: &MotionProject,
        frame: u32,
        view: RenderView,
    ) -> Result<(), RenderError> {
        self.check_device()?;
        let source_size = self.source.as_ref().ok_or(RenderError::MissingSource)?.size;
        project.output.validate().map_err(|e| e.within("output"))?;
        check_extent(
            &self.context.device,
            "project output",
            u64::from(project.output.width),
            u64::from(project.output.height),
            "resize the project output smaller",
        )?;
        let cover = CoverMap::new(source_size, project.output)
            .map_err(|error| ConfigError::invalid("output", error.to_string()))?
            .scale();

        let mut params = Params::zeroed();
        params.cover = [
            cover[0],
            cover[1],
            (f64::from(project.output.width) / f64::from(project.output.height)) as f32,
            0.0,
        ];
        params.modes[2] = u32::from(project.depth.invert);
        match view {
            RenderView::Source => params.modes[0] = MODE_SOURCE,
            RenderView::Depth => {
                if self.depth.is_none() {
                    return Err(RenderError::MissingDepth);
                }
                params.modes[0] = MODE_DEPTH_VIEW;
            }
            RenderView::Motion => {
                if self
                    .evaluator
                    .as_ref()
                    .is_none_or(|evaluator| evaluator.project() != project)
                {
                    self.evaluator = Some(MotionEvaluator::new(project)?);
                }
                let motion = self
                    .evaluator
                    .as_ref()
                    .expect("evaluator was just ensured")
                    .evaluate(frame)?;
                if matches!(motion.camera, CameraPose::Depth(_)) && self.depth.is_none() {
                    return Err(RenderError::MissingDepth);
                }
                Params::motion(&motion, &mut params);
            }
        }

        let GpuContext { device, queue } = &self.context;
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("burns render"),
        });
        let targets = &self.targets;
        let motion_target = targets
            .supersample
            .as_ref()
            .map_or(&targets.output_view, |(view, _)| view);
        draw(
            &mut encoder,
            motion_target,
            &self.motion_pipeline,
            &self.motion_group,
        );
        if let Some((_, resolve_group)) = &targets.supersample {
            draw(
                &mut encoder,
                &targets.output_view,
                &self.resolve_pipeline,
                resolve_group,
            );
        }
        queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Blits the output into `target` (for example a canvas or window surface),
    /// scaling to its size. `format` must be the view's format; sRGB formats are
    /// decoded first so the stored bytes match the output.
    pub fn present_to(
        &mut self,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
    ) -> Result<(), RenderError> {
        self.check_device()?;
        let index = match self
            .present_pipelines
            .iter()
            .position(|(f, _)| *f == format)
        {
            Some(index) => index,
            None => {
                let entry = if format.is_srgb() {
                    "fs_present_srgb"
                } else {
                    "fs_present"
                };
                let pipeline = fullscreen_pipeline(
                    &self.context.device,
                    "burns present",
                    &self.image_pipeline_layout,
                    &self.image_module,
                    entry,
                    format,
                );
                self.present_pipelines.push((format, pipeline));
                self.present_pipelines.len() - 1
            }
        };
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("burns present"),
                });
        draw(
            &mut encoder,
            target,
            &self.present_pipelines[index].1,
            &self.targets.present_group,
        );
        self.context.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Size of the output texture: the requested size for final quality, or the
    /// preview-capped size.
    pub fn output_size(&self) -> Size {
        self.targets.output_size
    }

    /// Size passed to [`Self::new`] or [`Self::resize`].
    pub fn requested_size(&self) -> Size {
        self.targets.requested
    }

    pub fn quality(&self) -> RenderQuality {
        self.targets.quality
    }

    /// Output texture (`Rgba8Unorm`, sRGB-encoded, opaque).
    pub fn output_texture(&self) -> &wgpu::Texture {
        &self.targets.output
    }

    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.targets.output_view
    }

    pub fn context(&self) -> &GpuContext {
        &self.context
    }

    /// Copies the output into the readback buffer and lends its mapped rows to
    /// `consume` as `(bytes, row_pitch)`. Row `y` starts at `y * row_pitch`; the
    /// slice ends after the last pixel of the last row. The buffer is unmapped
    /// afterwards even if `consume` panics. Works on native and WebGPU (the map
    /// is awaited, never blocked on, in the browser).
    pub async fn with_rgba8<R>(
        &mut self,
        consume: impl FnOnce(&[u8], usize) -> R,
    ) -> Result<R, RenderError> {
        self.check_device()?;
        let GpuContext { device, queue } = &self.context;
        let targets = &self.targets;
        let Size { width, height } = targets.output_size;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("burns readback"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &targets.output,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &targets.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(targets.padded_row),
                    rows_per_image: Some(height),
                },
            },
            extent(targets.output_size),
        );
        let submission = queue.submit([encoder.finish()]);

        let (sender, receiver) = oneshot::channel();
        let slice = targets.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            // The receiver only disappears if this future was dropped.
            let _ = sender.send(result);
        });
        let mapping = MappedReadback(&targets.readback);
        // Blocks natively until the copy and map complete; a no-op on WebGPU,
        // where the browser resolves the map callback.
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| {
                self.lost_or(RenderError::Readback {
                    message: error.to_string(),
                })
            })?;
        match receiver.await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                return Err(self.lost_or(RenderError::Readback {
                    message: error.to_string(),
                }));
            }
            Err(_) => {
                return Err(self.lost_or(RenderError::Readback {
                    message: "the buffer map callback was dropped".to_owned(),
                }));
            }
        }
        let bytes = slice
            .get_mapped_range()
            .map_err(|error| RenderError::Readback {
                message: error.to_string(),
            })?;
        let row = width as usize * 4;
        let pitch = targets.padded_row as usize;
        let used = pitch * (height as usize - 1) + row;
        let result = consume(&bytes[..used], pitch);
        drop(bytes);
        drop(mapping);
        Ok(result)
    }

    /// Copies the output into `dst` as RGBA8 rows starting every `stride` bytes.
    /// Bytes between rows and after the last row are left untouched.
    pub async fn read_rgba8(&mut self, dst: &mut [u8], stride: usize) -> Result<(), RenderError> {
        let Size { width, height } = self.targets.output_size;
        let row = width as usize * 4;
        if stride < row {
            return Err(RenderError::DestinationStride { stride, width, row });
        }
        let required = stride
            .checked_mul(height as usize - 1)
            .and_then(|bytes| bytes.checked_add(row))
            .unwrap_or(usize::MAX);
        if dst.len() < required {
            return Err(RenderError::DestinationTooShort {
                len: dst.len(),
                required,
                width,
                height,
                stride,
            });
        }
        self.with_rgba8(|src, pitch| {
            for y in 0..height as usize {
                dst[y * stride..y * stride + row].copy_from_slice(&src[y * pitch..y * pitch + row]);
            }
        })
        .await
    }

    fn check_device(&self) -> Result<(), RenderError> {
        self.lost.check()
    }

    fn lost_or(&self, error: RenderError) -> RenderError {
        self.check_device().err().unwrap_or(error)
    }

    fn rebind(&mut self) {
        self.motion_group = motion_group(
            &self.context.device,
            &self.motion_layout,
            &self.params,
            self.source
                .as_ref()
                .map_or(&self.placeholder_source, |source| &source.view),
            &self.sampler,
            self.depth.as_ref().unwrap_or(&self.placeholder_depth),
        );
    }
}

impl Targets {
    fn new(
        device: &wgpu::Device,
        requested: Size,
        quality: RenderQuality,
        image_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
    ) -> Result<Self, RenderError> {
        check_extent(
            device,
            "requested output",
            u64::from(requested.width),
            u64::from(requested.height),
            "resize the requested output smaller",
        )?;
        let output_size = match quality {
            RenderQuality::Preview => preview_size(requested),
            RenderQuality::Final => requested,
        };
        let render_size = match quality {
            RenderQuality::Preview => [u64::from(output_size.width), u64::from(output_size.height)],
            RenderQuality::Final => [
                u64::from(requested.width) * u64::from(FINAL_SUPERSAMPLE),
                u64::from(requested.height) * u64::from(FINAL_SUPERSAMPLE),
            ],
        };
        check_extent(
            device,
            match quality {
                RenderQuality::Preview => "preview target",
                RenderQuality::Final => "2x supersampled final target",
            },
            render_size[0],
            render_size[1],
            "resize the render target smaller",
        )?;

        let padded_row =
            (output_size.width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback_bytes = u64::from(padded_row) * u64::from(output_size.height);
        let max = device.limits().max_buffer_size;
        if readback_bytes > max {
            return Err(RenderError::BufferLimit {
                width: output_size.width,
                height: output_size.height,
                bytes: readback_bytes,
                max,
            });
        }

        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let output = create_texture(
            device,
            "burns output",
            output_size,
            COLOR_FORMAT,
            usage | wgpu::TextureUsages::COPY_SRC,
        );
        let output_view = output.create_view(&Default::default());
        let supersample = (quality == RenderQuality::Final).then(|| {
            // `check_extent` bounded both dimensions by a u32 texture limit.
            let size = Size::new(render_size[0] as u32, render_size[1] as u32);
            let view = create_texture(device, "burns supersample", size, COLOR_FORMAT, usage)
                .create_view(&Default::default());
            let group = image_group(device, image_layout, &view, sampler);
            (view, group)
        });
        let present_group = image_group(device, image_layout, &output_view, sampler);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("burns readback"),
            size: readback_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            requested,
            quality,
            output_size,
            output,
            output_view,
            supersample,
            present_group,
            readback,
            padded_row,
        })
    }
}

/// Requested size scaled so its long edge is at most [`PREVIEW_MAX_EDGE`],
/// rounding to keep the aspect.
pub fn preview_size(requested: Size) -> Size {
    let long = requested.width.max(requested.height);
    if long <= PREVIEW_MAX_EDGE {
        return requested;
    }
    let scale = |edge: u32| {
        let scaled =
            (u64::from(edge) * u64::from(PREVIEW_MAX_EDGE) + u64::from(long) / 2) / u64::from(long);
        // At most PREVIEW_MAX_EDGE, so it fits in u32.
        (scaled as u32).max(1)
    };
    Size::new(scale(requested.width), scale(requested.height))
}

fn check_extent(
    device: &wgpu::Device,
    what: &'static str,
    width: u64,
    height: u64,
    remedy: &'static str,
) -> Result<(), RenderError> {
    let max = device.limits().max_texture_dimension_2d;
    if width > u64::from(max) || height > u64::from(max) {
        return Err(RenderError::TextureLimit {
            what,
            width,
            height,
            max,
            remedy,
        });
    }
    Ok(())
}

fn extent(size: Size) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: size.width,
        height: size.height,
        depth_or_array_layers: 1,
    }
}

fn create_texture(
    device: &wgpu::Device,
    label: &str,
    size: Size,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: extent(size),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

/// Creates a sampled texture holding `data` (tightly packed rows of 4-byte texels).
fn upload(
    context: &GpuContext,
    label: &str,
    size: Size,
    format: wgpu::TextureFormat,
    data: &[u8],
) -> wgpu::TextureView {
    let texture = create_texture(
        &context.device,
        label,
        size,
        format,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    context.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size.width * 4),
            rows_per_image: Some(size.height),
        },
        extent(size),
    );
    texture.create_view(&Default::default())
}

fn motion_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    source: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    depth: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("burns motion"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(source),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(depth),
            },
        ],
    })
}

fn image_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("burns image"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    fragment_entry: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fragment_entry),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn draw(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    group: &wgpu::BindGroup,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("burns pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        ..Default::default()
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.draw(0..3, 0..1);
}
