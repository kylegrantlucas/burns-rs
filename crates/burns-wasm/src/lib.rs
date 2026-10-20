//! WebAssembly bindings for burns.
//!
//! [`BurnsEngine`] owns one portable project, the decoded source image and raw
//! depth map, a preview target, and (after `initCanvas`) the shared core
//! renderer presenting to a WebGPU canvas. Project controls, asset decoding,
//! presets, schema, and depth pre/postprocessing are pure Rust and work without
//! WebGPU. Nothing here fetches the depth model or runs inference: a browser
//! worker does, feeding [`prepare_depth_input`] and [`postprocess_depth`].
//!
//! # Ownership
//!
//! The engine keeps its state behind an `Rc<RefCell<_>>` and every export takes
//! `&self`, so wasm-bindgen never holds a mutable borrow of the JS object.
//! Rust borrows last only for one synchronous call. `initCanvas` and `framePng`
//! return Promises whose futures own a clone of the state handle and borrow it
//! only before and after their awaits:
//!
//! - While `initCanvas` is pending, every project control, asset load, and
//!   `resizeTarget` keeps working (a previous canvas GPU, if any, stays live);
//!   the new renderer is built from the state current when the device arrives.
//!   A second `initCanvas` or a `framePng` reports that initialization is pending.
//! - `framePng` snapshots the project and takes the GPU. Project/schema reads
//!   remain available, but edits, asset loads, rendering, resizing, initialization,
//!   and another capture report busy until preview restoration completes.

#[cfg(target_arch = "wasm32")]
mod canvas;
mod error;
mod session;

use std::cell::RefCell;
use std::rc::Rc;

use burns::{GpuContext, RenderView, SourceImage};
use js_sys::{Array, Float32Array, Object, Promise, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;
use web_sys::HtmlCanvasElement;

pub use error::EngineError;
use session::{CanvasHandle, Session};

#[wasm_bindgen(typescript_custom_section)]
const TYPES: &str = r#"
/** Model input tensor from `prepareDepthInput`: NCHW, normalized. */
export interface DepthInput {
  values: Float32Array;
  shape: [1, 3, number, number];
}

/** Typed setting value; choices are option indices. */
export type SettingValue =
  | { type: "float"; value: number }
  | { type: "int"; value: number }
  | { type: "bool"; value: boolean }
  | { type: "choice"; value: number };
"#;

fn js_error(error: EngineError) -> JsValue {
    js_sys::Error::new(error.message()).into()
}

fn to_json(value: &impl serde::Serialize) -> String {
    serde_json::to_string(value).expect("descriptor types serialize to JSON")
}

fn parse_view(view: &str) -> Result<RenderView, EngineError> {
    match view {
        "motion" => Ok(RenderView::Motion),
        "source" => Ok(RenderView::Source),
        "depth" => Ok(RenderView::Depth),
        other => Err(EngineError::new(format!(
            "unknown view {other:?}; expected \"motion\", \"source\", or \"depth\""
        ))),
    }
}

fn size_json(size: burns::Size) -> String {
    serde_json::json!({ "width": size.width, "height": size.height }).to_string()
}

#[cfg(target_arch = "wasm32")]
async fn request_canvas(canvas: HtmlCanvasElement) -> error::Result<(GpuContext, CanvasHandle)> {
    canvas::request(canvas).await
}

#[cfg(not(target_arch = "wasm32"))]
async fn request_canvas(_canvas: HtmlCanvasElement) -> error::Result<(GpuContext, CanvasHandle)> {
    Err(EngineError::new(
        "initCanvas needs the wasm32 build running in a WebGPU browser",
    ))
}

/// One editing and preview session.
#[wasm_bindgen]
pub struct BurnsEngine {
    session: Rc<RefCell<Session>>,
}

impl Default for BurnsEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl BurnsEngine {
    fn with<R>(&self, call: impl FnOnce(&mut Session) -> error::Result<R>) -> Result<R, JsValue> {
        call(&mut self.session.borrow_mut()).map_err(js_error)
    }
}

#[wasm_bindgen]
impl BurnsEngine {
    /// Default project (`kenburns`), no assets, no GPU. Never touches WebGPU.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            session: Rc::new(RefCell::new(Session::new())),
        }
    }

    // ----- project controls -------------------------------------------------

    /// Canonical project JSON.
    #[wasm_bindgen(js_name = getProjectJson)]
    pub fn get_project_json(&self) -> Result<String, JsValue> {
        self.with(|s| s.project_json())
    }

    /// Strictly parses and validates a whole project, replacing the current one.
    #[wasm_bindgen(js_name = setProjectJson)]
    pub fn set_project_json(&self, json: &str) -> Result<String, JsValue> {
        self.with(|s| s.set_project_json(json))
    }

    /// Sets one active setting from text (number, `true`/`false`, or a
    /// choice's stable value or index) atomically; returns the canonical project.
    #[wasm_bindgen(js_name = setSetting)]
    pub fn set_setting(&self, id: &str, raw: &str) -> Result<String, JsValue> {
        self.with(|s| s.set_setting(id, raw))
    }

    /// Current typed value of an active setting.
    #[wasm_bindgen(js_name = getSetting, unchecked_return_type = "SettingValue")]
    pub fn get_setting(&self, id: &str) -> Result<JsValue, JsValue> {
        let value = self.with(|s| s.setting(id))?;
        js_sys::JSON::parse(&to_json(&value))
    }

    /// Replaces timing with `Timeline::from_duration(fpsNum/fpsDen, durationSeconds)`.
    #[wasm_bindgen(js_name = setTiming)]
    pub fn set_timing(
        &self,
        #[wasm_bindgen(js_name = fpsNum)] fps_num: u32,
        #[wasm_bindgen(js_name = fpsDen)] fps_den: u32,
        #[wasm_bindgen(js_name = durationSeconds)] duration_seconds: f64,
    ) -> Result<String, JsValue> {
        self.with(|s| s.set_timing(fps_num, fps_den, duration_seconds))
    }

    /// Sets the project output size. The preview target is separate: call
    /// `resizeTarget` too for an aspect-correct preview.
    #[wasm_bindgen(js_name = setOutputSize)]
    pub fn set_output_size(&self, width: u32, height: u32) -> Result<String, JsValue> {
        self.with(|s| s.set_output_size(width, height))
    }

    /// Installs a preset's camera, shake, and warp, keeping output, timing,
    /// seed, and depth adjustments.
    #[wasm_bindgen(js_name = applyPreset)]
    pub fn apply_preset(&self, name: &str) -> Result<String, JsValue> {
        self.with(|s| s.apply_preset(name))
    }

    // ----- assets -----------------------------------------------------------

    /// Decodes a PNG/JPEG/WebP still (EXIF-oriented, alpha over black) as the
    /// source and clears the depth map; uploads it once when a GPU exists. On
    /// error nothing changes. Returns the canonical project JSON.
    #[wasm_bindgen(js_name = loadImage)]
    pub fn load_image(&self, bytes: &[u8]) -> Result<String, JsValue> {
        self.with(|s| s.load_image(bytes))
    }

    /// Decodes a raw 8/16-bit grayscale depth PNG exactly as stored (no
    /// postprocessing, inversion, or reorientation). Returns the canonical project JSON.
    #[wasm_bindgen(js_name = loadDepth)]
    pub fn load_depth(&self, bytes: &[u8]) -> Result<String, JsValue> {
        self.with(|s| s.load_depth(bytes))
    }

    #[wasm_bindgen(js_name = hasSource)]
    pub fn has_source(&self) -> bool {
        self.session.borrow().source().is_some()
    }

    #[wasm_bindgen(js_name = hasDepth)]
    pub fn has_depth(&self) -> bool {
        self.session.borrow().depth().is_some()
    }

    /// `{"width":W,"height":H}` of the oriented source image.
    #[wasm_bindgen(js_name = sourceSizeJson)]
    pub fn source_size_json(&self) -> Result<String, JsValue> {
        self.with(|s| {
            s.source()
                .map(|image| size_json(image.size()))
                .ok_or_else(|| EngineError::new("no source image is loaded; call loadImage"))
        })
    }

    /// `{"width":W,"height":H}` of the depth map.
    #[wasm_bindgen(js_name = depthSizeJson)]
    pub fn depth_size_json(&self) -> Result<String, JsValue> {
        self.with(|s| {
            s.depth()
                .map(|map| size_json(map.size()))
                .ok_or_else(|| EngineError::new("no depth map is loaded"))
        })
    }

    /// The raw depth map as a 16-bit grayscale PNG (never inverted).
    #[wasm_bindgen(js_name = depthPng)]
    pub fn depth_png(&self) -> Result<Vec<u8>, JsValue> {
        self.with(|s| {
            let map = s
                .depth()
                .ok_or_else(|| EngineError::new("no depth map is loaded"))?;
            Ok(map.encode_png()?)
        })
    }

    // ----- GPU --------------------------------------------------------------

    /// Acquires a WebGPU device for `canvas` (secure contexts only), builds the
    /// renderer for the current target, and uploads the loaded assets. Rejects
    /// with an actionable error otherwise; the project and assets are kept.
    #[wasm_bindgen(js_name = initCanvas, unchecked_return_type = "Promise<void>")]
    pub fn init_canvas(&self, canvas: HtmlCanvasElement) -> Promise {
        if let Err(error) = self.with(|s| s.begin_init()) {
            return Promise::reject(&error);
        }
        let session = Rc::clone(&self.session);
        future_to_promise(async move {
            let device = request_canvas(canvas).await;
            session
                .borrow_mut()
                .finish_init(device)
                .map(|()| JsValue::UNDEFINED)
                .map_err(js_error)
        })
    }

    #[wasm_bindgen(js_name = isGpuReady)]
    pub fn is_gpu_ready(&self) -> bool {
        self.session.borrow().gpu_ready()
    }

    /// Sets the preview target (`finalQuality`: exact size with 2x
    /// supersampling; otherwise the long edge is capped at 960). With a GPU
    /// the canvas drawing buffer is resized to the output size and cleared.
    #[wasm_bindgen(js_name = resizeTarget)]
    pub fn resize_target(
        &self,
        width: u32,
        height: u32,
        #[wasm_bindgen(js_name = finalQuality)] final_quality: bool,
    ) -> Result<(), JsValue> {
        self.with(|s| s.resize_target(width, height, final_quality))
    }

    /// `{"width","height","final","outputWidth","outputHeight"}`: the
    /// requested target and the canvas/output texture size it produces.
    #[wasm_bindgen(js_name = targetJson)]
    pub fn target_json(&self) -> String {
        let target = self.session.borrow().target();
        let output = target.output_size();
        serde_json::json!({
            "width": target.size.width,
            "height": target.size.height,
            "final": target.quality == burns::RenderQuality::Final,
            "outputWidth": output.width,
            "outputHeight": output.height,
        })
        .to_string()
    }

    /// Renders `view` (`"motion"`, `"source"`, or `"depth"`) of frame `index`
    /// and presents it to the canvas before returning, so a capture stream
    /// sampled right after this call sees exactly this frame.
    #[wasm_bindgen(js_name = renderFrame)]
    pub fn render_frame(
        &self,
        index: u32,
        #[wasm_bindgen(unchecked_param_type = "\"motion\" | \"source\" | \"depth\"")] view: &str,
    ) -> Result<(), JsValue> {
        let view = parse_view(view).map_err(js_error)?;
        self.with(|s| s.render_frame(index, view))
    }

    /// Resolves to a PNG of frame `index`: always the Motion view at the exact
    /// project output size with 2x supersampling, regardless of the preview
    /// view or target, which are restored afterwards.
    #[wasm_bindgen(js_name = framePng, unchecked_return_type = "Promise<Uint8Array>")]
    pub fn frame_png(&self, index: u32) -> Promise {
        let mut capture = match self.with(|s| s.begin_capture(index)) {
            Ok(capture) => capture,
            Err(error) => return Promise::reject(&error),
        };
        let session = Rc::clone(&self.session);
        future_to_promise(async move {
            let captured = capture.run().await;
            let png = session
                .borrow_mut()
                .finish_capture(capture, captured)
                .map_err(js_error)?;
            Ok(Uint8Array::from(png.as_slice()).into())
        })
    }
}

// ----- stateless exports ------------------------------------------------------

/// The settings registry: defaults, groups, ranges, choices, and visibility.
#[wasm_bindgen(js_name = schemaJson)]
pub fn schema_json() -> String {
    to_json(&burns::settings_schema())
}

/// Built-in preset metadata (`name`, `label`, `requires_depth`, `periodic`) in display order.
#[wasm_bindgen(js_name = presetsJson)]
pub fn presets_json() -> String {
    to_json(&burns::presets())
}

/// The default project with the named preset applied, as canonical JSON.
#[wasm_bindgen(js_name = presetJson)]
pub fn preset_json(name: &str) -> Result<String, JsValue> {
    burns::preset(name)
        .and_then(|project| project.to_json())
        .map_err(|error| js_error(error.into()))
}

/// The pinned depth model: URL, size, SHA-256, tensor names, preprocessing version.
#[wasm_bindgen(js_name = modelManifestJson)]
pub fn model_manifest_json() -> String {
    to_json(&burns_depth::model_manifest())
}

/// Decodes a still exactly like `loadImage` and returns the model input tensor.
#[wasm_bindgen(js_name = prepareDepthInput, unchecked_return_type = "DepthInput")]
pub fn prepare_depth_input(
    #[wasm_bindgen(js_name = imageBytes)] image_bytes: &[u8],
) -> Result<JsValue, JsValue> {
    let input = SourceImage::decode(image_bytes)
        .map_err(EngineError::from)
        .and_then(|image| Ok(burns_depth::prepare_input(&image)?))
        .map_err(js_error)?;
    let shape = Array::new();
    for dimension in input.shape {
        shape.push(&JsValue::from(dimension as u32));
    }
    let object = Object::new();
    Reflect::set(
        &object,
        &JsValue::from_str("values"),
        &Float32Array::from(input.values.as_slice()),
    )?;
    Reflect::set(&object, &JsValue::from_str("shape"), &shape)?;
    Ok(object.into())
}

/// Postprocesses raw `width * height` model output (larger is nearer) and
/// encodes the canonical 16-bit grayscale depth PNG.
#[wasm_bindgen(js_name = postprocessDepth)]
pub fn postprocess_depth(width: u32, height: u32, values: Vec<f32>) -> Result<Vec<u8>, JsValue> {
    burns_depth::postprocess_depth(burns::Size::new(width, height), values)
        .map_err(EngineError::from)
        .and_then(|map| Ok(map.encode_png()?))
        .map_err(js_error)
}

/// Content-addressed depth cache key of a still, shared with native.
#[wasm_bindgen(js_name = depthCacheKey)]
pub fn depth_cache_key(
    #[wasm_bindgen(js_name = imageBytes)] image_bytes: &[u8],
) -> Result<String, JsValue> {
    SourceImage::decode(image_bytes)
        .map(|image| burns_depth::source_cache_key(&image))
        .map_err(|error| js_error(error.into()))
}
