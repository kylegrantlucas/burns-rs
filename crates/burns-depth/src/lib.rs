//! Depth estimation for burns with one pinned model: Depth Anything V2 Small (ONNX, fp32).
//!
//! The default build is pure Rust with no filesystem, network, or ONNX Runtime access, so the
//! same [`model_manifest`], [`prepare_input`], [`postprocess_depth`], and [`source_cache_key`]
//! run natively and in WebAssembly. A browser worker can feed [`DepthInput`] to its own
//! inference backend and hand the raw output to [`postprocess_depth`].
//!
//! The `native` feature (non-wasm targets only) adds [`DepthEstimator`], which runs the model
//! with a dynamically loaded ONNX Runtime 1.27+, and [`DepthService`], which owns the platform
//! cache (`<cache>/burns/models/<sha256>.onnx`, `<cache>/burns/depth/<key>.png`), downloads and
//! verifies the model on explicit request, and reuses one inference session.
//!
//! Nothing in this crate loads ONNX Runtime or touches the network until
//! [`DepthService::generate`], [`DepthService::ensure_model`], or [`DepthEstimator::new`] is
//! called. Looking up a cached map with [`DepthService::cached`] needs neither.

mod cache_key;
mod error;
mod manifest;
mod postprocess;
mod preprocess;
mod progress;

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
mod native;

pub use cache_key::{CACHE_KEY_DOMAIN, source_cache_key};
pub use error::DepthError;
pub use manifest::{
    INPUT_NAME, MEAN, MODEL_BYTES, MODEL_ID, MODEL_LICENSE, MODEL_NAME, MODEL_REVISION,
    MODEL_SHA256, MODEL_URL, ModelManifest, OUTPUT_NAME, PATCH_SIZE, PREPROCESSING_VERSION,
    SHORT_EDGE, STD, model_manifest,
};
pub use postprocess::{GAUSSIAN_RADIUS, GAUSSIAN_SIGMA, MAX_FILTER_RADIUS, postprocess_depth};
pub use preprocess::{DepthInput, model_input_size, prepare_input};
pub use progress::DepthProgress;

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub use native::{
    DepthEstimator, DepthService, ModelSource, ORT_LIBRARY_NAME, ensure_model_file,
    resolve_ort_library, verify_model_file,
};
