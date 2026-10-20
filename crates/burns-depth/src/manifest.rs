//! The single pinned depth model and its tensor contract.

use serde::{Deserialize, Serialize};

/// Hugging Face repository of the pinned model.
pub const MODEL_ID: &str = "onnx-community/depth-anything-v2-small";
/// Human-readable model name.
pub const MODEL_NAME: &str = "Depth Anything V2 Small";
/// Pinned repository commit.
pub const MODEL_REVISION: &str = "4472b7362082ad9968fee890ca0f1e5aca36b93d";
/// Immutable download URL of the single-file fp32 ONNX graph at [`MODEL_REVISION`].
pub const MODEL_URL: &str = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/4472b7362082ad9968fee890ca0f1e5aca36b93d/onnx/model.onnx";
/// Exact size of the model file in bytes.
pub const MODEL_BYTES: u64 = 99_060_839;
/// Lowercase hex SHA-256 of the model file.
pub const MODEL_SHA256: &str = "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c";
/// SPDX license of the model weights.
pub const MODEL_LICENSE: &str = "Apache-2.0";
/// Float32 input `[batch, 3, height, width]`, batch 1.
pub const INPUT_NAME: &str = "pixel_values";
/// Float32 output `[batch, height, width]` (relative inverse depth; larger is nearer).
pub const OUTPUT_NAME: &str = "predicted_depth";
/// Version of the whole input-to-map pipeline: [`prepare_input`](crate::prepare_input) and
/// [`postprocess_depth`](crate::postprocess_depth). Any change to either must bump it, which
/// also invalidates every cached map through [`source_cache_key`](crate::source_cache_key).
pub const PREPROCESSING_VERSION: u32 = 1;

/// The model's patch size; input dimensions are multiples of it.
pub const PATCH_SIZE: u32 = 14;
/// Target length of the shorter input edge before rounding to [`PATCH_SIZE`].
pub const SHORT_EDGE: u32 = 518;
/// Per-channel RGB mean subtracted after scaling samples to `0..=1`.
pub const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// Per-channel RGB standard deviation divided out after subtracting [`MEAN`].
pub const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Serializable description of the pinned model, shared by native and browser front ends.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelManifest {
    pub id: String,
    pub name: String,
    pub revision: String,
    pub url: String,
    pub bytes: u64,
    pub sha256: String,
    pub license: String,
    pub input_name: String,
    pub output_name: String,
    pub preprocessing_version: u32,
}

/// The pinned model. There is no alternative or fallback model.
pub fn model_manifest() -> ModelManifest {
    ModelManifest {
        id: MODEL_ID.to_owned(),
        name: MODEL_NAME.to_owned(),
        revision: MODEL_REVISION.to_owned(),
        url: MODEL_URL.to_owned(),
        bytes: MODEL_BYTES,
        sha256: MODEL_SHA256.to_owned(),
        license: MODEL_LICENSE.to_owned(),
        input_name: INPUT_NAME.to_owned(),
        output_name: OUTPUT_NAME.to_owned(),
        preprocessing_version: PREPROCESSING_VERSION,
    }
}
