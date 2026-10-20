use serde::{Deserialize, Serialize};

/// Progress reported while generating a depth map.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum DepthProgress {
    /// A cached map for this source was found; no model, runtime, or inference was used.
    CacheHit,
    /// Verifying the length and SHA-256 of an already downloaded model file.
    Verifying,
    /// Model download progress in bytes.
    Downloading { received: u64, total: u64 },
    /// Loading ONNX Runtime and creating the inference session.
    Loading,
    /// Running the model.
    Inferring,
    /// Writing the generated map to the cache.
    Saving,
    /// The map was generated and is usable, but could not be written to the cache.
    PersistenceWarning { message: String },
}
