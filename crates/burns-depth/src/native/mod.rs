//! Native inference, model download, and the on-disk cache. Compiled only with the `native`
//! feature on non-wasm targets.

mod download;
mod estimator;
mod runtime;
mod service;

pub use download::{ModelSource, ensure_model_file, verify_model_file};
pub use estimator::DepthEstimator;
pub use runtime::{ORT_LIBRARY_NAME, resolve_ort_library};
pub use service::DepthService;
