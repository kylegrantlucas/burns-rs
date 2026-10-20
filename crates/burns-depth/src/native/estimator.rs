//! One reusable ONNX Runtime session for the pinned model.

use std::path::Path;

use burns::{DepthMap, SourceImage};
use ort::{
    session::Session,
    value::{Outlet, Tensor, TensorElementType, ValueType},
};

use crate::{DepthError, INPUT_NAME, OUTPUT_NAME, postprocess_depth, prepare_input};

use super::runtime;

/// Runs the pinned depth model on the CPU execution provider.
///
/// Construction loads ONNX Runtime (once per process), creates a session, and checks the
/// model's tensor contract; [`estimate`](Self::estimate) reuses that session. A single ONNX
/// Runtime call cannot be interrupted once started.
pub struct DepthEstimator {
    session: Session,
}

impl std::fmt::Debug for DepthEstimator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DepthEstimator").finish_non_exhaustive()
    }
}

impl DepthEstimator {
    /// Loads ONNX Runtime from `ort_library` and the model from `model_path`.
    ///
    /// The model must have exactly one float32 input `pixel_values` shaped
    /// `[batch, 3, height, width]` and one float32 output `predicted_depth` shaped
    /// `[batch, height, width]`, with batch 1 or dynamic and dynamic spatial dimensions.
    /// `model_path` is not hashed here; [`DepthService`](crate::DepthService) verifies it.
    pub fn new(model_path: &Path, ort_library: &Path) -> Result<Self, DepthError> {
        let environment = runtime::environment(ort_library)?;
        let load_error = |err: ort::Error| DepthError::ModelLoad {
            path: model_path.to_path_buf(),
            message: err.to_string(),
        };
        let session = Session::builder(&environment)
            .map_err(load_error)?
            .commit_from_file(model_path)
            .map_err(load_error)?;
        check_contract(&session, model_path)?;
        Ok(Self { session })
    }

    /// Estimates a canonical depth map at the model input resolution
    /// ([`model_input_size`](crate::model_input_size)) for `image`.
    pub fn estimate(&mut self, image: &SourceImage) -> Result<DepthMap, DepthError> {
        let input = prepare_input(image)?;
        let size = input.size();
        let [_, _, height, width] = input.shape;
        let inference = |err: ort::Error| DepthError::Inference {
            message: err.to_string(),
        };
        let tensor = Tensor::from_array((input.shape, input.values)).map_err(inference)?;
        let outputs = self
            .session
            .run(ort::inputs![INPUT_NAME => tensor])
            .map_err(inference)?;
        let output = outputs
            .get(OUTPUT_NAME)
            .ok_or_else(|| DepthError::ModelContract {
                message: format!("the run produced no `{OUTPUT_NAME}` output"),
            })?;
        let (shape, values) =
            output
                .try_extract_tensor::<f32>()
                .map_err(|err| DepthError::ModelContract {
                    message: format!("`{OUTPUT_NAME}` is not a float32 tensor: {err}"),
                })?;
        // Input dimensions are multiples of the patch size, so 14 * floor(dim / 14) == dim.
        let expected = [1, height as i64, width as i64];
        if **shape != expected {
            return Err(DepthError::ModelContract {
                message: format!(
                    "`{OUTPUT_NAME}` has shape {:?} for a {width}x{height} input, expected {expected:?}",
                    &**shape
                ),
            });
        }
        postprocess_depth(size, values.to_vec())
    }
}

fn check_contract(session: &Session, model_path: &Path) -> Result<(), DepthError> {
    let path = model_path.display();
    let contract = |message: String| DepthError::ModelContract {
        message: format!("`{path}`: {message}"),
    };
    let inputs = session.inputs();
    let outputs = session.outputs();
    let [input] = inputs else {
        return Err(contract(format!(
            "expected one input `{INPUT_NAME}`, found {:?}",
            names(inputs)
        )));
    };
    let [output] = outputs else {
        return Err(contract(format!(
            "expected one output `{OUTPUT_NAME}`, found {:?}",
            names(outputs)
        )));
    };
    let input_dims = float_tensor(input, INPUT_NAME, 4).map_err(contract)?;
    let batch_ok = matches!(input_dims[0], -1 | 1);
    if !batch_ok || input_dims[1] != 3 || input_dims[2] != -1 || input_dims[3] != -1 {
        return Err(contract(format!(
            "`{INPUT_NAME}` has shape {input_dims:?}, expected [batch, 3, height, width] with \
             batch 1 or dynamic and dynamic height and width"
        )));
    }
    let output_dims = float_tensor(output, OUTPUT_NAME, 3).map_err(contract)?;
    if !matches!(output_dims[0], -1 | 1) {
        return Err(contract(format!(
            "`{OUTPUT_NAME}` has shape {output_dims:?}, expected batch 1 or dynamic"
        )));
    }
    Ok(())
}

fn names(outlets: &[Outlet]) -> Vec<&str> {
    outlets.iter().map(Outlet::name).collect()
}

/// Checks name, float32 element type, and rank; returns the declared dimensions (`-1` dynamic).
fn float_tensor(outlet: &Outlet, name: &str, rank: usize) -> Result<Vec<i64>, String> {
    if outlet.name() != name {
        return Err(format!(
            "expected tensor `{name}`, found `{}`",
            outlet.name()
        ));
    }
    match outlet.dtype() {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape,
            ..
        } if shape.len() == rank => Ok(shape.to_vec()),
        other => Err(format!(
            "`{name}` is {other}, expected a rank-{rank} float32 tensor"
        )),
    }
}
