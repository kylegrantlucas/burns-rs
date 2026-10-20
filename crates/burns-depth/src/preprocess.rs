//! Source image to model input tensor.

use burns::{Size, SourceImage};
use image::{ImageBuffer, Rgba, imageops};

use crate::{DepthError, MEAN, PATCH_SIZE, SHORT_EDGE, STD};

/// A model input tensor in NCHW layout.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthInput {
    /// `shape[0] * shape[1] * shape[2] * shape[3]` normalized samples: the red plane, then green,
    /// then blue, each row-major with the top row first.
    pub values: Vec<f32>,
    /// `[1, 3, height, width]`.
    pub shape: [usize; 4],
}

impl DepthInput {
    /// Spatial size `shape[3] x shape[2]`, which is also the size of the model output.
    pub fn size(&self) -> Size {
        // Built from u32 dimensions in `prepare_input`.
        Size {
            width: self.shape[3] as u32,
            height: self.shape[2] as u32,
        }
    }
}

/// The model input size for a source of `source` size.
///
/// The image is scaled so its shorter edge is [`SHORT_EDGE`] (aspect ratio preserved, never
/// squashed to a square), then each scaled dimension is rounded to the nearest multiple of
/// [`PATCH_SIZE`] (halves round up, minimum one patch). A 16:9 source becomes 924x518.
pub fn model_input_size(source: Size) -> Result<Size, DepthError> {
    let Size { width, height } = source;
    if width == 0 || height == 0 {
        return Err(DepthError::InputSize {
            width,
            height,
            reason: "width and height must both be at least 1",
        });
    }
    let scale = f64::from(SHORT_EDGE) / f64::from(width.min(height));
    let patch = f64::from(PATCH_SIZE);
    let fit = |dim: u32| -> Option<u32> {
        let patches = (f64::from(dim) * scale / patch).round().max(1.0);
        let pixels = patches * patch;
        (pixels <= f64::from(u32::MAX)).then_some(pixels as u32)
    };
    match (fit(width), fit(height)) {
        (Some(width), Some(height)) => Ok(Size { width, height }),
        _ => Err(DepthError::InputSize {
            width,
            height,
            reason: "the aspect-preserving model dimensions exceed u32; use a less extreme aspect ratio",
        }),
    }
}

/// Resizes `image` to [`model_input_size`] with bicubic (Catmull-Rom) filtering and converts it
/// to a normalized NCHW tensor: `(rgb / 255 - MEAN) / STD` per channel.
///
/// The source is expected opaque and already oriented, as [`SourceImage`] guarantees; alpha is
/// ignored.
pub fn prepare_input(image: &SourceImage) -> Result<DepthInput, DepthError> {
    let source = image.size();
    let target = model_input_size(source)?;
    let invalid = || DepthError::InputSize {
        width: source.width,
        height: source.height,
        reason: "the aspect-preserving input tensor exceeds addressable memory",
    };
    let plane = target.pixel_count().map_err(|_| invalid())?;
    let elements = plane
        .checked_mul(3)
        .filter(|count| *count <= isize::MAX as usize / size_of::<f32>())
        .ok_or_else(invalid)?;
    let view =
        ImageBuffer::<Rgba<u8>, &[u8]>::from_raw(source.width, source.height, image.pixels())
            // SourceImage guarantees width * height * 4 bytes.
            .expect("SourceImage pixel buffer matches its size");
    let resized = imageops::resize(
        &view,
        target.width,
        target.height,
        imageops::FilterType::CatmullRom,
    );
    let (width, height) = (target.width as usize, target.height as usize);
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(|_| DepthError::InputSize {
        width: source.width,
        height: source.height,
        reason: "cannot allocate the aspect-preserving depth tensor; use a less extreme aspect ratio",
    })?;
    values.resize(elements, 0.0);
    let (red, rest) = values.split_at_mut(plane);
    let (green, blue) = rest.split_at_mut(plane);
    let normalize = |sample: u8, channel: usize| -> f32 {
        (f32::from(sample) / 255.0 - MEAN[channel]) / STD[channel]
    };
    for (index, pixel) in resized.pixels().enumerate() {
        let [r, g, b, _] = pixel.0;
        red[index] = normalize(r, 0);
        green[index] = normalize(g, 1);
        blue[index] = normalize(b, 2);
    }
    Ok(DepthInput {
        values,
        shape: [1, 3, height, width],
    })
}
