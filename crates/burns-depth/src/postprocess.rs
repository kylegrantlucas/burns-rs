//! Raw model output to a canonical [`DepthMap`].

use burns::{DepthMap, Size};

use crate::DepthError;

/// Standard deviation of the separable Gaussian smoothing, in pixels.
pub const GAUSSIAN_SIGMA: f64 = 0.6;
/// Gaussian kernel radius; the kernel has `2 * GAUSSIAN_RADIUS + 1` taps.
pub const GAUSSIAN_RADIUS: usize = 3;
/// Radius of the square max filter applied after smoothing (5x5).
pub const MAX_FILTER_RADIUS: usize = 2;

/// Turns raw `width * height` model output (row-major, larger is nearer) into a depth map.
///
/// 1. Rejects zero or overflowing dimensions, a wrong sample count, and non-finite samples.
/// 2. Min-max normalizes to `0..=1`. A constant input becomes all `0.0`; no gradient is made up.
/// 3. Smooths with a separable Gaussian (σ = [`GAUSSIAN_SIGMA`], radius [`GAUSSIAN_RADIUS`]),
///    horizontal pass first.
/// 4. Applies a separable 5x5 max filter, which slightly dilates near (bright) regions so
///    foreground edges do not tear during parallax.
/// 5. Canonicalizes each sample to `round(value * 65535) / 65535`, so the returned map is
///    exactly what a 16-bit PNG round trip through [`DepthMap::encode_png`] and
///    [`DepthMap::decode_png`] yields; a freshly generated map and its cached copy are identical.
///
/// Both filters extend the map by whole-sample mirror reflection about the edge samples, which
/// are not repeated: index `-1` reads `1`, `-2` reads `2`, and `n` reads `n - 2` (NumPy
/// `reflect`, OpenCV `BORDER_REFLECT_101`). Reflection repeats with period `2 * (n - 1)` when the
/// kernel is wider than the map, and a one-sample axis reads its only sample.
pub fn postprocess_depth(size: Size, mut values: Vec<f32>) -> Result<DepthMap, DepthError> {
    let Size { width, height } = size;
    let invalid = |reason| DepthError::OutputSize {
        width,
        height,
        reason,
    };
    if width == 0 || height == 0 {
        return Err(invalid("width and height must both be at least 1"));
    }
    let (w, h) = (width as usize, height as usize);
    let expected = w
        .checked_mul(h)
        .ok_or_else(|| invalid("its sample count overflows"))?;
    if values.len() != expected {
        return Err(DepthError::OutputLength {
            width,
            height,
            expected,
            actual: values.len(),
        });
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(DepthError::NonFinite {
            index,
            // Both quotients are below width/height, so they fit in u32.
            x: (index % w) as u32,
            y: (index / w) as u32,
            value: values[index],
        });
    }

    normalize(&mut values);
    let mut scratch = vec![0.0; values.len()];
    let gaussian = gaussian_kernel();
    let smooth = |window: &[f32]| weighted(&gaussian, window);
    filter_rows(&values, &mut scratch, w, GAUSSIAN_RADIUS, smooth);
    filter_columns(&scratch, &mut values, w, h, GAUSSIAN_RADIUS, smooth);
    filter_rows(&values, &mut scratch, w, MAX_FILTER_RADIUS, max_window);
    filter_columns(&scratch, &mut values, w, h, MAX_FILTER_RADIUS, max_window);
    for value in &mut values {
        *value = canonical(*value);
    }
    Ok(DepthMap::from_f32(size, values)?)
}

/// Min-max normalization in f64, so extreme finite ranges cannot overflow.
fn normalize(values: &mut [f32]) {
    let (min, max) = values
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    if min == max {
        values.fill(0.0);
        return;
    }
    let (min, range) = (f64::from(min), f64::from(max) - f64::from(min));
    for value in values {
        *value = ((f64::from(*value) - min) / range).clamp(0.0, 1.0) as f32;
    }
}

/// Normalized Gaussian weights for offsets `-GAUSSIAN_RADIUS..=GAUSSIAN_RADIUS`.
fn gaussian_kernel() -> [f32; 2 * GAUSSIAN_RADIUS + 1] {
    let mut weights = [0.0f64; 2 * GAUSSIAN_RADIUS + 1];
    for (tap, weight) in weights.iter_mut().enumerate() {
        let offset = tap as f64 - GAUSSIAN_RADIUS as f64;
        *weight = (-offset * offset / (2.0 * GAUSSIAN_SIGMA * GAUSSIAN_SIGMA)).exp();
    }
    let total: f64 = weights.iter().sum();
    weights.map(|weight| (weight / total) as f32)
}

/// Convex combination of the window; stays within `0..=1` up to rounding, clamped exactly.
fn weighted(kernel: &[f32; 2 * GAUSSIAN_RADIUS + 1], window: &[f32]) -> f32 {
    kernel
        .iter()
        .zip(window)
        .map(|(weight, value)| weight * value)
        .sum::<f32>()
        .clamp(0.0, 1.0)
}

fn max_window(window: &[f32]) -> f32 {
    window.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// `round(value * 65535) / 65535`, identical to a 16-bit PNG encode/decode round trip.
fn canonical(value: f32) -> f32 {
    // value is within 0..=1, so the rounded product fits in u16.
    f32::from((value * 65535.0).round() as u16) / 65535.0
}

/// Whole-sample mirror index into `0..len` (see [`postprocess_depth`]).
fn reflect(index: isize, len: usize) -> usize {
    if len == 1 {
        return 0;
    }
    let period = 2 * (len as isize - 1);
    let folded = index.rem_euclid(period);
    if folded < len as isize {
        folded as usize
    } else {
        (period - folded) as usize
    }
}

/// Applies a 1D filter over the reflected `2 * radius + 1` window along each row.
fn filter_rows(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    radius: usize,
    filter: impl Fn(&[f32]) -> f32,
) {
    let mut storage = [0.0; 2 * GAUSSIAN_RADIUS + 1];
    let window = &mut storage[..2 * radius + 1];
    for (row_in, row_out) in input
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
    {
        for (x, out) in row_out.iter_mut().enumerate() {
            for (tap, slot) in window.iter_mut().enumerate() {
                let source = x as isize + tap as isize - radius as isize;
                *slot = row_in[reflect(source, width)];
            }
            *out = filter(window);
        }
    }
}

/// Applies a 1D filter over the reflected `2 * radius + 1` window along each column.
fn filter_columns(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    filter: impl Fn(&[f32]) -> f32,
) {
    let mut storage = [0.0; 2 * GAUSSIAN_RADIUS + 1];
    let window = &mut storage[..2 * radius + 1];
    for y in 0..height {
        for x in 0..width {
            for (tap, slot) in window.iter_mut().enumerate() {
                let source = y as isize + tap as isize - radius as isize;
                *slot = input[reflect(source, height) * width + x];
            }
            output[y * width + x] = filter(window);
        }
    }
}
