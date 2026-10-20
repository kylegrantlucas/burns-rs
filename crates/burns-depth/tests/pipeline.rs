use burns::{DepthMap, Size, SourceImage};
use burns_depth::{
    DepthError, MEAN, STD, model_input_size, postprocess_depth, prepare_input, source_cache_key,
};

const LEVEL: f32 = 1.0 / 65535.0;

fn size(width: u32, height: u32) -> Size {
    Size { width, height }
}

fn solid(width: u32, height: u32, rgb: impl Fn(u32, u32) -> [u8; 3]) -> SourceImage {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let [r, g, b] = rgb(x, y);
            pixels.extend_from_slice(&[r, g, b, 255]);
        }
    }
    SourceImage::from_rgba8(size(width, height), pixels).unwrap()
}

fn normalized(sample: f32, channel: usize) -> f32 {
    (sample / 255.0 - MEAN[channel]) / STD[channel]
}

/// Independent f64 reference for the σ = 0.6, radius 3 kernel.
fn gaussian() -> [f64; 7] {
    let raw: Vec<f64> = (-3..=3)
        .map(|k: i32| (-f64::from(k * k) / (2.0 * 0.36)).exp())
        .collect();
    let total: f64 = raw.iter().sum();
    std::array::from_fn(|i| raw[i] / total)
}

fn canonical(value: f64) -> f32 {
    ((value * 65535.0).round() / 65535.0) as f32
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() <= tolerance, "index {index}: {a} vs {e}");
    }
}

#[test]
fn input_size_keeps_aspect_and_rounds_to_nearest_patch() {
    assert_eq!(model_input_size(size(3840, 2160)).unwrap(), size(924, 518));
    assert_eq!(model_input_size(size(1920, 1080)).unwrap(), size(924, 518));
    assert_eq!(model_input_size(size(2160, 3840)).unwrap(), size(518, 924));
    assert_eq!(model_input_size(size(1000, 1000)).unwrap(), size(518, 518));
    // 700 * 1.036 = 725.2 -> 51.8 patches -> 52 (nearest, not floor).
    assert_eq!(model_input_size(size(700, 500)).unwrap(), size(728, 518));
    // Small sources are scaled up.
    assert_eq!(model_input_size(size(1, 1)).unwrap(), size(518, 518));
    assert!(matches!(
        model_input_size(size(0, 10)),
        Err(DepthError::InputSize { .. })
    ));
}

#[test]
fn prepare_input_is_nchw_normalized_rgb_at_16_9() {
    // Left half red, right half blue, with a green stripe on the bottom rows.
    let image = solid(1920, 1080, |x, y| {
        if y >= 1000 {
            [0, 255, 0]
        } else if x < 960 {
            [255, 0, 0]
        } else {
            [0, 0, 255]
        }
    });
    let input = prepare_input(&image).unwrap();
    assert_eq!(input.shape, [1, 3, 518, 924]);
    assert_eq!(input.size(), size(924, 518));
    let plane = 924 * 518;
    assert_eq!(input.values.len(), 3 * plane);
    let at = |channel: usize, x: usize, y: usize| input.values[channel * plane + y * 924 + x];
    let tolerance = 1e-5;
    let red_pixel = [(0, 0), (100, 200), (400, 50)];
    for (x, y) in red_pixel {
        assert!((at(0, x, y) - normalized(255.0, 0)).abs() < tolerance);
        assert!((at(1, x, y) - normalized(0.0, 1)).abs() < tolerance);
        assert!((at(2, x, y) - normalized(0.0, 2)).abs() < tolerance);
    }
    for (x, y) in [(923, 0), (700, 300)] {
        assert!((at(0, x, y) - normalized(0.0, 0)).abs() < tolerance);
        assert!((at(1, x, y) - normalized(0.0, 1)).abs() < tolerance);
        assert!((at(2, x, y) - normalized(255.0, 2)).abs() < tolerance);
    }
    // The green stripe stays at the bottom: rows are top-first and not flipped.
    assert!((at(1, 10, 517) - normalized(255.0, 1)).abs() < tolerance);
    assert!((at(0, 10, 517) - normalized(0.0, 0)).abs() < tolerance);
}

#[test]
fn prepare_input_normalizes_each_channel_with_its_own_mean_and_std() {
    let image = solid(1600, 900, |_, _| [128, 64, 32]);
    let input = prepare_input(&image).unwrap();
    let plane = input.shape[2] * input.shape[3];
    for (channel, sample) in [128.0, 64.0, 32.0].into_iter().enumerate() {
        let expected = normalized(sample, channel);
        let values = &input.values[channel * plane..(channel + 1) * plane];
        assert!(
            values.iter().all(|v| (v - expected).abs() < 1e-5),
            "channel {channel}"
        );
    }
}

#[test]
fn postprocess_rejects_bad_dimensions_lengths_and_non_finite_values() {
    assert!(matches!(
        postprocess_depth(size(0, 2), vec![]),
        Err(DepthError::OutputSize { .. })
    ));
    assert!(matches!(
        postprocess_depth(size(2, 2), vec![0.0; 3]),
        Err(DepthError::OutputLength {
            expected: 4,
            actual: 3,
            ..
        })
    ));
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err = postprocess_depth(size(3, 2), vec![0.0, 1.0, 2.0, 3.0, bad, 5.0]).unwrap_err();
        assert!(
            matches!(
                err,
                DepthError::NonFinite {
                    index: 4,
                    x: 1,
                    y: 1,
                    ..
                }
            ),
            "{err:?}"
        );
    }
}

#[test]
fn constant_output_becomes_all_zero() {
    let map = postprocess_depth(size(7, 5), vec![5.0; 35]).unwrap();
    assert!(map.values().iter().all(|&v| v == 0.0));
}

#[test]
fn min_max_normalization_spans_zero_to_one_and_ignores_affine_scale() {
    // A step from -3 to 7 midway across a 24-wide map.
    let raw: Vec<f32> = (0..24 * 4)
        .map(|i| if i % 24 < 12 { -3.0 } else { 7.0 })
        .collect();
    let map = postprocess_depth(size(24, 4), raw.clone()).unwrap();
    for row in map.values().chunks_exact(24) {
        assert_eq!(row[0], 0.0);
        assert_eq!(row[23], 1.0);
        assert!(row.windows(2).all(|pair| pair[0] <= pair[1]), "{row:?}");
    }
    let scaled = raw.iter().map(|v| v * 1000.0 + 42.0).collect();
    let rescaled = postprocess_depth(size(24, 4), scaled).unwrap();
    assert_close(rescaled.values(), map.values(), LEVEL);
}

/// Impulse at index 0 along a 9-sample axis, expected after smoothing and the 5-wide max with
/// mirror reflection that does not repeat the edge sample.
fn expected_edge_impulse() -> Vec<f32> {
    let k = gaussian();
    let smoothed = [k[3], k[2], k[1], k[0], 0.0, 0.0, 0.0, 0.0, 0.0];
    (0..9)
        .map(|i: isize| {
            let max = (i - 2..=i + 2)
                .map(|j| smoothed[j.unsigned_abs().min(16 - j.unsigned_abs())])
                .fold(0.0, f64::max);
            canonical(max)
        })
        .collect()
}

#[test]
fn edge_impulse_reflects_without_repeating_the_edge_sample() {
    let expected = expected_edge_impulse();
    // Mirror reflection gives exactly the centre weight at the edge; repeating the edge sample
    // would add the neighbour weight too.
    let k = gaussian();
    assert!((f64::from(expected[0]) - k[3]).abs() < 1e-4);

    let mut row = vec![0.0; 9];
    row[0] = 1.0;
    let horizontal = postprocess_depth(size(9, 1), row.clone()).unwrap();
    assert_close(horizontal.values(), &expected, LEVEL);

    // The same on the right edge and along columns.
    row.reverse();
    let mut reversed = expected.clone();
    reversed.reverse();
    let right = postprocess_depth(size(9, 1), row.clone()).unwrap();
    assert_close(right.values(), &reversed, LEVEL);
    let bottom = postprocess_depth(size(1, 9), row).unwrap();
    assert_close(bottom.values(), &reversed, LEVEL);
}

#[test]
fn kernels_wider_than_the_map_keep_reflecting() {
    // Two samples: reflection alternates 1, 0, 1, [0, 1], 0, 1 around index 0.
    let k = gaussian();
    let left = 2.0 * (k[2] + k[0]);
    let right = k[3] + 2.0 * k[1];
    let expected = canonical(left.max(right));
    let map = postprocess_depth(size(2, 1), vec![0.0, 1.0]).unwrap();
    assert_close(map.values(), &[expected, expected], LEVEL);
}

#[test]
fn generated_map_survives_16_bit_png_round_trip_unchanged() {
    let (width, height) = (64u32, 48u32);
    let raw: Vec<f32> = (0..width * height)
        .map(|i| {
            let (x, y) = ((i % width) as f32, (i / width) as f32);
            (x * 0.37).sin() * 3.0 + y * 0.11 + (i.wrapping_mul(2_654_435_761) % 1000) as f32 * 1e-3
        })
        .collect();
    let map = postprocess_depth(size(width, height), raw).unwrap();
    for &value in map.values() {
        let scaled = value * 65535.0;
        assert_eq!(scaled, scaled.round(), "{value} is not a 16-bit level");
    }
    let decoded = DepthMap::decode_png(&map.encode_png().unwrap()).unwrap();
    assert_eq!(decoded, map);
}

#[test]
fn cache_key_follows_the_documented_byte_layout() {
    let image =
        SourceImage::from_rgba8(size(2, 1), vec![10, 20, 30, 255, 40, 50, 60, 255]).unwrap();
    // sha256 of: "burns-depth source cache key\0", 1u32 LE, model sha bytes, 2u32 LE, 1u32 LE,
    // pixels — computed independently with shasum.
    assert_eq!(
        source_cache_key(&image),
        "e771e370baf8c1438dde6fd74f3fcfd172a134226b8a6040b5cc3f80107fe048"
    );
}

#[test]
fn cache_key_distinguishes_pixels_and_dimensions() {
    let pixels = vec![10, 20, 30, 255, 40, 50, 60, 255];
    let wide = SourceImage::from_rgba8(size(2, 1), pixels.clone()).unwrap();
    let tall = SourceImage::from_rgba8(size(1, 2), pixels.clone()).unwrap();
    let mut changed = pixels;
    changed[0] = 11;
    let edited = SourceImage::from_rgba8(size(2, 1), changed).unwrap();
    let key = source_cache_key(&wide);
    assert_ne!(key, source_cache_key(&tall));
    assert_ne!(key, source_cache_key(&edited));
}
