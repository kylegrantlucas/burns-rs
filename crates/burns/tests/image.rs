use std::io::Cursor;

use ::image::codecs::jpeg::JpegEncoder;
use ::image::codecs::png::PngEncoder;
use ::image::codecs::webp::WebPEncoder;
use ::image::{ExtendedColorType, ImageEncoder};
use burns::Size;
use burns::image::{CoverMap, DepthMap, ImageError, SourceImage};

fn size(width: u32, height: u32) -> Size {
    Size { width, height }
}

/// Big-endian TIFF/EXIF block holding only the orientation tag.
fn exif_orientation(orientation: u16) -> Vec<u8> {
    let mut exif = b"MM\0\x2a".to_vec();
    exif.extend_from_slice(&8u32.to_be_bytes()); // first IFD offset
    exif.extend_from_slice(&1u16.to_be_bytes()); // entry count
    exif.extend_from_slice(&0x0112u16.to_be_bytes()); // Orientation tag
    exif.extend_from_slice(&3u16.to_be_bytes()); // SHORT
    exif.extend_from_slice(&1u32.to_be_bytes()); // count
    exif.extend_from_slice(&orientation.to_be_bytes());
    exif.extend_from_slice(&[0, 0]); // value padding
    exif.extend_from_slice(&0u32.to_be_bytes()); // no next IFD
    exif
}

fn encode(
    encoder: impl ImageEncoder,
    pixels: &[u8],
    size: Size,
    color: ExtendedColorType,
    exif: Option<u16>,
) {
    let mut encoder = encoder;
    if let Some(orientation) = exif {
        encoder
            .set_exif_metadata(exif_orientation(orientation))
            .unwrap();
    }
    encoder
        .write_image(pixels, size.width, size.height, color)
        .unwrap();
}

fn png_rgba(pixels: &[u8], size: Size, exif: Option<u16>) -> Vec<u8> {
    let mut out = Vec::new();
    encode(
        PngEncoder::new(&mut out),
        pixels,
        size,
        ExtendedColorType::Rgba8,
        exif,
    );
    out
}

fn jpeg_rgb(pixels: &[u8], size: Size, exif: Option<u16>) -> Vec<u8> {
    let mut out = Vec::new();
    encode(
        JpegEncoder::new_with_quality(&mut out, 100),
        pixels,
        size,
        ExtendedColorType::Rgb8,
        exif,
    );
    out
}

fn webp_rgba(pixels: &[u8], size: Size, exif: Option<u16>) -> Vec<u8> {
    let mut out = Vec::new();
    encode(
        WebPEncoder::new_lossless(&mut out),
        pixels,
        size,
        ExtendedColorType::Rgba8,
        exif,
    );
    out
}

/// Raw PNG through the `png` crate so color type, bit depth, and chunks are exact.
fn png_raw(
    size: Size,
    color: png::ColorType,
    depth: png::BitDepth,
    data: &[u8],
    configure: impl FnOnce(&mut png::Encoder<'_, &mut Vec<u8>>),
) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, size.width, size.height);
        encoder.set_color(color);
        encoder.set_depth(depth);
        configure(&mut encoder);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(data).unwrap();
        writer.finish().unwrap();
    }
    out
}

fn gray16(size: Size, samples: &[u16]) -> Vec<u8> {
    let data: Vec<u8> = samples.iter().flat_map(|s| s.to_be_bytes()).collect();
    png_raw(
        size,
        png::ColorType::Grayscale,
        png::BitDepth::Sixteen,
        &data,
        |_| {},
    )
}

/// A distinct, asymmetric opaque color for every pixel of a `w`x`h` image.
fn labelled_rgba(size: Size) -> Vec<u8> {
    let mut pixels = Vec::new();
    for y in 0..size.height {
        for x in 0..size.width {
            pixels.extend_from_slice(&[(x * 40 + 10) as u8, (y * 60 + 20) as u8, 7, 255]);
        }
    }
    pixels
}

fn pixel(image: &SourceImage, x: u32, y: u32) -> [u8; 4] {
    let start = ((y * image.size().width + x) * 4) as usize;
    image.pixels()[start..start + 4].try_into().unwrap()
}

/// Source coordinate shown at output `(x, y)` for an EXIF orientation, from the EXIF spec.
fn exif_source_coord(orientation: u16, w: u32, h: u32, x: u32, y: u32) -> (u32, u32) {
    match orientation {
        1 => (x, y),
        2 => (w - 1 - x, y),
        3 => (w - 1 - x, h - 1 - y),
        4 => (x, h - 1 - y),
        5 => (y, x),
        6 => (y, h - 1 - x),
        7 => (w - 1 - y, h - 1 - x),
        8 => (w - 1 - y, x),
        _ => unreachable!(),
    }
}

#[test]
fn every_exif_orientation_is_applied_to_png_and_webp() {
    let src = size(3, 2);
    let pixels = labelled_rgba(src);
    for orientation in 1..=8u16 {
        let swapped = orientation >= 5;
        let expected_size = if swapped { size(2, 3) } else { src };
        for (format, bytes) in [
            ("png", png_rgba(&pixels, src, Some(orientation))),
            ("webp", webp_rgba(&pixels, src, Some(orientation))),
        ] {
            let image = SourceImage::decode(&bytes).unwrap();
            assert_eq!(
                image.size(),
                expected_size,
                "{format} orientation {orientation}"
            );
            for y in 0..expected_size.height {
                for x in 0..expected_size.width {
                    let (sx, sy) = exif_source_coord(orientation, 3, 2, x, y);
                    let start = ((sy * 3 + sx) * 4) as usize;
                    assert_eq!(
                        pixel(&image, x, y),
                        pixels[start..start + 4],
                        "{format} orientation {orientation} at ({x}, {y})"
                    );
                }
            }
        }
    }
}

#[test]
fn jpeg_exif_rotation_moves_quadrants() {
    // 32x16 with four flat 16x8 quadrants: red, green / blue, white.
    let src = size(32, 16);
    let colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]];
    let mut pixels = Vec::new();
    for y in 0..16 {
        for x in 0..32 {
            pixels.extend_from_slice(&colors[(y / 8) * 2 + x / 16]);
        }
    }
    let near = |got: [u8; 4], want: [u8; 3]| {
        got[3] == 255 && got[..3].iter().zip(want).all(|(&g, w)| g.abs_diff(w) <= 24)
    };

    let plain = SourceImage::decode(&jpeg_rgb(&pixels, src, None)).unwrap();
    assert_eq!(plain.size(), src);
    assert!(near(pixel(&plain, 4, 4), colors[0]));
    assert!(near(pixel(&plain, 28, 12), colors[3]));

    // Orientation 6: rotate 90 degrees clockwise; bottom-left (blue) becomes top-left.
    let cw = SourceImage::decode(&jpeg_rgb(&pixels, src, Some(6))).unwrap();
    assert_eq!(cw.size(), size(16, 32));
    assert!(near(pixel(&cw, 4, 4), colors[2]), "{:?}", pixel(&cw, 4, 4));
    assert!(near(pixel(&cw, 12, 4), colors[0]));
    assert!(near(pixel(&cw, 4, 28), colors[3]));
    assert!(near(pixel(&cw, 12, 28), colors[1]));

    // Orientation 8: rotate 270 degrees clockwise; top-right (green) becomes top-left.
    let ccw = SourceImage::decode(&jpeg_rgb(&pixels, src, Some(8))).unwrap();
    assert_eq!(ccw.size(), size(16, 32));
    assert!(near(pixel(&ccw, 4, 4), colors[1]));
    assert!(near(pixel(&ccw, 12, 4), colors[3]));
    assert!(near(pixel(&ccw, 4, 28), colors[0]));
    assert!(near(pixel(&ccw, 12, 28), colors[2]));
}

#[test]
fn alpha_is_composited_over_black_with_rounding() {
    let src = size(5, 1);
    #[rustfmt::skip]
    let pixels = [
        200, 100, 50, 128, // 100.39, 50.20, 25.10
        3, 1, 255, 128,    // 1.506 rounds up to 2; 0.502 rounds up to 1; 128
        90, 80, 70, 0,     // fully transparent becomes black
        12, 34, 56, 255,   // opaque is untouched
        255, 255, 255, 1,  // 1
    ];
    let expected = [
        100, 50, 25, 255, 2, 1, 128, 255, 0, 0, 0, 255, 12, 34, 56, 255, 1, 1, 1, 255,
    ];
    let decoded = SourceImage::decode(&png_rgba(&pixels, src, None)).unwrap();
    assert_eq!(decoded.pixels(), expected);
    let webp = SourceImage::decode(&webp_rgba(&pixels, src, None)).unwrap();
    assert_eq!(webp.pixels(), expected);
    let raw = SourceImage::from_rgba8(src, pixels.to_vec()).unwrap();
    assert_eq!(raw.pixels(), expected);
}

#[test]
fn sixteen_bit_and_gray_sources_keep_resolution() {
    let src = size(301, 7);
    let gray: Vec<u8> = (0..src.width * src.height)
        .map(|i| (i % 251) as u8)
        .collect();
    let bytes = png_raw(
        src,
        png::ColorType::Grayscale,
        png::BitDepth::Eight,
        &gray,
        |_| {},
    );
    let image = SourceImage::decode(&bytes).unwrap();
    assert_eq!(image.size(), src);
    assert_eq!(image.pixels().len(), 301 * 7 * 4);
    let v = gray[300 + 301 * 6];
    assert_eq!(pixel(&image, 300, 6), [v, v, v, 255]);

    let rgba16: Vec<u8> = [0x12u8, 0x34, 0xff, 0xff, 0x00, 0x00, 0xff, 0xff]
        .into_iter()
        .cycle()
        .take(2 * 3 * 8)
        .collect();
    let bytes = png_raw(
        size(2, 3),
        png::ColorType::Rgba,
        png::BitDepth::Sixteen,
        &rgba16,
        |_| {},
    );
    let image = SourceImage::decode(&bytes).unwrap();
    assert_eq!(image.size(), size(2, 3));
    assert_eq!(pixel(&image, 1, 2), [0x12, 0xff, 0x00, 0xff]);
}

#[test]
fn source_rejects_invalid_buffers_formats_and_animation() {
    assert!(matches!(
        SourceImage::from_rgba8(size(0, 4), Vec::new()),
        Err(ImageError::ZeroSize { width: 0, .. })
    ));
    assert!(matches!(
        SourceImage::from_rgba8(size(2, 2), vec![0; 15]),
        Err(ImageError::BufferLength {
            expected: 16,
            actual: 15,
            ..
        })
    ));
    if usize::BITS == 64 {
        assert!(matches!(
            SourceImage::from_rgba8(size(u32::MAX, u32::MAX), Vec::new()),
            Err(ImageError::SizeOverflow { .. })
        ));
    }
    assert!(matches!(
        SourceImage::decode(b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"),
        Err(ImageError::UnsupportedFormat { .. })
    ));
    assert!(matches!(
        SourceImage::decode(b"not an image at all"),
        Err(ImageError::UnrecognizedFormat)
    ));
    let mut truncated = png_rgba(&labelled_rgba(size(3, 2)), size(3, 2), None);
    truncated.truncate(40);
    assert!(matches!(
        SourceImage::decode(&truncated),
        Err(ImageError::SourceDecode { format: "PNG", .. })
    ));

    let apng = {
        let data = [9u8; 2 * 2 * 4];
        let mut out = Vec::new();
        let mut encoder = png::Encoder::new(&mut out, 2, 2);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_animated(2, 0).unwrap();
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&data).unwrap();
        writer.write_image_data(&data).unwrap();
        writer.finish().unwrap();
        out
    };
    assert!(matches!(
        SourceImage::decode(&apng),
        Err(ImageError::AnimatedSource { format: "PNG" })
    ));
    assert!(matches!(
        SourceImage::decode(&animated_webp()),
        Err(ImageError::AnimatedSource { format: "WebP" })
    ));
}

fn riff_chunk(out: &mut Vec<u8>, fourcc: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(fourcc);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        out.push(0);
    }
}

fn u24(value: u32) -> [u8; 3] {
    let [a, b, c, _] = value.to_le_bytes();
    [a, b, c]
}

/// Two-frame animated WebP built from a lossless still's VP8L bitstream.
fn animated_webp() -> Vec<u8> {
    let still = webp_rgba(&[200; 2 * 2 * 4], size(2, 2), None);
    assert_eq!(&still[12..16], b"VP8L");
    let vp8l_len = u32::from_le_bytes(still[16..20].try_into().unwrap()) as usize;
    let vp8l = &still[20..20 + vp8l_len];

    let mut body = b"WEBP".to_vec();
    let mut vp8x = vec![0x02 | 0x10, 0, 0, 0]; // animation + alpha flags
    vp8x.extend_from_slice(&u24(1));
    vp8x.extend_from_slice(&u24(1));
    riff_chunk(&mut body, b"VP8X", &vp8x);
    riff_chunk(&mut body, b"ANIM", &[0, 0, 0, 0, 0, 0]);
    for _ in 0..2 {
        let mut frame = Vec::new();
        frame.extend_from_slice(&u24(0));
        frame.extend_from_slice(&u24(0));
        frame.extend_from_slice(&u24(1));
        frame.extend_from_slice(&u24(1));
        frame.extend_from_slice(&u24(100));
        frame.push(0);
        riff_chunk(&mut frame, b"VP8L", vp8l);
        riff_chunk(&mut body, b"ANMF", &frame);
    }
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

/// Asymmetric source: `stripe`-pixel bands along one axis (red = band index) and a half split
/// across the other axis (green = 0 or 255).
fn striped_source(src: Size, stripe: u32, vertical: bool) -> SourceImage {
    let mut pixels = Vec::new();
    for y in 0..src.height {
        for x in 0..src.width {
            let (along, across, extent) = if vertical {
                (y, x, src.width)
            } else {
                (x, y, src.height)
            };
            let half = if across < extent / 2 { 0 } else { 255 };
            pixels.extend_from_slice(&[(along / stripe) as u8, half, 0, 255]);
        }
    }
    SourceImage::from_rgba8(src, pixels).unwrap()
}

/// Samples the source nearest to the cover-mapped centre of output pixel `(x, y)`.
fn sample(cover: CoverMap, image: &SourceImage, output: Size, x: u32, y: u32) -> [u8; 4] {
    let uv = [
        (x as f32 + 0.5) / output.width as f32,
        (y as f32 + 0.5) / output.height as f32,
    ];
    let [u, v] = cover.map(uv);
    let src = image.size();
    pixel(
        image,
        (u * src.width as f32) as u32,
        (v * src.height as f32) as u32,
    )
}

#[test]
fn wide_source_cover_crops_centered_columns_without_flipping() {
    let src = size(400, 100);
    let output = size(4, 4);
    let image = striped_source(src, 25, false);
    let cover = CoverMap::new(src, output).unwrap();
    assert_eq!(cover.scale(), [0.25, 1.0]);
    assert_eq!(cover.map([0.0, 0.0]), [0.375, 0.0]);
    assert_eq!(cover.map([1.0, 1.0]), [0.625, 1.0]);
    // Output column centres land on source x 162.5, 187.5, 212.5, 237.5: stripes 6..=9.
    for (x, stripe) in [(0, 6), (1, 7), (2, 8), (3, 9)] {
        assert_eq!(sample(cover, &image, output, x, 0), [stripe, 0, 0, 255]);
        assert_eq!(sample(cover, &image, output, x, 3), [stripe, 255, 0, 255]);
    }
}

#[test]
fn portrait_source_cover_crops_centered_rows_without_flipping() {
    let src = size(100, 400);
    let output = size(16, 9);
    let image = striped_source(src, 10, true);
    let cover = CoverMap::new(src, output).unwrap();
    assert_eq!(cover.scale(), [1.0, 0.140625]);
    assert_eq!(cover.map([0.0, 0.0]), [0.0, 0.4296875]);
    assert_eq!(cover.map([1.0, 1.0]), [1.0, 0.5703125]);
    // Visible source rows span 171.875..228.125. The top-left output pixel centre lands on
    // source (3.125, 175) and the bottom-right one on (96.875, 225): bands 17 and 22.
    let top = sample(cover, &image, output, 0, 0);
    let bottom = sample(cover, &image, output, 15, 8);
    assert_eq!(top, [17, 0, 0, 255]);
    assert_eq!(bottom, [22, 255, 0, 255]);
}

#[test]
fn cover_region_matches_output_aspect_and_rejects_zero_sizes() {
    for (src, output) in [
        (size(1920, 1080), size(1080, 1920)),
        (size(3000, 2000), size(1920, 1080)),
        (size(7, 13), size(640, 480)),
        (size(1024, 1024), size(1024, 1024)),
    ] {
        let [sx, sy] = CoverMap::new(src, output).unwrap().scale();
        assert!(sx <= 1.0 && sy <= 1.0 && (sx == 1.0 || sy == 1.0));
        let region = (sx * src.width as f32) / (sy * src.height as f32);
        let aspect = output.width as f32 / output.height as f32;
        assert!(
            (region - aspect).abs() <= aspect * 1e-6,
            "{src:?} -> {output:?}"
        );
    }
    assert_eq!(
        CoverMap::new(size(9, 9), size(2, 2))
            .unwrap()
            .map([0.3, 0.8]),
        [0.3, 0.8]
    );
    assert!(matches!(
        CoverMap::new(size(0, 9), size(2, 2)),
        Err(ImageError::CoverSize {
            which: "source",
            ..
        })
    ));
    assert!(matches!(
        CoverMap::new(size(9, 9), size(2, 0)),
        Err(ImageError::CoverSize {
            which: "output",
            ..
        })
    ));
}

#[test]
fn depth_16_bit_roundtrip_is_exact_without_normalization() {
    // A narrow mid-range band in a non-square map: any min/max stretch or transpose would show.
    let map_size = size(3, 2);
    let samples = [1000u16, 1001, 2000, 12345, 1500, 1999];
    let depth = DepthMap::decode_png(&gray16(map_size, &samples)).unwrap();
    assert_eq!(depth.size(), map_size);
    let expected: Vec<f32> = samples.iter().map(|&s| f32::from(s) / 65535.0).collect();
    assert_eq!(depth.values(), expected);

    let encoded = depth.encode_png().unwrap();
    let mut decoder = png::Decoder::new(Cursor::new(&encoded));
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info().unwrap();
    assert_eq!(reader.info().color_type, png::ColorType::Grayscale);
    assert_eq!(reader.info().bit_depth, png::BitDepth::Sixteen);
    assert_eq!(reader.info().size(), (3, 2));
    let mut raw = vec![0; reader.output_buffer_size().unwrap()];
    reader.next_frame(&mut raw).unwrap();
    let stored: Vec<u16> = raw
        .chunks_exact(2)
        .map(|b| u16::from_be_bytes([b[0], b[1]]))
        .collect();
    assert_eq!(stored, samples);
    assert_eq!(DepthMap::decode_png(&encoded).unwrap(), depth);

    // Every 16-bit level survives f32 storage and re-encoding.
    let all: Vec<u16> = (0..=u16::MAX).collect();
    let full = DepthMap::decode_png(&gray16(size(256, 256), &all)).unwrap();
    let again = DepthMap::decode_png(&full.encode_png().unwrap()).unwrap();
    assert_eq!(again.values(), full.values());
    assert_eq!(full.values()[0], 0.0);
    assert_eq!(full.values()[65535], 1.0);
}

#[test]
fn depth_8_bit_values_are_preserved() {
    let map_size = size(2, 3);
    let samples = [0u8, 1, 128, 254, 255, 77];
    let bytes = png_raw(
        map_size,
        png::ColorType::Grayscale,
        png::BitDepth::Eight,
        &samples,
        |_| {},
    );
    let depth = DepthMap::decode_png(&bytes).unwrap();
    assert_eq!(depth.size(), map_size);
    let expected: Vec<f32> = samples.iter().map(|&s| f32::from(s) / 255.0).collect();
    assert_eq!(depth.values(), expected);
    // 8-bit level v is stored as 16-bit v * 257 and decodes to the same value.
    let reencoded = DepthMap::decode_png(&depth.encode_png().unwrap()).unwrap();
    assert_eq!(reencoded.values(), expected);
}

#[test]
fn depth_from_f32_quantizes_on_encode() {
    let depth = DepthMap::from_f32(size(4, 1), vec![0.0, 0.25, 0.5, 1.0]).unwrap();
    let decoded = DepthMap::decode_png(&depth.encode_png().unwrap()).unwrap();
    let levels: Vec<f32> = [0.0f32, 16384.0, 32768.0, 65535.0]
        .iter()
        .map(|l| l / 65535.0)
        .collect();
    assert_eq!(decoded.values(), levels);
}

#[test]
fn depth_rejects_non_grayscale_and_animated_pngs() {
    let s = size(2, 2);
    let cases: [(png::ColorType, png::BitDepth, usize); 4] = [
        (png::ColorType::Rgb, png::BitDepth::Eight, 12),
        (png::ColorType::Rgba, png::BitDepth::Sixteen, 32),
        (png::ColorType::GrayscaleAlpha, png::BitDepth::Eight, 8),
        (png::ColorType::Indexed, png::BitDepth::Eight, 4),
    ];
    for (color, depth, len) in cases {
        let bytes = png_raw(s, color, depth, &vec![0; len], |encoder| {
            if color == png::ColorType::Indexed {
                encoder.set_palette(vec![0, 0, 0]);
            }
        });
        assert!(
            matches!(
                DepthMap::decode_png(&bytes),
                Err(ImageError::DepthColor { .. })
            ),
            "{color:?}"
        );
    }

    let four_bit = png_raw(
        s,
        png::ColorType::Grayscale,
        png::BitDepth::Four,
        &[0, 0],
        |_| {},
    );
    assert!(matches!(
        DepthMap::decode_png(&four_bit),
        Err(ImageError::DepthBitDepth { bits: 4 })
    ));

    let keyed = png_raw(
        s,
        png::ColorType::Grayscale,
        png::BitDepth::Eight,
        &[0; 4],
        |encoder| encoder.set_trns(vec![0, 0]),
    );
    assert!(matches!(
        DepthMap::decode_png(&keyed),
        Err(ImageError::DepthTransparency)
    ));

    let apng = {
        let mut out = Vec::new();
        let mut encoder = png::Encoder::new(&mut out, 2, 2);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Sixteen);
        encoder.set_animated(2, 0).unwrap();
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&[0; 8]).unwrap();
        writer.write_image_data(&[0; 8]).unwrap();
        writer.finish().unwrap();
        out
    };
    assert!(matches!(
        DepthMap::decode_png(&apng),
        Err(ImageError::DepthAnimated)
    ));

    let jpeg = jpeg_rgb(&[0; 2 * 2 * 3], s, None);
    assert!(matches!(
        DepthMap::decode_png(&jpeg),
        Err(ImageError::DepthDecode(_))
    ));
    let mut truncated = gray16(s, &[1, 2, 3, 4]);
    truncated.truncate(truncated.len() - 20);
    assert!(DepthMap::decode_png(&truncated).is_err());
}

#[test]
fn depth_from_f32_rejects_bad_values_and_lengths() {
    let s = size(3, 2);
    for (bad, value) in [
        (f32::NAN, None),
        (f32::INFINITY, Some(f32::INFINITY)),
        (-0.001, Some(-0.001)),
        (1.0001, Some(1.0001)),
    ] {
        let mut values = vec![0.5; 6];
        values[4] = bad;
        match DepthMap::from_f32(s, values) {
            Err(ImageError::DepthValue {
                index: 4,
                x: 1,
                y: 1,
                value: got,
            }) => match value {
                Some(value) => assert_eq!(got, value),
                None => assert!(got.is_nan()),
            },
            other => panic!("{bad}: {other:?}"),
        }
    }
    assert!(matches!(
        DepthMap::from_f32(s, vec![0.5; 5]),
        Err(ImageError::BufferLength {
            expected: 6,
            actual: 5,
            ..
        })
    ));
    assert!(matches!(
        DepthMap::from_f32(size(3, 0), Vec::new()),
        Err(ImageError::ZeroSize { height: 0, .. })
    ));
}
