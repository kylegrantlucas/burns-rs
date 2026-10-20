//! Source images, raw depth maps, and cover-crop UV mapping.
//!
//! Everything here works on in-memory bytes; there is no filesystem, network, or GPU access.
//!
//! * [`SourceImage`] is an opaque RGBA8 still decoded from PNG, JPEG, or WebP. EXIF orientation
//!   is applied, any alpha is composited over black, and the oriented resolution is kept as-is.
//! * [`DepthMap`] holds normalized `f32` depth samples where `1.0` (white) is nearest. It imports
//!   and exports grayscale PNG without reorienting, cropping, or renormalizing.
//! * [`CoverMap`] maps output UV coordinates into source UV coordinates so the full source covers
//!   the output aspect ratio with a centered crop.

use std::fmt;
use std::io::{Cursor, Write};

use ::image::codecs::jpeg::JpegDecoder;
use ::image::codecs::png::PngDecoder;
use ::image::codecs::webp::WebPDecoder;
use ::image::{DynamicImage, ImageDecoder, ImageFormat, Limits, RgbaImage};

use crate::Size;

/// Largest decoded sample buffer accepted for a depth map, matching the `image` crate's default
/// allocation limit that also bounds [`SourceImage::decode`].
const MAX_DECODE_BYTES: usize = 512 * 1024 * 1024;

/// Bytes per RGBA8 pixel.
const RGBA_CHANNELS: usize = 4;

/// Errors produced while constructing, decoding, or encoding images and depth maps.
#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    /// A width or height is zero.
    #[error("{what} size {width}x{height} is invalid: width and height must both be at least 1")]
    ZeroSize {
        what: &'static str,
        width: u32,
        height: u32,
    },
    /// The dimensions overflow the addressable buffer size.
    #[error(
        "{what} size {width}x{height} is too large: its sample buffer does not fit in memory; \
         downscale it before loading"
    )]
    SizeOverflow {
        what: &'static str,
        width: u32,
        height: u32,
    },
    /// The decoded buffer would exceed the decode allocation limit.
    #[error(
        "{what} size {width}x{height} needs {required} bytes, above the {limit}-byte decode limit; \
         downscale it before loading"
    )]
    TooLarge {
        what: &'static str,
        width: u32,
        height: u32,
        required: u64,
        limit: u64,
    },
    /// A raw buffer does not hold exactly one entry per pixel channel.
    #[error(
        "{what} buffer for {width}x{height} must contain exactly {expected} {unit}, got {actual}"
    )]
    BufferLength {
        what: &'static str,
        width: u32,
        height: u32,
        expected: usize,
        actual: usize,
        unit: &'static str,
    },
    /// The bytes do not start with a recognizable image signature.
    #[error("source image format was not recognized; provide a PNG, JPEG, or WebP file")]
    UnrecognizedFormat,
    /// The bytes are a recognized image format other than PNG, JPEG, or WebP.
    #[error("source image format {format} is not supported; convert it to PNG, JPEG, or WebP")]
    UnsupportedFormat { format: String },
    /// The source is an animation rather than a single still image.
    #[error("animated {format} source images are not supported; export a single still frame")]
    AnimatedSource { format: &'static str },
    /// The source image bytes could not be decoded.
    #[error("failed to decode {format} source image: {source}")]
    SourceDecode {
        format: &'static str,
        #[source]
        source: ::image::ImageError,
    },
    /// The depth PNG is not single-channel grayscale.
    #[error(
        "depth map PNG has {color} color; depth maps must be single-channel grayscale without \
         alpha or palette, so export it as 8- or 16-bit grayscale"
    )]
    DepthColor { color: String },
    /// The depth PNG is grayscale but neither 8 nor 16 bits per sample.
    #[error(
        "depth map PNG has {bits}-bit samples; depth maps must use 8- or 16-bit grayscale samples"
    )]
    DepthBitDepth { bits: u8 },
    /// The depth PNG declares transparency through a `tRNS` chunk.
    #[error(
        "depth map PNG declares transparency (tRNS chunk); export it as opaque 8- or 16-bit \
         grayscale"
    )]
    DepthTransparency,
    /// The depth PNG is an animated PNG.
    #[error("depth map PNG is animated (APNG); export a single still grayscale frame")]
    DepthAnimated,
    /// The depth PNG bytes could not be decoded.
    #[error("failed to decode depth map PNG: {0}")]
    DepthDecode(#[source] png::DecodingError),
    /// The depth map could not be encoded as PNG.
    #[error("failed to encode depth map PNG: {0}")]
    DepthEncode(#[source] png::EncodingError),
    /// A depth sample is non-finite or outside `0.0..=1.0`.
    #[error(
        "depth value at index {index} (x {x}, y {y}) is {value}; depth values must be finite \
         and within 0.0..=1.0"
    )]
    DepthValue {
        index: usize,
        x: u32,
        y: u32,
        value: f32,
    },
    /// A cover-map source or output size has a zero dimension.
    #[error(
        "cover map {which} size {width}x{height} is invalid: width and height must both be at \
         least 1"
    )]
    CoverSize {
        which: &'static str,
        width: u32,
        height: u32,
    },
}

/// Returns `width * height * channels` after rejecting zero dimensions and overflow.
fn sample_len(what: &'static str, size: Size, channels: usize) -> Result<usize, ImageError> {
    let Size { width, height } = size;
    if width == 0 || height == 0 {
        return Err(ImageError::ZeroSize {
            what,
            width,
            height,
        });
    }
    usize::try_from(width)
        .ok()
        .zip(usize::try_from(height).ok())
        .and_then(|(w, h)| w.checked_mul(h))
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or(ImageError::SizeOverflow {
            what,
            width,
            height,
        })
}

fn check_len(
    what: &'static str,
    size: Size,
    expected: usize,
    actual: usize,
    unit: &'static str,
) -> Result<(), ImageError> {
    if expected == actual {
        Ok(())
    } else {
        Err(ImageError::BufferLength {
            what,
            width: size.width,
            height: size.height,
            expected,
            actual,
            unit,
        })
    }
}

/// An opaque RGBA8 source still at its original (EXIF-oriented) resolution.
///
/// Pixels are row-major, top row first, four bytes per pixel, and every alpha byte is `255`.
#[derive(Clone, PartialEq, Eq)]
pub struct SourceImage {
    size: Size,
    pixels: Vec<u8>,
}

impl SourceImage {
    const WHAT: &'static str = "source image";

    /// Decodes a PNG, JPEG, or WebP still.
    ///
    /// The EXIF orientation is read from the decoder before decoding and applied to the pixels,
    /// so the result is upright with the oriented resolution. Translucent pixels are composited
    /// over black. Animated PNG and WebP files are rejected.
    pub fn decode(bytes: &[u8]) -> Result<Self, ImageError> {
        let format = ::image::guess_format(bytes).map_err(|_| ImageError::UnrecognizedFormat)?;
        let rgba = match format {
            ImageFormat::Png => {
                let decoder = PngDecoder::with_limits(Cursor::new(bytes), Limits::default())
                    .map_err(source_decode("PNG"))?;
                if decoder.is_apng().map_err(source_decode("PNG"))? {
                    return Err(ImageError::AnimatedSource { format: "PNG" });
                }
                decode_oriented(decoder, "PNG")?
            }
            ImageFormat::Jpeg => {
                let decoder =
                    JpegDecoder::new(Cursor::new(bytes)).map_err(source_decode("JPEG"))?;
                decode_oriented(decoder, "JPEG")?
            }
            ImageFormat::WebP => {
                let decoder =
                    WebPDecoder::new(Cursor::new(bytes)).map_err(source_decode("WebP"))?;
                if decoder.has_animation() {
                    return Err(ImageError::AnimatedSource { format: "WebP" });
                }
                decode_oriented(decoder, "WebP")?
            }
            other => {
                return Err(ImageError::UnsupportedFormat {
                    format: format!("{other:?}"),
                });
            }
        };
        let size = Size {
            width: rgba.width(),
            height: rgba.height(),
        };
        Self::from_rgba8(size, rgba.into_raw())
    }

    /// Wraps row-major RGBA8 pixels, compositing any translucent pixel over black.
    ///
    /// `pixels` must hold exactly `width * height * 4` bytes.
    pub fn from_rgba8(size: Size, mut pixels: Vec<u8>) -> Result<Self, ImageError> {
        let expected = sample_len(Self::WHAT, size, RGBA_CHANNELS)?;
        check_len(Self::WHAT, size, expected, pixels.len(), "bytes")?;
        composite_over_black(&mut pixels);
        Ok(Self { size, pixels })
    }

    /// Pixel dimensions.
    pub fn size(&self) -> Size {
        self.size
    }

    /// Row-major opaque RGBA8 bytes.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
}

impl fmt::Debug for SourceImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SourceImage")
            .field("size", &self.size)
            .field("bytes", &self.pixels.len())
            .finish()
    }
}

fn source_decode(format: &'static str) -> impl Fn(::image::ImageError) -> ImageError {
    move |source| ImageError::SourceDecode { format, source }
}

/// Reads the EXIF orientation, decodes within the default allocation limits, then orients.
fn decode_oriented(
    mut decoder: impl ImageDecoder,
    format: &'static str,
) -> Result<RgbaImage, ImageError> {
    let orientation = decoder.orientation().map_err(source_decode(format))?;
    let mut limits = Limits::default();
    limits
        .reserve(decoder.total_bytes())
        .map_err(source_decode(format))?;
    decoder.set_limits(limits).map_err(source_decode(format))?;
    let mut image = DynamicImage::from_decoder(decoder).map_err(source_decode(format))?;
    image.apply_orientation(orientation);
    Ok(image.into_rgba8())
}

/// Premultiplies each channel by alpha with rounding (`round(c * a / 255)`), then sets alpha to
/// 255, which is "over black" compositing.
fn composite_over_black(pixels: &mut [u8]) {
    for pixel in pixels.chunks_exact_mut(RGBA_CHANNELS) {
        let alpha = u16::from(pixel[3]);
        if alpha == 255 {
            continue;
        }
        for channel in &mut pixel[..3] {
            // c * a <= 65025, and adding 127 rounds to nearest; the quotient is at most 255.
            *channel = ((u16::from(*channel) * alpha + 127) / 255) as u8;
        }
        pixel[3] = 255;
    }
}

/// Normalized depth samples, row-major, top row first; `1.0` (white) is nearest.
#[derive(Clone, PartialEq)]
pub struct DepthMap {
    size: Size,
    values: Vec<f32>,
}

impl DepthMap {
    const WHAT: &'static str = "depth map";

    /// Wraps row-major depth samples. Every value must be finite and within `0.0..=1.0`, and
    /// `values` must hold exactly `width * height` samples.
    pub fn from_f32(size: Size, values: Vec<f32>) -> Result<Self, ImageError> {
        let expected = sample_len(Self::WHAT, size, 1)?;
        check_len(Self::WHAT, size, expected, values.len(), "samples")?;
        if let Some(index) = values.iter().position(|value| !(0.0..=1.0).contains(value)) {
            let width = size.width as usize;
            return Err(ImageError::DepthValue {
                index,
                // Both quotients are below `width`/`height`, so they fit in u32.
                x: (index % width) as u32,
                y: (index / width) as u32,
                value: values[index],
            });
        }
        Ok(Self { size, values })
    }

    /// Decodes an 8- or 16-bit grayscale PNG at any resolution.
    ///
    /// Samples become `value / 255` or `value / 65535` exactly as stored: no reorientation,
    /// cropping, range stretching, or other processing. Color, palette, alpha, `tRNS`, sub-8-bit,
    /// and animated PNGs are rejected.
    pub fn decode_png(bytes: &[u8]) -> Result<Self, ImageError> {
        let mut decoder = png::Decoder::new_with_limits(
            Cursor::new(bytes),
            png::Limits {
                bytes: MAX_DECODE_BYTES,
            },
        );
        // Keep raw samples: no 16-to-8-bit stripping, expansion, or gamma handling.
        decoder.set_transformations(png::Transformations::IDENTITY);
        let mut reader = decoder.read_info().map_err(ImageError::DepthDecode)?;
        let info = reader.info();
        if info.animation_control.is_some() {
            return Err(ImageError::DepthAnimated);
        }
        if info.color_type != png::ColorType::Grayscale {
            return Err(ImageError::DepthColor {
                color: format!("{:?}", info.color_type),
            });
        }
        let bytes_per_sample = match info.bit_depth {
            png::BitDepth::Eight => 1,
            png::BitDepth::Sixteen => 2,
            other => {
                return Err(ImageError::DepthBitDepth { bits: other as u8 });
            }
        };
        if info.trns.is_some() {
            return Err(ImageError::DepthTransparency);
        }
        let size = Size {
            width: info.width,
            height: info.height,
        };
        let samples = sample_len(Self::WHAT, size, 1)?;
        let value_bytes = samples.checked_mul(size_of::<f32>());
        if value_bytes.is_none_or(|value_bytes| value_bytes > MAX_DECODE_BYTES) {
            return Err(ImageError::TooLarge {
                what: Self::WHAT,
                width: size.width,
                height: size.height,
                required: (samples as u64).saturating_mul(size_of::<f32>() as u64),
                limit: MAX_DECODE_BYTES as u64,
            });
        }
        let raw_len = samples * bytes_per_sample;
        let mut raw = vec![0; raw_len];
        let frame = reader
            .next_frame(&mut raw)
            .map_err(ImageError::DepthDecode)?;
        check_len(Self::WHAT, size, raw_len, frame.buffer_size(), "bytes")?;
        let values = if bytes_per_sample == 1 {
            raw.iter().map(|&v| f32::from(v) / 255.0).collect()
        } else {
            // PNG stores 16-bit samples in network (big-endian) byte order.
            raw.chunks_exact(2)
                .map(|pair| f32::from(u16::from_be_bytes([pair[0], pair[1]])) / 65535.0)
                .collect()
        };
        Ok(Self { size, values })
    }

    /// Encodes the full map, unchanged in orientation and extent, as a 16-bit grayscale PNG with
    /// samples `round(value * 65535)`.
    pub fn encode_png(&self) -> Result<Vec<u8>, ImageError> {
        let Size { width, height } = self.size;
        let mut out = Vec::new();
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Sixteen);
        let mut writer = encoder.write_header().map_err(ImageError::DepthEncode)?;
        let mut stream = writer.stream_writer().map_err(ImageError::DepthEncode)?;
        let mut row = vec![0; width as usize * 2];
        for line in self.values.chunks_exact(width as usize) {
            for (sample, &value) in row.chunks_exact_mut(2).zip(line) {
                // Values are validated to 0.0..=1.0, so the rounded product fits in u16.
                let quantized = (value * 65535.0).round() as u16;
                sample.copy_from_slice(&quantized.to_be_bytes());
            }
            stream
                .write_all(&row)
                .map_err(|err| ImageError::DepthEncode(err.into()))?;
        }
        stream.finish().map_err(ImageError::DepthEncode)?;
        writer.finish().map_err(ImageError::DepthEncode)?;
        Ok(out)
    }

    /// Sample dimensions.
    pub fn size(&self) -> Size {
        self.size
    }

    /// Row-major normalized samples; `1.0` is nearest.
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

impl fmt::Debug for DepthMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DepthMap")
            .field("size", &self.size)
            .field("samples", &self.values.len())
            .finish()
    }
}

/// Centered cover crop from output UV space into source UV space.
///
/// With `R = source.width / source.height` and `A = output.width / output.height`, the scale is
/// `[min(1, A / R), min(1, R / A)]` and `map(uv) = 0.5 + (uv - 0.5) * scale`. The whole source is
/// the crop domain, so the visible region keeps the output aspect ratio and is never cropped
/// twice. UV `[0, 0]` is the top-left corner in both spaces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoverMap {
    scale: [f32; 2],
}

impl CoverMap {
    /// Builds the mapping, rejecting zero-sized source or output dimensions.
    pub fn new(source: Size, output: Size) -> Result<Self, ImageError> {
        for (which, size) in [("source", source), ("output", output)] {
            if size.width == 0 || size.height == 0 {
                return Err(ImageError::CoverSize {
                    which,
                    width: size.width,
                    height: size.height,
                });
            }
        }
        let source_aspect = f64::from(source.width) / f64::from(source.height);
        let output_aspect = f64::from(output.width) / f64::from(output.height);
        Ok(Self {
            scale: [
                (output_aspect / source_aspect).min(1.0) as f32,
                (source_aspect / output_aspect).min(1.0) as f32,
            ],
        })
    }

    /// Per-axis UV scale `[sx, sy]`; at least one component is `1.0`.
    pub fn scale(self) -> [f32; 2] {
        self.scale
    }

    /// Maps an output UV coordinate to the source UV coordinate it samples.
    pub fn map(self, uv: [f32; 2]) -> [f32; 2] {
        [
            0.5 + (uv[0] - 0.5) * self.scale[0],
            0.5 + (uv[1] - 0.5) * self.scale[1],
        ]
    }
}
