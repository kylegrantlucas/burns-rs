//! Row-streamed sRGB-encoded RGBA8 PNG output, independent of GPU and filesystem APIs.

use std::io::{self, Write};

use crate::{ConfigError, Size};

#[derive(Debug, thiserror::Error)]
pub enum PngError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(
        "invalid RGBA rows for {width}x{height}: row {row} bytes, stride {stride}, buffer {len} bytes, need {required}"
    )]
    Layout {
        width: u32,
        height: u32,
        row: usize,
        stride: usize,
        len: usize,
        required: usize,
    },
    #[error("PNG encoding failed: {0}")]
    Encode(#[from] png::EncodingError),
    #[error("PNG output I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("PNG encoding cancelled")]
    Cancelled,
}

/// Writes only useful RGBA bytes from each padded row, tagged as sRGB.
/// Cancellation is checked before encoding and between rows; publication belongs to the caller.
pub fn write_rgba8_png(
    output: &mut impl Write,
    size: Size,
    rows: &[u8],
    stride: usize,
    mut cancelled: impl FnMut() -> bool,
) -> Result<(), PngError> {
    size.validate()?;
    let row = size.width as usize * 4;
    let required = stride
        .checked_mul(size.height as usize - 1)
        .and_then(|bytes| bytes.checked_add(row))
        .unwrap_or(usize::MAX);
    if stride < row || rows.len() < required {
        return Err(PngError::Layout {
            width: size.width,
            height: size.height,
            row,
            stride,
            len: rows.len(),
            required,
        });
    }
    if cancelled() {
        return Err(PngError::Cancelled);
    }
    let mut encoder = png::Encoder::new(&mut *output, size.width, size.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    let mut writer = encoder.write_header()?;
    let mut stream = writer.stream_writer()?;
    for y in 0..size.height as usize {
        if cancelled() {
            return Err(PngError::Cancelled);
        }
        stream.write_all(&rows[y * stride..y * stride + row])?;
    }
    stream.finish()?;
    writer.finish()?;
    output.flush()?;
    Ok(())
}
