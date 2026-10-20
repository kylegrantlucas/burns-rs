//! Reading source images and depth maps from disk, and saving depth maps.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::AtomicBool;

use burns::{DepthMap, SourceImage};

use crate::frame::check_cancel;
use crate::{MediaError, safe_write};

/// Reads `path` once and decodes it as a PNG, JPEG, or WebP source still.
pub fn load_image(path: &Path) -> Result<SourceImage, MediaError> {
    let bytes = fs::read(path).map_err(|error| MediaError::io("read source image", path, error))?;
    SourceImage::decode(&bytes).map_err(|source| MediaError::Asset {
        path: path.to_owned(),
        source,
    })
}

/// Reads `path` once and decodes it as an 8- or 16-bit grayscale depth PNG (white is near).
pub fn load_depth(path: &Path) -> Result<DepthMap, MediaError> {
    let bytes = fs::read(path).map_err(|error| MediaError::io("read depth map", path, error))?;
    DepthMap::decode_png(&bytes).map_err(|source| MediaError::Asset {
        path: path.to_owned(),
        source,
    })
}

/// Saves `depth` as a 16-bit grayscale PNG through [`safe_write`].
pub fn write_depth_png(
    depth: &DepthMap,
    output: &Path,
    overwrite: bool,
    cancel: Option<&AtomicBool>,
) -> Result<(), MediaError> {
    safe_write(output, overwrite, |file| {
        check_cancel(cancel)?;
        let png = depth.encode_png()?;
        check_cancel(cancel)?;
        file.write_all(&png)
            .map_err(|error| MediaError::io("write depth map", output, error))?;
        check_cancel(cancel)
    })
}
