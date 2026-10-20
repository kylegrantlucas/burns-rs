//! Content-addressed cache key for generated depth maps.

use burns::SourceImage;
use sha2::{Digest, Sha256};
use std::sync::LazyLock;

use crate::{MODEL_SHA256, PREPROCESSING_VERSION};

/// Domain separator that starts every hashed key preimage.
pub const CACHE_KEY_DOMAIN: &[u8] = b"burns-depth source cache key\0";

static MODEL_HASH: LazyLock<[u8; 32]> = LazyLock::new(|| {
    let mut hash = [0; 32];
    hex::decode_to_slice(MODEL_SHA256, &mut hash).expect("MODEL_SHA256 is valid hex");
    hash
});

/// Lowercase hex SHA-256 identifying the depth map generated for `image`.
///
/// The hashed bytes, in order, are:
///
/// 1. [`CACHE_KEY_DOMAIN`] (29 bytes, including the trailing NUL),
/// 2. [`PREPROCESSING_VERSION`] as a little-endian `u32`,
/// 3. the 32 raw bytes of the model SHA-256 ([`MODEL_SHA256`] hex-decoded),
/// 4. the oriented width, then height, each a little-endian `u32`,
/// 5. the oriented, opaque RGBA8 pixels, row-major with the top row first.
///
/// File names, paths, and encodings never contribute, so the same pixels decoded from PNG,
/// JPEG, or WebP natively or in the browser share one key.
pub fn source_cache_key(image: &SourceImage) -> String {
    let size = image.size();
    let mut hasher = Sha256::new();
    hasher.update(CACHE_KEY_DOMAIN);
    hasher.update(PREPROCESSING_VERSION.to_le_bytes());
    hasher.update(*MODEL_HASH);
    hasher.update(size.width.to_le_bytes());
    hasher.update(size.height.to_le_bytes());
    hasher.update(image.pixels());
    hex::encode(hasher.finalize())
}
