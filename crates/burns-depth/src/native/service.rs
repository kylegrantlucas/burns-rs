//! Cache-first depth generation shared by the native CLI and GUI.

use std::{
    fs, io,
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

use burns::{DepthMap, SourceImage};
use tempfile::NamedTempFile;

use crate::{DepthError, DepthProgress, model_input_size, source_cache_key};

use super::{DepthEstimator, ModelSource, ensure_model_file, resolve_ort_library};

/// Owns the depth cache under `<platform cache dir>/burns/` and one lazily created estimator.
///
/// * `depth/<source_cache_key>.png`: canonical 16-bit generated maps.
/// * `models/<sha256>.onnx`: the verified pinned model.
///
/// [`cached`](Self::cached) only reads the cache. [`generate`](Self::generate) is the explicit
/// action that may download the model (about 99 MB), load ONNX Runtime, and run inference.
/// Imported depth maps are never written to this cache.
#[derive(Debug)]
pub struct DepthService {
    root: PathBuf,
    model: ModelSource,
    /// Set once the model file has been verified by this service; not re-hashed afterwards.
    verified_model: Option<PathBuf>,
    estimator: Option<DepthEstimator>,
}

impl DepthService {
    /// Uses `burns/` inside the platform user cache directory (for example
    /// `~/Library/Caches/burns` on macOS, `$XDG_CACHE_HOME/burns` on Linux).
    pub fn new() -> Result<Self, DepthError> {
        let dirs = directories::BaseDirs::new().ok_or(DepthError::NoCacheDir)?;
        Ok(Self::with_cache_root(dirs.cache_dir().join("burns")))
    }

    /// Uses `root` as the cache root instead of the platform location.
    pub fn with_cache_root(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            model: ModelSource::pinned(),
            verified_model: None,
            estimator: None,
        }
    }

    /// The cache root.
    pub fn cache_root(&self) -> &Path {
        &self.root
    }

    /// Where the generated map for `image` is cached.
    pub fn depth_cache_path(&self, image: &SourceImage) -> PathBuf {
        self.root
            .join("depth")
            .join(format!("{}.png", source_cache_key(image)))
    }

    /// Where the verified model is stored.
    pub fn model_path(&self) -> PathBuf {
        self.root.join("models").join(self.model.file_name())
    }

    /// The cached map for `image`, if any. Needs no model, runtime, or network.
    ///
    /// A cached file that cannot be decoded, or whose size differs from the model input size for
    /// `image`, is reported as [`DepthError::CacheCorrupt`].
    pub fn cached(&self, image: &SourceImage) -> Result<Option<DepthMap>, DepthError> {
        let path = self.depth_cache_path(image);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(DepthError::io("read", &path, err)),
        };
        let corrupt = |reason: String| DepthError::CacheCorrupt {
            path: path.clone(),
            reason,
        };
        let map = DepthMap::decode_png(&bytes).map_err(|err| corrupt(err.to_string()))?;
        let expected = model_input_size(image.size())?;
        if map.size() != expected {
            return Err(corrupt(format!(
                "it is {}x{}, expected {}x{}",
                map.size().width,
                map.size().height,
                expected.width,
                expected.height
            )));
        }
        Ok(Some(map))
    }

    /// Returns the verified model path, downloading it if absent or invalid. An existing file is
    /// hashed once per service, not on every generation.
    pub fn ensure_model(
        &mut self,
        cancel: &AtomicBool,
        progress: impl FnMut(DepthProgress),
    ) -> Result<PathBuf, DepthError> {
        if let Some(path) = &self.verified_model {
            return Ok(path.clone());
        }
        let path = ensure_model_file(&self.root.join("models"), &self.model, cancel, progress)?;
        self.verified_model = Some(path.clone());
        Ok(path)
    }

    /// Returns the depth map for `image`: the cached one if present
    /// ([`DepthProgress::CacheHit`]), otherwise downloads the model if needed, runs inference,
    /// and caches the result.
    ///
    /// The ONNX Runtime library is located before any download, so a missing runtime fails fast.
    /// `cancel` is checked before each step and each download chunk. A running inference call is
    /// not interruptible; cancelling during it discards the result instead of caching or
    /// returning it. If the generated map cannot be cached, it is still returned and a
    /// [`DepthProgress::PersistenceWarning`] is reported. The returned map is identical to what a
    /// later [`cached`](Self::cached) call yields.
    pub fn generate(
        &mut self,
        image: &SourceImage,
        cancel: &AtomicBool,
        mut progress: impl FnMut(DepthProgress),
    ) -> Result<DepthMap, DepthError> {
        let check_cancel = || {
            if cancel.load(Ordering::Relaxed) {
                Err(DepthError::Cancelled)
            } else {
                Ok(())
            }
        };
        check_cancel()?;
        // Validate the size before touching the network or runtime.
        model_input_size(image.size())?;
        // An unreadable or corrupt cache entry is regenerated and overwritten.
        if let Ok(Some(map)) = self.cached(image) {
            progress(DepthProgress::CacheHit);
            return Ok(map);
        }
        if self.estimator.is_none() {
            let library = resolve_ort_library()?;
            let model = self.ensure_model(cancel, &mut progress)?;
            check_cancel()?;
            progress(DepthProgress::Loading);
            self.estimator = Some(DepthEstimator::new(&model, &library)?);
        }
        check_cancel()?;
        progress(DepthProgress::Inferring);
        let estimator = self.estimator.as_mut().expect("estimator was just created");
        let map = estimator.estimate(image)?;
        check_cancel()?;
        progress(DepthProgress::Saving);
        if let Err(err) = self.store(image, &map) {
            progress(DepthProgress::PersistenceWarning {
                message: format!("depth map was generated but not cached: {err}"),
            });
        }
        Ok(map)
    }

    /// Atomically writes `map` to the cache via a temporary file in the same directory.
    fn store(&self, image: &SourceImage, map: &DepthMap) -> Result<(), DepthError> {
        let path = self.depth_cache_path(image);
        let dir = path.parent().expect("cache path has a parent");
        let png = map.encode_png()?;
        fs::create_dir_all(dir).map_err(|err| DepthError::io("create", dir, err))?;
        let mut temp = NamedTempFile::new_in(dir)
            .map_err(|err| DepthError::io("create temporary file in", dir, err))?;
        temp.write_all(&png)
            .map_err(|err| DepthError::io("write", temp.path(), err))?;
        // Same key means same pixels and pipeline, so replacing a concurrent writer's file is
        // harmless and replaces any corrupt entry.
        temp.persist(&path)
            .map_err(|err| DepthError::io("publish", &path, err.error))?;
        Ok(())
    }
}
