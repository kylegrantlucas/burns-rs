//! Verified, cancellable, atomically published model downloads.

use std::{
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;

use crate::{DepthError, DepthProgress, MODEL_BYTES, MODEL_SHA256, MODEL_URL};

/// Where a model comes from and what it must be.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelSource {
    pub url: String,
    /// Exact length in bytes.
    pub bytes: u64,
    /// Lowercase hex SHA-256.
    pub sha256: String,
}

impl ModelSource {
    /// The pinned model from [`model_manifest`](crate::model_manifest).
    pub fn pinned() -> Self {
        Self {
            url: MODEL_URL.to_owned(),
            bytes: MODEL_BYTES,
            sha256: MODEL_SHA256.to_owned(),
        }
    }

    /// Content-addressed file name, `<sha256>.onnx`.
    pub fn file_name(&self) -> String {
        format!("{}.onnx", self.sha256)
    }
}

/// Checks that `path` is exactly `source.bytes` long and hashes to `source.sha256`.
pub fn verify_model_file(path: &Path, source: &ModelSource) -> Result<(), DepthError> {
    let origin = || format!("`{}`", path.display());
    let mut file = File::open(path).map_err(|err| DepthError::io("open", path, err))?;
    let length = file
        .metadata()
        .map_err(|err| DepthError::io("inspect", path, err))?
        .len();
    if length != source.bytes {
        return Err(DepthError::ModelLength {
            origin: origin(),
            expected: source.bytes,
            actual: length,
        });
    }
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher).map_err(|err| DepthError::io("read", path, err))?;
    check_hash(hasher, source, origin())
}

fn check_hash(hasher: Sha256, source: &ModelSource, origin: String) -> Result<(), DepthError> {
    let actual = hex::encode(hasher.finalize());
    if actual.eq_ignore_ascii_case(&source.sha256) {
        Ok(())
    } else {
        Err(DepthError::ModelHash {
            origin,
            expected: source.sha256.clone(),
            actual,
        })
    }
}

/// Returns `dir/<sha256>.onnx`, verified, downloading it first when needed.
///
/// An existing file is trusted only after its length and SHA-256 match; a mismatching one is
/// deleted and downloaded again. Downloads stream into an owned temporary file in `dir`, are
/// checked for exact length and hash, synced, and only then published with a no-clobber rename,
/// so the final name never holds partial or unverified bytes. `cancel` is checked before every
/// chunk; on cancellation or any failure the temporary file is removed and nothing is published.
/// If another process publishes the same file concurrently, its copy is verified and used.
pub fn ensure_model_file(
    dir: &Path,
    source: &ModelSource,
    cancel: &AtomicBool,
    mut progress: impl FnMut(DepthProgress),
) -> Result<PathBuf, DepthError> {
    let dest = dir.join(source.file_name());
    if dest.is_file() {
        progress(DepthProgress::Verifying);
        match verify_model_file(&dest, source) {
            Ok(()) => return Ok(dest),
            Err(DepthError::ModelLength { .. } | DepthError::ModelHash { .. }) => {
                match fs::remove_file(&dest) {
                    Ok(()) => {}
                    Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                    Err(err) => return Err(DepthError::io("remove corrupt model", &dest, err)),
                }
            }
            Err(err) => return Err(err),
        }
    }
    fs::create_dir_all(dir).map_err(|err| DepthError::io("create", dir, err))?;
    let temp = download_verified(dir, source, cancel, &mut progress)?;
    match temp.persist_noclobber(&dest) {
        Ok(_) => Ok(dest),
        Err(err) if err.error.kind() == io::ErrorKind::AlreadyExists => {
            // Another process won the race; trust its file only after verifying it.
            progress(DepthProgress::Verifying);
            verify_model_file(&dest, source)?;
            Ok(dest)
        }
        Err(err) => Err(DepthError::io("publish", &dest, err.error)),
    }
}

fn download_verified(
    dir: &Path,
    source: &ModelSource,
    cancel: &AtomicBool,
    progress: &mut impl FnMut(DepthProgress),
) -> Result<NamedTempFile, DepthError> {
    if cancel.load(Ordering::Relaxed) {
        return Err(DepthError::Cancelled);
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| DepthError::Download {
            url: source.url.clone(),
            message: err.to_string(),
        })?;
    runtime.block_on(async {
        let cancellation = async {
            loop {
                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        };
        tokio::select! {
            biased;
            () = cancellation => Err(DepthError::Cancelled),
            result = download_body(dir, source, cancel, progress) => result,
        }
    })
}

async fn download_body(
    dir: &Path,
    source: &ModelSource,
    cancel: &AtomicBool,
    progress: &mut impl FnMut(DepthProgress),
) -> Result<NamedTempFile, DepthError> {
    let download_error = |message: String| DepthError::Download {
        url: source.url.clone(),
        message,
    };
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .user_agent(concat!("burns-depth/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|err| download_error(err.to_string()))?;
    let mut response =
        tokio::time::timeout(Duration::from_secs(60), client.get(&source.url).send())
            .await
            .map_err(|err| download_error(err.to_string()))?
            .and_then(reqwest::Response::error_for_status)
            .map_err(|err| download_error(err.to_string()))?;
    if let Some(declared) = response.content_length()
        && declared != source.bytes
    {
        return Err(DepthError::ModelLength {
            origin: source.url.clone(),
            expected: source.bytes,
            actual: declared,
        });
    }
    let mut temp = tempfile::Builder::new()
        .prefix(&format!(".{}.", source.sha256))
        .suffix(".partial")
        .tempfile_in(dir)
        .map_err(|err| DepthError::io("create temporary file in", dir, err))?;
    let mut hasher = Sha256::new();
    let mut received = 0u64;
    progress(DepthProgress::Downloading {
        received,
        total: source.bytes,
    });
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|err| download_error(err.to_string()))?
    {
        if cancel.load(Ordering::Relaxed) {
            return Err(DepthError::Cancelled);
        }
        received += chunk.len() as u64;
        if received > source.bytes {
            return Err(DepthError::ModelTooLong {
                origin: source.url.clone(),
                expected: source.bytes,
            });
        }
        hasher.update(&chunk);
        temp.write_all(&chunk)
            .map_err(|err| DepthError::io("write", temp.path(), err))?;
        progress(DepthProgress::Downloading {
            received,
            total: source.bytes,
        });
    }
    if received != source.bytes {
        return Err(DepthError::ModelLength {
            origin: source.url.clone(),
            expected: source.bytes,
            actual: received,
        });
    }
    check_hash(hasher, source, source.url.clone())?;
    temp.as_file()
        .sync_all()
        .map_err(|err| DepthError::io("sync", temp.path(), err))?;
    if cancel.load(Ordering::Relaxed) {
        return Err(DepthError::Cancelled);
    }
    Ok(temp)
}
