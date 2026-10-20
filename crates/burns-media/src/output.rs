//! Output publishing through an owned temporary file in the destination directory.

use std::ffi::OsString;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use tempfile::NamedTempFile;

use crate::MediaError;

/// Writes `output` through a uniquely named temporary file in its directory, then publishes it.
///
/// Without `overwrite`, any existing entry at `output` (a file, directory, or symlink, even a
/// dangling one) is refused before `write` runs, and an entry that appears while `write` runs is
/// never replaced: [`MediaError::OutputExists`] is returned and the temporary is removed. With
/// `overwrite`, publishing atomically renames over the existing file or symlink entry without
/// following the link; directories are refused.
///
/// The destination is untouched until `write` succeeds and the data is synced. When `write` fails
/// or publishing is refused, the temporary is deleted and nothing else changes.
pub fn safe_write(
    output: &Path,
    overwrite: bool,
    write: impl FnOnce(&mut File) -> Result<(), MediaError>,
) -> Result<(), MediaError> {
    let destination = Destination::check(output, overwrite)?;
    let mut temp = destination.temp_file()?;
    write(temp.as_file_mut())?;
    destination.publish(temp)
}

/// A checked output path: its directory exists and the entry is free (or replaceable).
#[derive(Debug)]
pub(crate) struct Destination {
    path: PathBuf,
    directory: PathBuf,
    overwrite: bool,
}

impl Destination {
    pub(crate) fn check(path: &Path, overwrite: bool) -> Result<Self, MediaError> {
        let invalid = |reason| MediaError::InvalidOutputPath {
            path: path.to_owned(),
            reason,
        };
        if path.as_os_str().is_empty() {
            return Err(invalid("the path is empty"));
        }
        if ends_with_separator(path) || path.file_name().is_none() {
            return Err(invalid("it must name a file, not a directory"));
        }
        let directory = match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent.to_owned(),
            _ => PathBuf::from("."),
        };
        let directory_error = |reason| MediaError::OutputDirectory {
            path: path.to_owned(),
            directory: directory.clone(),
            reason,
        };
        match fs::metadata(&directory) {
            Ok(meta) if meta.is_dir() => {}
            Ok(_) => return Err(directory_error("is not a directory")),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Err(directory_error("does not exist; create it first"));
            }
            Err(error) => {
                return Err(MediaError::io(
                    "inspect output directory",
                    &directory,
                    error,
                ));
            }
        }
        // `symlink_metadata` sees the entry itself, so dangling symlinks count as existing.
        match fs::symlink_metadata(path) {
            Ok(_) if !overwrite => {
                return Err(MediaError::OutputExists {
                    path: path.to_owned(),
                });
            }
            Ok(meta) if meta.is_dir() => {
                return Err(MediaError::OutputIsDirectory {
                    path: path.to_owned(),
                });
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(MediaError::io("inspect output", path, error)),
        }
        Ok(Self {
            path: path.to_owned(),
            directory,
            overwrite,
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Creates `.NAME.XXXXXX.partial` next to the destination; it is deleted when dropped.
    pub(crate) fn temp_file(&self) -> Result<NamedTempFile, MediaError> {
        let mut prefix = OsString::from(".");
        prefix.push(self.path.file_name().expect("checked paths name a file"));
        prefix.push(".");
        let mut builder = tempfile::Builder::new();
        builder.prefix(&prefix).suffix(".partial");
        // Ordinary file permissions (the umask still applies) rather than owner-only.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            builder.permissions(fs::Permissions::from_mode(0o666));
        }
        builder
            .tempfile_in(&self.directory)
            .map_err(|error| MediaError::io("create a temporary file in", &self.directory, error))
    }

    /// Syncs `temp` and renames it to the destination: no-clobber unless overwriting.
    pub(crate) fn publish(&self, temp: NamedTempFile) -> Result<(), MediaError> {
        temp.as_file()
            .sync_all()
            .map_err(|error| MediaError::io("sync", temp.path(), error))?;
        let published = if self.overwrite {
            temp.persist(&self.path)
        } else {
            temp.persist_noclobber(&self.path)
        };
        // A failed persist hands back the temporary, which is deleted when dropped here.
        match published {
            Ok(_) => Ok(()),
            Err(error) if !self.overwrite && error.error.kind() == io::ErrorKind::AlreadyExists => {
                Err(MediaError::OutputExists {
                    path: self.path.clone(),
                })
            }
            Err(error) => Err(MediaError::io("publish", &self.path, error.error)),
        }
    }
}

/// Trailing separators or a trailing `.` component name a directory even when `file_name` is set.
fn ends_with_separator(path: &Path) -> bool {
    let bytes = path.as_os_str().as_encoded_bytes();
    let separator = |byte: u8| std::path::is_separator(char::from(byte));
    match bytes {
        [.., last] if separator(*last) => true,
        [.., before, b'.'] => separator(*before),
        _ => false,
    }
}
