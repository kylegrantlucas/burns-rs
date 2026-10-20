//! Locating and initializing the dynamically loaded ONNX Runtime, once per process.

use std::{
    env,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    sync::{Mutex, PoisonError},
};

use ort::environment::Environment;

use crate::DepthError;

/// File name of the ONNX Runtime shared library on this platform.
#[cfg(target_os = "macos")]
pub const ORT_LIBRARY_NAME: &str = "libonnxruntime.dylib";
/// File name of the ONNX Runtime shared library on this platform.
#[cfg(target_os = "windows")]
pub const ORT_LIBRARY_NAME: &str = "onnxruntime.dll";
/// File name of the ONNX Runtime shared library on this platform.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub const ORT_LIBRARY_NAME: &str = "libonnxruntime.so";

/// Finds the ONNX Runtime shared library without loading it.
///
/// A non-empty `ORT_DYLIB_PATH` is authoritative and must name an existing file. Otherwise a
/// runtime packaged beside the executable is used when present: `<exe dir>/runtime/lib/` or
/// `<exe dir>/../lib/`, each containing [`ORT_LIBRARY_NAME`]. Nothing is installed or downloaded.
pub fn resolve_ort_library() -> Result<PathBuf, DepthError> {
    if let Some(path) = env::var_os("ORT_DYLIB_PATH").filter(|path| !path.is_empty()) {
        let path = PathBuf::from(path);
        return if path.is_file() {
            Ok(path)
        } else {
            Err(DepthError::RuntimeMissing {
                tried: format!("ORT_DYLIB_PATH `{}` is not a file", path.display()),
            })
        };
    }
    let exe_dir = env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    let candidates: Vec<PathBuf> = exe_dir
        .iter()
        .flat_map(|dir| {
            [
                dir.join("runtime").join("lib").join(ORT_LIBRARY_NAME),
                dir.join("..").join("lib").join(ORT_LIBRARY_NAME),
            ]
        })
        .collect();
    if let Some(found) = candidates.iter().find(|path| path.is_file()) {
        return Ok(found.clone());
    }
    let mut tried = String::from("ORT_DYLIB_PATH is unset");
    for candidate in &candidates {
        tried.push_str(&format!("; no `{}`", candidate.display()));
    }
    Err(DepthError::RuntimeMissing { tried })
}

/// Process-wide ONNX Runtime state. `ort` keeps the first loaded library for the life of the
/// process and allows a single environment, so both are recorded here and reused.
struct Runtime {
    library: Option<PathBuf>,
    environment: Option<Environment>,
}

static RUNTIME: Mutex<Runtime> = Mutex::new(Runtime {
    library: None,
    environment: None,
});

fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// Loads `library` (once per process) and returns the shared environment with telemetry off.
/// Requesting a different library after one was loaded is an error, never a silent reuse.
pub(crate) fn environment(library: &Path) -> Result<Environment, DepthError> {
    let mut runtime = RUNTIME.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(loaded) = &runtime.library
        && !same_file(loaded, library)
    {
        return Err(DepthError::RuntimeMismatch {
            loaded: loaded.clone(),
            requested: library.to_path_buf(),
        });
    }
    if let Some(environment) = &runtime.environment {
        return Ok(environment.clone());
    }
    let init_error = |message: String| DepthError::RuntimeInit {
        path: library.to_path_buf(),
        message,
    };
    // `ort` asserts on some malformed libraries; surface those as errors rather than panics.
    let builder = catch_unwind(AssertUnwindSafe(|| ort::init_from(library)))
        .map_err(|panic| init_error(panic_message(&panic)))?
        .map_err(|err| init_error(err.to_string()))?;
    runtime.library = Some(library.to_path_buf());
    let environment = catch_unwind(AssertUnwindSafe(|| {
        builder
            .with_name("burns-depth")
            .with_telemetry(false)
            .build()
    }))
    .map_err(|panic| init_error(panic_message(&panic)))?
    .map_err(|err| init_error(err.to_string()))?;
    runtime.environment = Some(environment.clone());
    Ok(environment)
}

fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_else(|| "ONNX Runtime initialization panicked".to_owned())
}
