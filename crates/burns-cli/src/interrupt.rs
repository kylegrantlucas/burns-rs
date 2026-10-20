//! Ctrl-C as a cancellation request instead of process termination.

use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::CliError;

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// The process's single Ctrl-C handler, installed on first use; it only sets [`INTERRUPTED`].
static HANDLER: LazyLock<Result<(), String>> = LazyLock::new(|| {
    ctrlc::set_handler(|| INTERRUPTED.store(true, Ordering::SeqCst))
        .map_err(|error| error.to_string())
});

/// Installs the Ctrl-C handler (once per process) and returns the flag it sets. Long-running
/// commands poll the flag and cancel their work cleanly, so FFmpeg children are killed and
/// temporary outputs removed rather than the process dying mid-write.
pub fn watch() -> Result<&'static AtomicBool, CliError> {
    HANDLER.clone().map_err(CliError::Signal)?;
    Ok(&INTERRUPTED)
}

/// Fails with [`CliError::Interrupted`] once Ctrl-C has been pressed.
pub fn check(flag: &AtomicBool) -> Result<(), CliError> {
    if flag.load(Ordering::SeqCst) {
        Err(CliError::Interrupted)
    } else {
        Ok(())
    }
}
