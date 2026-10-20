//! Background MP4 export through FFmpeg.

use std::any::Any;
use std::io::{self, Write};
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ExitStatus};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use burns::{DepthMap, GpuContext, MotionProject, RenderView, Renderer, Size, SourceImage};
use tempfile::NamedTempFile;

use crate::MediaError;
use crate::encoder::{self, Encoder};
use crate::output::Destination;
use crate::prepare::{check_inputs, renderer};

/// x264 constant rate factor used when the user does not choose one.
pub const DEFAULT_CRF: u8 = 20;

/// Largest valid x264 constant rate factor.
pub const MAX_CRF: u8 = 51;

/// How often the worker polls a finishing FFmpeg so cancellation can still kill it.
const EXIT_POLL: Duration = Duration::from_millis(5);

const PHASE_RUNNING: u8 = 0;
const PHASE_CANCELLED: u8 = 1;
const PHASE_PUBLISHING: u8 = 2;

/// Everything one MP4 export needs. Encoder settings stay here, out of the portable project.
#[derive(Debug, Clone)]
pub struct RenderRequest {
    pub project: MotionProject,
    pub image: Arc<SourceImage>,
    /// Required by the depth camera; ignored by every other camera.
    pub depth: Option<Arc<DepthMap>>,
    pub output: PathBuf,
    /// x264 constant rate factor, `0..=MAX_CRF`; [`DEFAULT_CRF`] unless the user adjusts it.
    pub crf: u8,
    /// Replace an existing file or symlink entry at `output` once the export succeeds.
    pub overwrite: bool,
}

/// Progress and outcome of a [`RenderJob`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderStatus {
    /// `completed` of `total` frames have been handed to the encoder. A cancelled job stays
    /// `Running` until its encoder is stopped and its temporary file is removed.
    Running { completed: u32, total: u32 },
    /// The finished MP4 has been published at `output`.
    Done { output: PathBuf },
    /// The export failed; any previous file at the output path is unchanged.
    Failed { message: String },
    /// The export was cancelled; any previous file at the output path is unchanged.
    Cancelled,
}

impl RenderStatus {
    /// Whether the job has stopped and cleaned up.
    pub fn is_finished(&self) -> bool {
        !matches!(self, Self::Running { .. })
    }
}

/// An MP4 export running on its own worker thread.
///
/// [`RenderJob::start`] preflights everything before any frame is rendered. Dropping the job
/// cancels it, kills its FFmpeg child, and joins the worker; use [`RenderJob::wait`] to block
/// until it finishes instead.
#[derive(Debug)]
pub struct RenderJob {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

#[derive(Debug)]
struct Shared {
    status: Mutex<RenderStatus>,
    phase: AtomicU8,
    /// The owned FFmpeg child, without its stdin, so cancellation can kill it while the worker
    /// is blocked writing a frame.
    child: Mutex<Child>,
}

impl Shared {
    fn status(&self) -> MutexGuard<'_, RenderStatus> {
        self.status.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn child(&self) -> MutexGuard<'_, Child> {
        self.child.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn cancelled(&self) -> bool {
        self.phase.load(Ordering::Acquire) == PHASE_CANCELLED
    }

    fn cancel(&self) {
        if self
            .phase
            .compare_exchange(
                PHASE_RUNNING,
                PHASE_CANCELLED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            // A reaped child is never signalled again; only our encoder is killed.
            let _ = self.child().kill();
        }
    }
}

impl RenderJob {
    /// Preflights `request` and starts exporting it on a new worker thread.
    ///
    /// Before any frame is produced this checks the CRF, the project (including frame count,
    /// frame rate, and temporal limits), even output dimensions, the depth map the depth camera
    /// needs, the output path, FFmpeg with libx264, and the GPU limits of the 2x supersampled
    /// final render; the source and depth are uploaded once here.
    pub fn start(context: GpuContext, request: RenderRequest) -> Result<Self, MediaError> {
        Self::start_with(&Encoder::system(), context, request)
    }

    fn start_with(
        encoder: &Encoder,
        context: GpuContext,
        request: RenderRequest,
    ) -> Result<Self, MediaError> {
        let RenderRequest {
            project,
            image,
            depth,
            output,
            crf,
            overwrite,
        } = request;
        if crf > MAX_CRF {
            return Err(MediaError::InvalidCrf { crf });
        }
        let depth = check_inputs(&project, depth.as_deref())?;
        let Size { width, height } = project.output;
        if width % 2 != 0 || height % 2 != 0 {
            return Err(MediaError::OddDimensions { width, height });
        }
        let destination = Destination::check(&output, overwrite)?;
        encoder.probe()?;
        let renderer = renderer(context, &project, &image, depth)?;

        let temp = destination.temp_file()?;
        let mut child = encoder.spawn(project.output, project.timeline, crf, temp.path())?;
        let stdin = child.stdin.take().expect("stdin is piped");
        let stderr = child.stderr.take().expect("stderr is piped");
        let total = project.timeline.frames;
        let shared = Arc::new(Shared {
            status: Mutex::new(RenderStatus::Running {
                completed: 0,
                total,
            }),
            phase: AtomicU8::new(PHASE_RUNNING),
            child: Mutex::new(child),
        });
        let mut export = Export {
            shared: Arc::clone(&shared),
            renderer,
            project,
            program: encoder.program().to_owned(),
            stdin: Some(stdin),
            stderr: None,
            temp: Some(temp),
            destination,
        };
        export.stderr = Some(
            encoder::drain(stderr)
                .map_err(|error| MediaError::io("start a thread for", encoder.program(), error))?,
        );
        let worker_shared = Arc::clone(&shared);
        // If spawning fails, the dropped closure drops `export`, which stops FFmpeg and removes
        // the temporary.
        let worker = thread::Builder::new()
            .name("burns export".to_owned())
            .spawn(move || run(worker_shared, export))
            .map_err(|error| MediaError::io("start the export thread for", &output, error))?;
        Ok(Self {
            shared,
            worker: Some(worker),
        })
    }

    /// Current progress or outcome.
    pub fn status(&self) -> RenderStatus {
        self.shared.status().clone()
    }

    /// Requests cancellation and kills the owned FFmpeg child, interrupting a blocked frame
    /// write. Returns immediately; the status becomes [`RenderStatus::Cancelled`] once the
    /// worker has cleaned up. Once atomic publication begins, completion wins.
    pub fn cancel(&self) {
        self.shared.cancel();
    }

    /// Blocks until the job finishes and returns its final status.
    pub fn wait(mut self) -> RenderStatus {
        self.join();
        self.status()
    }

    fn join(&mut self) {
        if let Some(worker) = self.worker.take()
            && let Err(panic) = worker.join()
        {
            *self.shared.status() = RenderStatus::Failed {
                message: panic_message("export worker panicked", &*panic),
            };
        }
    }
}

impl Drop for RenderJob {
    fn drop(&mut self) {
        if self.worker.is_some() {
            self.cancel();
            self.join();
        }
    }
}

/// Worker entry: runs the export and publishes the final status only after cleanup.
fn run(shared: Arc<Shared>, export: Export) {
    let outcome = panic::catch_unwind(AssertUnwindSafe(move || {
        let mut export = export;
        let result = export.run();
        let output = export.destination.path().to_owned();
        // Stop FFmpeg, join the stderr reader, and delete any temporary before reporting.
        drop(export);
        result.map(|()| output)
    }));
    let status = match outcome {
        Ok(Ok(output)) => RenderStatus::Done { output },
        Ok(Err(Stop::Cancelled)) => RenderStatus::Cancelled,
        Ok(Err(Stop::Failed(_))) if shared.cancelled() => RenderStatus::Cancelled,
        Ok(Err(Stop::Failed(error))) => RenderStatus::Failed {
            message: error.to_string(),
        },
        Err(panic) => RenderStatus::Failed {
            message: panic_message("export worker panicked", &*panic),
        },
    };
    *shared.status() = status;
}

fn panic_message(context: &str, panic: &(dyn Any + Send)) -> String {
    let detail = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic payload");
    format!("{context}: {detail}")
}

enum Stop {
    Cancelled,
    Failed(MediaError),
}

impl From<MediaError> for Stop {
    fn from(error: MediaError) -> Self {
        Self::Failed(error)
    }
}

/// State owned by the worker. Dropping it closes stdin, kills and reaps FFmpeg if it is still
/// running, joins the stderr reader, and deletes the temporary unless it was published.
struct Export {
    shared: Arc<Shared>,
    renderer: Renderer,
    project: MotionProject,
    program: PathBuf,
    stdin: Option<ChildStdin>,
    stderr: Option<JoinHandle<String>>,
    temp: Option<NamedTempFile>,
    destination: Destination,
}

impl Export {
    fn program(&self) -> &std::path::Path {
        &self.program
    }

    fn run(&mut self) -> Result<(), Stop> {
        let total = self.project.timeline.frames;
        let Size { width, height } = self.renderer.output_size();
        let row = width as usize * 4;
        for frame in 0..total {
            self.check_cancel()?;
            self.renderer
                .render(&self.project, frame, RenderView::Motion)
                .map_err(MediaError::from)?;
            let Self {
                renderer,
                stdin,
                shared,
                ..
            } = self;
            let stdin = stdin
                .as_mut()
                .expect("stdin stays open until every frame is written");
            let written = pollster::block_on(renderer.with_rgba8(|bytes, pitch| {
                if pitch == row {
                    if shared.cancelled() {
                        return Err(io::Error::other("export cancelled"));
                    }
                    return stdin.write_all(bytes);
                }
                for y in 0..height as usize {
                    if shared.cancelled() {
                        return Err(io::Error::other("export cancelled"));
                    }
                    stdin.write_all(&bytes[y * pitch..y * pitch + row])?;
                }
                Ok(())
            }))
            .map_err(MediaError::from)?;
            if let Err(error) = written {
                return Err(self.pipe_failed(error));
            }
            *self.shared.status() = RenderStatus::Running {
                completed: frame + 1,
                total,
            };
        }
        let mut stdin = self.stdin.take().expect("stdin is still open");
        if let Err(error) = stdin.flush() {
            return Err(self.pipe_failed(error));
        }
        // Closing stdin signals end of input.
        drop(stdin);
        let status = self.wait_for_exit()?;
        self.check_cancel()?;
        if !status.success() {
            return Err(Stop::Failed(MediaError::EncoderFailed {
                reason: format!("ffmpeg exited with {status}"),
                stderr: self.stderr_tail(),
            }));
        }
        // Linearize cancellation against publication, rather than checking a
        // flag and leaving a window in which a cancelled job could overwrite.
        if self
            .shared
            .phase
            .compare_exchange(
                PHASE_RUNNING,
                PHASE_PUBLISHING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return Err(Stop::Cancelled);
        }
        let temp = self.temp.take().expect("the temporary is published once");
        self.destination.publish(temp)?;
        Ok(())
    }

    fn check_cancel(&self) -> Result<(), Stop> {
        if self.shared.cancelled() {
            Err(Stop::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Writing a frame failed: cancellation (the killed child broke the pipe) or an encoder
    /// that stopped reading; the latter reports FFmpeg's exit status and stderr.
    fn pipe_failed(&mut self, error: io::Error) -> Stop {
        if self.shared.cancelled() {
            return Stop::Cancelled;
        }
        // Without stdin FFmpeg sees end of input instead of waiting for more frames.
        self.discard_stdin();
        let status = match self.wait_for_exit() {
            Ok(status) => status.to_string(),
            Err(stop) => return stop,
        };
        if self.shared.cancelled() {
            return Stop::Cancelled;
        }
        Stop::Failed(MediaError::EncoderFailed {
            reason: format!("ffmpeg stopped reading frames ({error}) and exited with {status}"),
            stderr: self.stderr_tail(),
        })
    }

    /// Polls for FFmpeg's exit without holding the child lock, so `cancel` can kill it.
    fn wait_for_exit(&self) -> Result<ExitStatus, Stop> {
        loop {
            match self.shared.child().try_wait() {
                Ok(Some(status)) => return Ok(status),
                Ok(None) => thread::sleep(EXIT_POLL),
                Err(error) => {
                    return Err(Stop::Failed(MediaError::io(
                        "wait for",
                        self.program(),
                        error,
                    )));
                }
            }
        }
    }

    /// FFmpeg's retained stderr; call only after it exited and closed the pipe.
    fn stderr_tail(&mut self) -> String {
        self.stderr
            .take()
            .map(|reader| {
                reader
                    .join()
                    .unwrap_or_else(|_| "(the stderr reader panicked)".to_owned())
            })
            .unwrap_or_default()
    }

    /// Closing the unbuffered pipe never flushes pixels into a stopped encoder.
    fn discard_stdin(&mut self) {
        drop(self.stdin.take());
    }
}

impl Drop for Export {
    fn drop(&mut self) {
        {
            let mut child = self.shared.child();
            // Both are no-ops once the child has been reaped. Killing first means closing
            // stdin never waits on a reader.
            let _ = child.kill();
            drop(self.stdin.take());
            let _ = child.wait();
        }
        if let Some(reader) = self.stderr.take() {
            let _ = reader.join();
        }
        // Dropping `temp` deletes the temporary unless it was published.
    }
}

/// Encoder failures injected through stand-in executables; integration tests cover real FFmpeg.
#[cfg(all(test, unix))]
mod tests {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;
    use std::sync::LazyLock;

    use burns::{FrameRate, Timeline, preset};

    use super::*;

    static CONTEXT: LazyLock<GpuContext> = LazyLock::new(|| {
        pollster::block_on(GpuContext::headless())
            .unwrap_or_else(|error| panic!("GPU tests need a real adapter: {error}"))
    });

    fn request(output: PathBuf, overwrite: bool) -> RenderRequest {
        let mut project = preset("kenburns").unwrap();
        project.output = Size::new(64, 36);
        project.timeline = Timeline::new(FrameRate::new(30, 1), 8).unwrap();
        let pixels = (0..64 * 36)
            .flat_map(|i| {
                if i % 9 == 0 {
                    [255; 4]
                } else {
                    [20, 40, 160, 255]
                }
            })
            .collect();
        RenderRequest {
            project,
            image: Arc::new(SourceImage::from_rgba8(Size::new(64, 36), pixels).unwrap()),
            depth: None,
            output,
            crf: DEFAULT_CRF,
            overwrite,
        }
    }

    fn script(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("fake-ffmpeg");
        fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn entries(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    #[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
    fn missing_encoder_is_refused_before_any_output() {
        let out = tempfile::tempdir().unwrap();
        let encoder = Encoder::at(out.path().join("no-such-ffmpeg"));
        let error = RenderJob::start_with(
            &encoder,
            CONTEXT.clone(),
            request(out.path().join("out.mp4"), false),
        )
        .unwrap_err();
        assert!(
            matches!(error, MediaError::EncoderUnavailable { .. }),
            "{error}"
        );
        assert!(entries(out.path()).is_empty());
    }

    #[test]
    #[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
    fn encoder_without_libx264_is_refused() {
        let bin = tempfile::tempdir().unwrap();
        let out = tempfile::tempdir().unwrap();
        let encoder = Encoder::at(script(
            bin.path(),
            r#"echo "Encoders:"; echo " V....D libx265              libx265 H.265 / HEVC"; exit 0"#,
        ));
        let error = RenderJob::start_with(
            &encoder,
            CONTEXT.clone(),
            request(out.path().join("out.mp4"), false),
        )
        .unwrap_err();
        let MediaError::EncoderUnavailable { reason, .. } = &error else {
            panic!("expected EncoderUnavailable, got {error}");
        };
        assert!(reason.contains("libx264"), "{reason}");
        assert!(entries(out.path()).is_empty());
    }

    #[test]
    #[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
    fn failing_encoder_reports_stderr_and_keeps_previous_output() {
        let bin = tempfile::tempdir().unwrap();
        let out = tempfile::tempdir().unwrap();
        let encoder = Encoder::at(script(
            bin.path(),
            r#"case "$*" in *-encoders*) echo " V....D libx264              libx264 H.264"; exit 0;; esac
echo "synthetic encoder failure" >&2
exit 3"#,
        ));
        let output = out.path().join("out.mp4");
        fs::write(&output, b"previous export").unwrap();
        let job = RenderJob::start_with(&encoder, CONTEXT.clone(), request(output.clone(), true))
            .unwrap();
        let RenderStatus::Failed { message } = job.wait() else {
            panic!("a failing encoder must not report success or cancellation");
        };
        assert!(message.contains("synthetic encoder failure"), "{message}");
        assert!(message.contains("exit status: 3"), "{message}");
        assert_eq!(fs::read(&output).unwrap(), b"previous export");
        assert_eq!(entries(out.path()), ["out.mp4"]);
    }
}
