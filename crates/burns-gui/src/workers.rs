//! Background work owned by the workbench. Decoding, file I/O, depth lookup and generation,
//! PNG capture, and MP4 export never run on the UI thread. Every worker owns a cancellation
//! flag and is joined when the workbench shuts down.

use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use burns::{DepthMap, GpuContext, MotionProject, SourceImage};
use burns_depth::{DepthError, DepthProgress, DepthService};
use burns_media::{FrameRequest, MediaError, RenderJob, RenderRequest, RenderStatus};
use eframe::egui;

/// How often the export thread polls its job.
const EXPORT_POLL: Duration = Duration::from_millis(40);

/// Identifies one depth request for one source image. Results whose ticket is no longer
/// current (a newer image, import, lookup, or generation was requested) are discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthTicket {
    pub image: u64,
    pub request: u64,
}

/// Results delivered to the UI thread.
pub enum Event {
    Image {
        ticket: u64,
        path: PathBuf,
        result: Result<SourceImage, String>,
    },
    DepthFile {
        ticket: DepthTicket,
        path: PathBuf,
        result: Result<DepthMap, String>,
    },
    DepthLookup {
        ticket: DepthTicket,
        result: Result<Option<DepthMap>, String>,
    },
    DepthProgress {
        ticket: DepthTicket,
        progress: DepthProgress,
    },
    DepthGenerated {
        ticket: DepthTicket,
        outcome: Generated,
    },
    Project {
        path: PathBuf,
        result: Result<MotionProject, String>,
    },
    Saved {
        what: Saved,
        path: PathBuf,
        result: Result<(), String>,
    },
}

pub enum Generated {
    Map(DepthMap),
    Cancelled,
    Failed(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Saved {
    Project,
    Depth,
}

/// Sends results to the UI thread and wakes it.
#[derive(Clone)]
struct Outbox {
    sender: Sender<Event>,
    ctx: egui::Context,
}

impl Outbox {
    fn send(&self, event: Event) {
        // The receiver only disappears during shutdown, when results are moot.
        if self.sender.send(event).is_ok() {
            self.ctx.request_repaint();
        }
    }
}

/// A thread serving requests in order until its sender is dropped.
struct Queue<R> {
    sender: Option<Sender<R>>,
    handle: Option<JoinHandle<()>>,
}

impl<R: Send + 'static> Queue<R> {
    fn spawn(name: &str, mut serve: impl FnMut(R) + Send + 'static) -> io::Result<Self> {
        let (sender, receiver) = mpsc::channel::<R>();
        let handle = thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                for request in receiver {
                    serve(request);
                }
            })?;
        Ok(Self {
            sender: Some(sender),
            handle: Some(handle),
        })
    }

    fn send(&self, request: R) -> Result<(), String> {
        self.sender
            .as_ref()
            .and_then(|sender| sender.send(request).ok())
            .ok_or_else(|| "the background worker has stopped; restart burns".to_owned())
    }

    /// Closes the queue and waits for the thread to drain it.
    fn join(&mut self) {
        self.sender = None;
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl<R> Drop for Queue<R> {
    fn drop(&mut self) {
        self.sender = None;
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub enum IoRequest {
    LoadImage {
        ticket: u64,
        path: PathBuf,
    },
    LoadDepth {
        ticket: DepthTicket,
        path: PathBuf,
    },
    LoadProject {
        path: PathBuf,
    },
    /// Writes validated project JSON; the save dialog already confirmed any replacement.
    SaveProject {
        path: PathBuf,
        json: String,
    },
    /// Writes the raw map as a 16-bit PNG; the save dialog already confirmed any replacement.
    SaveDepth {
        path: PathBuf,
        depth: Arc<DepthMap>,
    },
}

enum DepthRequest {
    Lookup {
        ticket: DepthTicket,
        image: Arc<SourceImage>,
    },
    Generate {
        ticket: DepthTicket,
        image: Arc<SourceImage>,
        cancel: Arc<AtomicBool>,
    },
}

/// The asset I/O thread and the depth thread, with their shared result channel.
pub struct Workers {
    io: Queue<IoRequest>,
    depth: Queue<DepthRequest>,
    events: Receiver<Event>,
    /// Stops queued I/O and the depth file write at shutdown.
    shutdown: Arc<AtomicBool>,
    /// Cancellation flag of the newest depth generation.
    generation: Option<Arc<AtomicBool>>,
}

impl Workers {
    pub fn spawn(ctx: &egui::Context) -> io::Result<Self> {
        let (sender, events) = mpsc::channel();
        let outbox = Outbox {
            sender,
            ctx: ctx.clone(),
        };
        let shutdown = Arc::new(AtomicBool::new(false));

        let io_outbox = outbox.clone();
        let io_shutdown = Arc::clone(&shutdown);
        let io = Queue::spawn("burns assets", move |request| {
            if !io_shutdown.load(Ordering::Acquire) {
                serve_io(&io_outbox, &io_shutdown, request);
            }
        })?;

        // The service is created on first use; it only resolves the platform cache directory.
        let mut service: Option<Result<DepthService, String>> = None;
        let depth = Queue::spawn("burns depth", move |request| {
            let service = service
                .get_or_insert_with(|| DepthService::new().map_err(|error| error.to_string()));
            serve_depth(&outbox, service, request);
        })?;

        Ok(Self {
            io,
            depth,
            events,
            shutdown,
            generation: None,
        })
    }

    pub fn io(&self, request: IoRequest) -> Result<(), String> {
        self.io.send(request)
    }

    /// Looks up the cached generated map for `image`; never loads a model or downloads.
    pub fn lookup_depth(&self, ticket: DepthTicket, image: Arc<SourceImage>) -> Result<(), String> {
        self.depth.send(DepthRequest::Lookup { ticket, image })
    }

    /// Explicitly generates depth for `image`, which may download the pinned model.
    pub fn generate_depth(
        &mut self,
        ticket: DepthTicket,
        image: Arc<SourceImage>,
    ) -> Result<(), String> {
        self.cancel_generation();
        let cancel = Arc::new(AtomicBool::new(false));
        self.generation = Some(Arc::clone(&cancel));
        self.depth.send(DepthRequest::Generate {
            ticket,
            image,
            cancel,
        })
    }

    /// Cancels the running or queued generation, if any. Inference itself cannot be
    /// interrupted; its result is then discarded instead of cached.
    pub fn cancel_generation(&mut self) {
        if let Some(cancel) = self.generation.take() {
            cancel.store(true, Ordering::Release);
        }
    }

    pub fn try_recv(&self) -> Option<Event> {
        self.events.try_recv().ok()
    }

    /// Cancels pending work and joins both threads.
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.cancel_generation();
        self.io.join();
        self.depth.join();
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn serve_io(outbox: &Outbox, shutdown: &AtomicBool, request: IoRequest) {
    let event = match request {
        IoRequest::LoadImage { ticket, path } => Event::Image {
            ticket,
            result: burns_media::load_image(&path).map_err(|error| error.to_string()),
            path,
        },
        IoRequest::LoadDepth { ticket, path } => Event::DepthFile {
            ticket,
            result: burns_media::load_depth(&path).map_err(|error| error.to_string()),
            path,
        },
        IoRequest::LoadProject { path } => Event::Project {
            result: std::fs::read_to_string(&path)
                .map_err(|error| format!("could not read `{}`: {error}", path.display()))
                .and_then(|json| {
                    MotionProject::from_json(&json).map_err(|error| error.to_string())
                }),
            path,
        },
        IoRequest::SaveProject { path, json } => Event::Saved {
            what: Saved::Project,
            result: burns_media::safe_write(&path, true, |file| {
                use std::io::Write;
                file.write_all(json.as_bytes())
                    .and_then(|()| file.write_all(b"\n"))
                    .map_err(|error| MediaError::Io {
                        action: "write project",
                        path: path.clone(),
                        source: error,
                    })
            })
            .map_err(|error| error.to_string()),
            path,
        },
        IoRequest::SaveDepth { path, depth } => Event::Saved {
            what: Saved::Depth,
            result: burns_media::write_depth_png(&depth, &path, true, Some(shutdown))
                .map_err(|error| error.to_string()),
            path,
        },
    };
    outbox.send(event);
}

fn serve_depth(outbox: &Outbox, service: &mut Result<DepthService, String>, request: DepthRequest) {
    let unavailable = |message: &String| format!("the depth cache is unavailable: {message}");
    match request {
        DepthRequest::Lookup { ticket, image } => {
            let result = match service {
                Ok(service) => service.cached(&image).map_err(|error| error.to_string()),
                Err(message) => Err(unavailable(message)),
            };
            outbox.send(Event::DepthLookup { ticket, result });
        }
        DepthRequest::Generate {
            ticket,
            image,
            cancel,
        } => {
            let outcome = match service {
                Err(message) => Generated::Failed(unavailable(message)),
                Ok(_) if cancel.load(Ordering::Acquire) => Generated::Cancelled,
                Ok(service) => {
                    let progress =
                        |progress| outbox.send(Event::DepthProgress { ticket, progress });
                    match service.generate(&image, &cancel, progress) {
                        Ok(_) if cancel.load(Ordering::Acquire) => Generated::Cancelled,
                        Ok(map) => Generated::Map(map),
                        Err(DepthError::Cancelled) => Generated::Cancelled,
                        Err(error) => Generated::Failed(error.to_string()),
                    }
                }
            };
            outbox.send(Event::DepthGenerated { ticket, outcome });
        }
    }
}

/// A single final-quality PNG frame rendered on its own thread from snapshotted inputs.
pub struct Capture {
    pub frame: u32,
    pub output: PathBuf,
    cancel: Arc<AtomicBool>,
    handle: Option<JoinHandle<Result<(), MediaError>>>,
}

pub struct CaptureInputs {
    pub gpu: GpuContext,
    pub project: MotionProject,
    pub image: Arc<SourceImage>,
    pub depth: Option<Arc<DepthMap>>,
    pub frame: u32,
    pub output: PathBuf,
    pub overwrite: bool,
}

impl Capture {
    pub fn start(ctx: &egui::Context, inputs: CaptureInputs) -> io::Result<Self> {
        let cancel = Arc::new(AtomicBool::new(false));
        let frame = inputs.frame;
        let output = inputs.output.clone();
        let flag = Arc::clone(&cancel);
        let ctx = ctx.clone();
        let handle = thread::Builder::new()
            .name("burns frame".to_owned())
            .spawn(move || {
                let CaptureInputs {
                    gpu,
                    project,
                    image,
                    depth,
                    frame,
                    output,
                    overwrite,
                } = inputs;
                let result = burns_media::save_frame(
                    gpu,
                    FrameRequest {
                        project: &project,
                        image: &image,
                        depth: depth.as_deref(),
                        frame,
                        output: &output,
                        overwrite,
                        cancel: Some(&flag),
                    },
                );
                ctx.request_repaint();
                result
            })?;
        Ok(Self {
            frame,
            output,
            cancel,
            handle: Some(handle),
        })
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    pub fn is_cancelling(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }

    pub fn is_finished(&self) -> bool {
        self.handle.as_ref().is_none_or(JoinHandle::is_finished)
    }

    /// Joins the finished thread.
    pub fn finish(mut self) -> Result<(), MediaError> {
        match self.handle.take().map(JoinHandle::join) {
            Some(Ok(result)) => result,
            Some(Err(_)) => Err(MediaError::Io {
                action: "render frame",
                path: self.output.clone(),
                source: io::Error::other("the frame thread panicked"),
            }),
            None => Ok(()),
        }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.cancel();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// What the export panel shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportState {
    /// Preflight: checking FFmpeg, the output path, and GPU limits; uploading assets.
    Preparing,
    Running {
        completed: u32,
        total: u32,
    },
    Done {
        output: PathBuf,
    },
    Failed {
        message: String,
    },
    Cancelled,
}

impl ExportState {
    pub fn is_finished(&self) -> bool {
        matches!(
            self,
            Self::Done { .. } | Self::Failed { .. } | Self::Cancelled
        )
    }
}

struct ExportShared {
    cancel: AtomicBool,
    state: Mutex<ExportState>,
}

impl ExportShared {
    fn set(&self, state: ExportState) {
        *self.state.lock().unwrap_or_else(PoisonError::into_inner) = state;
    }
}

/// An MP4 export: a thread that starts a burns-media [`RenderJob`] from a snapshotted request
/// and mirrors its status. Dropping it cancels the job, which kills its FFmpeg child, and joins.
pub struct Export {
    pub output: PathBuf,
    shared: Arc<ExportShared>,
    handle: Option<JoinHandle<()>>,
}

impl Export {
    pub fn start(ctx: &egui::Context, gpu: GpuContext, request: RenderRequest) -> io::Result<Self> {
        let output = request.output.clone();
        let shared = Arc::new(ExportShared {
            cancel: AtomicBool::new(false),
            state: Mutex::new(ExportState::Preparing),
        });
        let worker = Arc::clone(&shared);
        let ctx = ctx.clone();
        let handle = thread::Builder::new()
            .name("burns movie".to_owned())
            .spawn(move || run_export(&ctx, &worker, gpu, request))?;
        Ok(Self {
            output,
            shared,
            handle: Some(handle),
        })
    }

    pub fn state(&self) -> ExportState {
        self.shared
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    pub fn cancel(&self) {
        self.shared.cancel.store(true, Ordering::Release);
    }

    pub fn is_cancelling(&self) -> bool {
        self.shared.cancel.load(Ordering::Acquire)
    }
}

impl Drop for Export {
    fn drop(&mut self) {
        self.cancel();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn run_export(ctx: &egui::Context, shared: &ExportShared, gpu: GpuContext, request: RenderRequest) {
    let job = match RenderJob::start(gpu, request) {
        Ok(job) => job,
        Err(error) => {
            shared.set(ExportState::Failed {
                message: error.to_string(),
            });
            ctx.request_repaint();
            return;
        }
    };
    loop {
        if shared.cancel.load(Ordering::Acquire) {
            job.cancel();
        }
        let state = match job.status() {
            RenderStatus::Running { completed, total } => ExportState::Running { completed, total },
            RenderStatus::Done { output } => ExportState::Done { output },
            RenderStatus::Failed { message } => ExportState::Failed { message },
            RenderStatus::Cancelled => ExportState::Cancelled,
        };
        let finished = state.is_finished();
        shared.set(state);
        ctx.request_repaint();
        if finished {
            break;
        }
        thread::sleep(EXPORT_POLL);
    }
    // Joins the finished job's worker.
    drop(job);
}
