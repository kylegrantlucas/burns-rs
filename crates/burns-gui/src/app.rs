//! The workbench window: toolbar, motion inspector, preview stage, transport, and export
//! inspector, plus the state that ties them to the background workers.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::task::Poll;
use std::time::{Duration, Instant};

use burns::presets::preset_motion;
use burns::{
    CameraMotion, ConfigError, DepthMap, FrameRate, GpuContext, MotionProject, RenderView, Size,
    SourceImage, Timeline, presets, setting_groups,
};
use burns_depth::DepthProgress;
use burns_media::{DEFAULT_CRF, MAX_CRF, MediaError, RenderRequest};
use eframe::egui::{
    self, Align, Align2, Color32, CornerRadius, DragValue, FontId, Key, Layout, Margin, Modifiers,
    Pos2, Rect, RichText, Stroke, TextEdit, UiBuilder, Vec2,
};

use crate::dialogs::{PendingDialog, Purpose};
use crate::inspector::{self, FieldError};
use crate::preview::Preview;
use crate::scrubber::scrubber;
use crate::style;
use crate::workers::{
    Capture, CaptureInputs, DepthTicket, Event, Export, ExportState, Generated, IoRequest, Saved,
    Workers,
};

/// Storage key of the portable project JSON.
const KEY_PROJECT: &str = "burns.project";
/// Storage key of the most recent source image path; never part of the project.
const KEY_RECENT_SOURCE: &str = "burns.recent-source";

/// File extensions refused as sources: burns animates one still image.
const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "m4v", "mov", "avi", "mkv", "webm", "mpg", "mpeg", "wmv", "flv", "gif", "apng",
];

/// Common frame rates offered next to the rational fields.
const FRAME_RATES: &[(&str, u32, u32)] = &[
    ("23.976", 24000, 1001),
    ("24", 24, 1),
    ("25", 25, 1),
    ("29.97", 30000, 1001),
    ("30", 30, 1),
    ("50", 50, 1),
    ("59.94", 60000, 1001),
    ("60", 60, 1),
];

/// Common output sizes; "match source" is added when an image is open.
const OUTPUT_SIZES: &[(&str, u32, u32)] = &[
    ("Full HD landscape", 1920, 1080),
    ("Full HD portrait", 1080, 1920),
    ("Square", 1080, 1080),
    ("4K UHD", 3840, 2160),
    ("HD 720p", 1280, 720),
];

struct Source {
    id: u64,
    path: PathBuf,
    name: String,
    image: Arc<SourceImage>,
}

#[derive(Clone)]
enum DepthOrigin {
    Cached,
    Generated,
    Imported(PathBuf),
}

struct Depth {
    map: Arc<DepthMap>,
    origin: DepthOrigin,
}

enum DepthTask {
    Idle,
    LookingUp,
    Loading(PathBuf),
    Generating { progress: Option<DepthProgress> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewMode {
    Motion,
    Source,
    Depth,
}

/// Playback runs from a wall-clock anchor; the shown frame is the nearest integer frame.
struct Playback {
    started: Instant,
    frame: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tone {
    Info,
    Ok,
    Error,
}

struct Note {
    tone: Tone,
    text: String,
}

impl Note {
    fn info(text: impl Into<String>) -> Option<Self> {
        Some(Self {
            tone: Tone::Info,
            text: text.into(),
        })
    }

    fn ok(text: impl Into<String>) -> Option<Self> {
        Some(Self {
            tone: Tone::Ok,
            text: text.into(),
        })
    }

    fn error(text: impl Into<String>) -> Option<Self> {
        Some(Self {
            tone: Tone::Error,
            text: text.into(),
        })
    }

    fn show(&self, ui: &mut egui::Ui) {
        let text = match self.tone {
            Tone::Info => style::muted(&self.text),
            Tone::Ok => style::ok(&self.text),
            Tone::Error => style::error(&self.text),
        };
        ui.add(egui::Label::new(text).wrap());
    }
}

/// A rejected project edit. Its message is shown only while the project is still in the state
/// it was rejected from, so a later change anywhere retires it.
struct Rejection {
    message: String,
    project: MotionProject,
}

impl Rejection {
    fn from_result(result: Result<(), ConfigError>, project: &MotionProject) -> Option<Self> {
        result.err().map(|error| Self {
            message: error.to_string(),
            project: project.clone(),
        })
    }

    fn current<'a>(rejection: &'a Option<Self>, project: &MotionProject) -> Option<&'a str> {
        rejection
            .as_ref()
            .filter(|rejection| rejection.project == *project)
            .map(|rejection| rejection.message.as_str())
    }
}

/// What the stage draws on top of the preview.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Overlay {
    None,
    /// Depth camera without a map: the unchanged source is shown with instructions.
    NeedsDepth,
    /// Depth view: legend and grid alignment.
    DepthLegend,
}

pub struct Workbench {
    ctx: egui::Context,
    gpu: GpuContext,
    preview: Preview,
    preview_error: Option<String>,
    workers: Workers,
    dialog: Option<PendingDialog>,

    project: MotionProject,
    field_error: Option<FieldError>,
    timing_error: Option<Rejection>,
    output_error: Option<Rejection>,
    /// Duration captured when a frame-rate edit gesture began, so dragging the rate never
    /// accumulates rounding.
    fps_edit_duration: Option<f64>,
    shown_timeline: Timeline,

    source: Option<Source>,
    image_ticket: u64,
    loading: Option<PathBuf>,
    recent_source: Option<PathBuf>,

    depth: Option<Depth>,
    depth_request: u64,
    depth_task: DepthTask,
    depth_note: Option<Note>,
    next_id: u64,

    view: ViewMode,
    frame: u32,
    playback: Option<Playback>,
    repeat: bool,

    crf: u8,
    movie_path: String,
    frame_path: String,
    overwrite: bool,
    export: Option<Export>,
    export_note: Option<Note>,
    capture: Option<Capture>,
    capture_note: Option<Note>,

    notice: Option<Note>,
}

impl Workbench {
    pub fn new(cc: &eframe::CreationContext<'_>, image: Option<PathBuf>) -> Result<Self, String> {
        style::apply(&cc.egui_ctx);
        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .ok_or("the wgpu renderer did not start, so there is no GPU to preview on")?;

        let mut notice = None;
        let project = match cc
            .storage
            .and_then(|storage| storage.get_string(KEY_PROJECT))
        {
            None => MotionProject::default(),
            Some(json) => MotionProject::from_json(&json).unwrap_or_else(|error| {
                notice = Note::error(format!(
                    "Saved settings could not be restored, so defaults are in use: {error}"
                ));
                MotionProject::default()
            }),
        };
        let recent_source = cc
            .storage
            .and_then(|storage| storage.get_string(KEY_RECENT_SOURCE))
            .filter(|path| !path.is_empty())
            .map(PathBuf::from);

        let preview = Preview::new(render_state, project.output)
            .map_err(|error| format!("could not start the GPU preview: {error}"))?;
        let gpu = preview.context().clone();
        let workers = Workers::spawn(&cc.egui_ctx)
            .map_err(|error| format!("could not start background workers: {error}"))?;

        let mut app = Self {
            ctx: cc.egui_ctx.clone(),
            gpu,
            preview,
            preview_error: None,
            workers,
            dialog: None,
            shown_timeline: project.timeline,
            project,
            field_error: None,
            timing_error: None,
            output_error: None,
            fps_edit_duration: None,
            source: None,
            image_ticket: 0,
            loading: None,
            recent_source: None,
            depth: None,
            depth_request: 0,
            depth_task: DepthTask::Idle,
            depth_note: None,
            next_id: 0,
            view: ViewMode::Motion,
            frame: 0,
            playback: None,
            repeat: true,
            crf: DEFAULT_CRF,
            movie_path: String::new(),
            frame_path: String::new(),
            overwrite: false,
            export: None,
            export_note: None,
            capture: None,
            capture_note: None,
            notice,
        };
        match image {
            Some(path) => app.open_image(path),
            None => match recent_source {
                Some(path) if path.is_file() => app.open_image(path),
                Some(path) if app.notice.is_none() => {
                    app.notice = Note::info(format!(
                        "The last image, {}, is no longer there. Open another to continue.",
                        path.display()
                    ));
                }
                _ => {}
            },
        }
        Ok(app)
    }

    fn next_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    // -----------------------------------------------------------------------------------------
    // Assets

    fn open_image(&mut self, path: PathBuf) {
        self.image_ticket += 1;
        match self.workers.io(IoRequest::LoadImage {
            ticket: self.image_ticket,
            path: path.clone(),
        }) {
            Ok(()) => self.loading = Some(path),
            Err(error) => self.notice = Note::error(error),
        }
    }

    /// Installs a decoded source. The project is kept; the previous image's depth map and any
    /// pending depth work are dropped, and the cache is checked for this image.
    fn install_source(&mut self, ticket: u64, path: PathBuf, image: SourceImage) {
        if let Err(error) = self.preview.set_source(ticket, &image) {
            self.notice = Note::error(format!("Could not show {}: {error}", display_name(&path)));
            return;
        }
        self.workers.cancel_generation();
        self.clear_depth();
        let image = Arc::new(image);
        self.source = Some(Source {
            id: ticket,
            name: display_name(&path),
            path: path.clone(),
            image: Arc::clone(&image),
        });
        self.recent_source = Some(path);
        self.notice = None;
        self.depth_task = DepthTask::LookingUp;
        let ticket = self.new_depth_ticket(ticket);
        if let Err(error) = self.workers.lookup_depth(ticket, image) {
            self.depth_task = DepthTask::Idle;
            self.depth_note = Note::error(error);
        }
    }

    fn clear_depth(&mut self) {
        self.depth = None;
        self.depth_note = None;
        self.depth_task = DepthTask::Idle;
        if let Err(error) = self.preview.set_depth(None) {
            self.preview_error = Some(error.to_string());
        }
    }

    /// A new depth request: every older lookup, import, or generation result becomes stale.
    fn new_depth_ticket(&mut self, image: u64) -> DepthTicket {
        self.depth_request += 1;
        DepthTicket {
            image,
            request: self.depth_request,
        }
    }

    fn is_current(&self, ticket: DepthTicket) -> bool {
        ticket.request == self.depth_request
            && self
                .source
                .as_ref()
                .is_some_and(|source| source.id == ticket.image)
    }

    fn install_depth(&mut self, map: DepthMap, origin: DepthOrigin) {
        let id = self.next_id();
        self.depth_task = DepthTask::Idle;
        match self.preview.set_depth(Some((id, &map))) {
            Ok(()) => {
                self.depth = Some(Depth {
                    map: Arc::new(map),
                    origin,
                });
                self.depth_note = None;
            }
            Err(error) => self.depth_note = Note::error(error.to_string()),
        }
    }

    fn load_depth(&mut self, path: PathBuf) {
        let Some(image) = self.source.as_ref().map(|source| source.id) else {
            return;
        };
        // An imported map wins over any lookup or generation still in flight.
        self.workers.cancel_generation();
        let ticket = self.new_depth_ticket(image);
        match self.workers.io(IoRequest::LoadDepth {
            ticket,
            path: path.clone(),
        }) {
            Ok(()) => {
                self.depth_task = DepthTask::Loading(path);
                self.depth_note = None;
            }
            Err(error) => self.depth_note = Note::error(error),
        }
    }

    fn generate_depth(&mut self) {
        let Some((id, image)) = self
            .source
            .as_ref()
            .map(|source| (source.id, Arc::clone(&source.image)))
        else {
            return;
        };
        let ticket = self.new_depth_ticket(id);
        match self.workers.generate_depth(ticket, image) {
            Ok(()) => {
                self.depth_task = DepthTask::Generating { progress: None };
                self.depth_note = None;
            }
            Err(error) => self.depth_note = Note::error(error),
        }
    }

    fn handle(&mut self, event: Event) {
        match event {
            Event::Image {
                ticket,
                path,
                result,
            } => {
                if ticket != self.image_ticket {
                    return;
                }
                self.loading = None;
                match result {
                    Ok(image) => self.install_source(ticket, path, image),
                    Err(error) => {
                        self.notice =
                            Note::error(format!("Could not open {}: {error}", display_name(&path)));
                    }
                }
            }
            Event::DepthFile {
                ticket,
                path,
                result,
            } => {
                if !self.is_current(ticket) {
                    return;
                }
                match result {
                    Ok(map) => self.install_depth(map, DepthOrigin::Imported(path)),
                    Err(error) => {
                        self.depth_task = DepthTask::Idle;
                        self.depth_note = Note::error(error);
                    }
                }
            }
            Event::DepthLookup { ticket, result } => {
                if !self.is_current(ticket) {
                    return;
                }
                match result {
                    Ok(Some(map)) => self.install_depth(map, DepthOrigin::Cached),
                    Ok(None) => self.depth_task = DepthTask::Idle,
                    Err(error) => {
                        self.depth_task = DepthTask::Idle;
                        self.depth_note = Note::error(error);
                    }
                }
            }
            Event::DepthProgress { ticket, progress } => {
                if !self.is_current(ticket) {
                    return;
                }
                if let DepthProgress::PersistenceWarning { message } = &progress {
                    self.depth_note = Note::error(message.clone());
                }
                if let DepthTask::Generating { progress: slot } = &mut self.depth_task {
                    *slot = Some(progress);
                }
            }
            Event::DepthGenerated { ticket, outcome } => {
                if !self.is_current(ticket) {
                    return;
                }
                self.depth_task = DepthTask::Idle;
                match outcome {
                    Generated::Map(map) => {
                        let warning = self.depth_note.take();
                        self.install_depth(map, DepthOrigin::Generated);
                        if self.depth_note.is_none() {
                            self.depth_note = warning;
                        }
                    }
                    Generated::Cancelled => {
                        self.depth_note = Note::info("Depth generation was cancelled.");
                    }
                    Generated::Failed(error) => self.depth_note = Note::error(error),
                }
            }
            Event::Project { path, result } => match result {
                Ok(project) => {
                    self.project = project;
                    self.notice = Note::ok(format!("Opened project {}.", display_name(&path)));
                }
                Err(error) => {
                    self.notice = Note::error(format!(
                        "Could not open project {}: {error}",
                        display_name(&path)
                    ));
                }
            },
            Event::Saved { what, path, result } => {
                let name = display_name(&path);
                let note = match (what, result) {
                    (Saved::Project, Ok(())) => Note::ok(format!("Saved project {name}.")),
                    (Saved::Depth, Ok(())) => Note::ok(format!("Saved raw depth map {name}.")),
                    (_, Err(error)) => Note::error(error),
                };
                match what {
                    Saved::Project => self.notice = note,
                    Saved::Depth => self.depth_note = note,
                }
            }
        }
    }

    fn open_dialog(&mut self, frame: &eframe::Frame, purpose: Purpose) {
        if self.dialog.is_some() {
            return;
        }
        let source_dir = self
            .source
            .as_ref()
            .and_then(|source| source.path.parent())
            .map(Path::to_path_buf);
        let stem = self.source.as_ref().and_then(|source| {
            source
                .path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
        });
        let existing = |text: &str| {
            let path = PathBuf::from(text.trim());
            (!text.trim().is_empty()).then_some(path)
        };
        let (directory, name) = match purpose {
            Purpose::OpenImage | Purpose::LoadProject | Purpose::LoadDepth => (source_dir, None),
            Purpose::SaveProject => (
                source_dir,
                Some(format!(
                    "{}.burns.json",
                    stem.as_deref().unwrap_or("motion")
                )),
            ),
            Purpose::SaveDepth => (
                source_dir,
                Some(format!("{}-depth.png", stem.as_deref().unwrap_or("image"))),
            ),
            Purpose::ChooseMovie | Purpose::ChooseFrame => {
                let current = existing(if purpose == Purpose::ChooseMovie {
                    &self.movie_path
                } else {
                    &self.frame_path
                });
                let extension = if purpose == Purpose::ChooseMovie {
                    "mp4"
                } else {
                    "png"
                };
                match current {
                    Some(path) => (
                        path.parent().map(Path::to_path_buf).or(source_dir),
                        path.file_name()
                            .map(|name| name.to_string_lossy().into_owned()),
                    ),
                    None => (
                        source_dir,
                        Some(format!(
                            "{}-burns.{extension}",
                            stem.as_deref().unwrap_or("motion")
                        )),
                    ),
                }
            }
        };
        self.dialog = Some(PendingDialog::open(
            &self.ctx, frame, purpose, directory, name,
        ));
    }

    fn poll_dialog(&mut self) {
        let Some(dialog) = &mut self.dialog else {
            return;
        };
        let Poll::Ready(path) = dialog.poll() else {
            return;
        };
        let purpose = dialog.purpose;
        self.dialog = None;
        let Some(path) = path else {
            return;
        };
        match purpose {
            Purpose::OpenImage => self.open_image(path),
            Purpose::LoadDepth => self.load_depth(path),
            Purpose::SaveDepth => {
                if let Some(depth) = &self.depth {
                    let path = with_extension(path, "png");
                    if let Err(error) = self.workers.io(IoRequest::SaveDepth {
                        path,
                        depth: Arc::clone(&depth.map),
                    }) {
                        self.depth_note = Note::error(error);
                    }
                }
            }
            Purpose::LoadProject => {
                if let Err(error) = self.workers.io(IoRequest::LoadProject { path }) {
                    self.notice = Note::error(error);
                }
            }
            Purpose::SaveProject => match self.project.to_json() {
                Ok(json) => {
                    let path = with_extension(path, "json");
                    if let Err(error) = self.workers.io(IoRequest::SaveProject { path, json }) {
                        self.notice = Note::error(error);
                    }
                }
                Err(error) => self.notice = Note::error(error.to_string()),
            },
            Purpose::ChooseMovie => {
                self.movie_path = with_extension(path, "mp4").display().to_string();
            }
            Purpose::ChooseFrame => {
                self.frame_path = with_extension(path, "png").display().to_string();
            }
        }
    }

    fn handle_drops(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|input| input.raw.dropped_files.clone());
        if dropped.is_empty() {
            return;
        }
        if dropped.len() > 1 {
            self.notice = Note::error(
                "Drop one image at a time. burns animates a single still image.".to_owned(),
            );
            return;
        }
        let path = dropped[0].path().to_path_buf();
        if path.as_os_str().is_empty() {
            self.notice = Note::error("The dropped item is not a file on disk.");
            return;
        }
        let extension = path
            .extension()
            .map(|extension| extension.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        if path.is_dir() {
            self.notice = Note::error(format!(
                "{} is a folder. Drop one PNG, JPEG, or WebP image.",
                display_name(&path)
            ));
        } else if VIDEO_EXTENSIONS.contains(&extension.as_str()) {
            self.notice = Note::error(format!(
                "{} is a video or animation. burns animates one still image: PNG, JPEG, or WebP.",
                display_name(&path)
            ));
        } else if extension == "json" {
            if let Err(error) = self.workers.io(IoRequest::LoadProject { path }) {
                self.notice = Note::error(error);
            }
        } else {
            self.open_image(path);
        }
    }

    // -----------------------------------------------------------------------------------------
    // Transport

    fn last_frame(&self) -> u32 {
        self.project.timeline.frames.saturating_sub(1)
    }

    fn toggle_playback(&mut self) {
        if self.playback.take().is_none() {
            if self.frame >= self.last_frame() && !self.repeat {
                self.frame = 0;
            }
            self.playback = Some(Playback {
                started: Instant::now(),
                frame: self.frame,
            });
        }
    }

    /// Sets the shown frame from a scrub or step; scrubbing always pauses.
    fn seek(&mut self, frame: u32) {
        self.playback = None;
        self.frame = frame.min(self.last_frame());
    }

    fn advance_playback(&mut self, ctx: &egui::Context) {
        let Some(playback) = &self.playback else {
            return;
        };
        let fps = self.project.timeline.fps;
        let frames = u64::from(self.project.timeline.frames);
        let elapsed = playback.started.elapsed().as_secs_f64();
        let advanced = (elapsed * f64::from(fps.num) / f64::from(fps.den)).round() as u64;
        let target = u64::from(playback.frame).saturating_add(advanced);
        if target < frames {
            self.frame = target as u32;
        } else if self.repeat {
            self.frame = (target % frames) as u32;
        } else {
            self.frame = self.last_frame();
            self.playback = None;
            return;
        }
        ctx.request_repaint();
    }

    /// Keeps the frame in range and playback anchored after any timeline change.
    fn sync_timeline(&mut self) {
        if self.project.timeline == self.shown_timeline {
            return;
        }
        self.shown_timeline = self.project.timeline;
        self.frame = self.frame.min(self.last_frame());
        if self.playback.is_some() {
            self.playback = Some(Playback {
                started: Instant::now(),
                frame: self.frame,
            });
        }
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        if ctx.text_edit_focused() {
            return;
        }
        let nothing_focused = ctx.memory(|memory| memory.focused().is_none());
        let (space, left, right, home, end) = ctx.input_mut(|input| {
            let space = input.consume_key(Modifiers::NONE, Key::Space);
            if !nothing_focused {
                return (space, false, false, false, false);
            }
            (
                space,
                input.consume_key(Modifiers::NONE, Key::ArrowLeft),
                input.consume_key(Modifiers::NONE, Key::ArrowRight),
                input.consume_key(Modifiers::NONE, Key::Home),
                input.consume_key(Modifiers::NONE, Key::End),
            )
        });
        if space {
            self.toggle_playback();
        }
        if left {
            self.seek(self.frame.saturating_sub(1));
        }
        if right {
            self.seek(self.frame.saturating_add(1));
        }
        if home {
            self.seek(0);
        }
        if end {
            self.seek(self.last_frame());
        }
    }

    /// Retimes to `fps`, keeping `duration` seconds: `Timeline::from_duration` at the final rate.
    fn retime(&mut self, fps: FrameRate, duration: f64) {
        let result = Timeline::from_duration(fps, duration)
            .map_err(|error| error.within("timeline"))
            .and_then(|timeline| self.project.set_timeline(timeline));
        self.timing_error = Rejection::from_result(result, &self.project);
    }

    // -----------------------------------------------------------------------------------------
    // Export

    fn start_export(&mut self) {
        let Some(source) = &self.source else {
            return;
        };
        let output = PathBuf::from(self.movie_path.trim());
        let request = RenderRequest {
            project: self.project.clone(),
            image: Arc::clone(&source.image),
            depth: self.depth.as_ref().map(|depth| Arc::clone(&depth.map)),
            output,
            crf: self.crf,
            overwrite: self.overwrite,
        };
        // A finished previous export is joined here.
        self.export = None;
        self.export_note = None;
        match Export::start(&self.ctx, self.gpu.clone(), request) {
            Ok(export) => self.export = Some(export),
            Err(error) => {
                self.export_note = Note::error(format!("Could not start the export: {error}"));
            }
        }
    }

    fn start_capture(&mut self) {
        let Some(source) = &self.source else {
            return;
        };
        let inputs = CaptureInputs {
            gpu: self.gpu.clone(),
            project: self.project.clone(),
            image: Arc::clone(&source.image),
            depth: self.depth.as_ref().map(|depth| Arc::clone(&depth.map)),
            frame: self.frame,
            output: PathBuf::from(self.frame_path.trim()),
            overwrite: self.overwrite,
        };
        self.capture_note = None;
        match Capture::start(&self.ctx, inputs) {
            Ok(capture) => self.capture = Some(capture),
            Err(error) => {
                self.capture_note = Note::error(format!("Could not start the capture: {error}"));
            }
        }
    }

    fn poll_capture(&mut self) {
        if !self.capture.as_ref().is_some_and(Capture::is_finished) {
            return;
        }
        let Some(capture) = self.capture.take() else {
            return;
        };
        let frame = capture.frame;
        let output = capture.output.clone();
        self.capture_note = match capture.finish() {
            Ok(()) => Note::ok(format!("Saved frame {frame} to {}.", output.display())),
            Err(MediaError::Cancelled) => {
                Note::info("Frame capture was cancelled. Nothing was written.")
            }
            Err(error) => Note::error(error.to_string()),
        };
    }

    /// Cancels every owned job, then joins them.
    fn shutdown(&mut self) {
        if let Some(export) = &self.export {
            export.cancel();
        }
        if let Some(capture) = &self.capture {
            capture.cancel();
        }
        self.workers.cancel_generation();
        self.export = None;
        self.capture = None;
        self.workers.shutdown();
    }

    // -----------------------------------------------------------------------------------------
    // UI regions

    fn toolbar(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        ui.horizontal_centered(|ui| {
            let idle = self.dialog.is_none();
            if ui
                .add_enabled(idle, egui::Button::new("Open image…"))
                .on_hover_text("Open one PNG, JPEG, or WebP still. You can also drop it here.")
                .clicked()
            {
                self.open_dialog(frame, Purpose::OpenImage);
            }
            match (&self.loading, &self.source) {
                (Some(path), _) => {
                    ui.spinner();
                    ui.label(style::muted(format!("Opening {}…", display_name(path))));
                }
                (None, Some(source)) => {
                    let size = source.image.size();
                    ui.label(RichText::new(&source.name).color(style::TEXT))
                        .on_hover_text(source.path.display().to_string());
                    ui.label(style::muted(format!("{} × {}", size.width, size.height)))
                        .on_hover_text("Source size after EXIF orientation");
                }
                (None, None) => {
                    ui.label(style::muted("No image open"));
                }
            }

            ui.add_space(12.0);
            ui.separator();
            let label = ui.label(style::muted("Preset")).id;
            self.preset_menu(ui, label);
            ui.separator();

            if ui
                .add_enabled(idle, egui::Button::new("Open project…"))
                .on_hover_text("Replace every setting, including timing, output size, and invert")
                .clicked()
            {
                self.open_dialog(frame, Purpose::LoadProject);
            }
            if ui
                .add_enabled(idle, egui::Button::new("Save project…"))
                .on_hover_text("Save the portable settings as JSON; the image path is not included")
                .clicked()
            {
                self.open_dialog(frame, Purpose::SaveProject);
            }
            if ui
                .button("Reset")
                .on_hover_text("Restore every setting to the defaults")
                .clicked()
            {
                self.project = MotionProject::default();
                self.notice = Note::info("Settings were reset to the defaults.");
            }
        });
    }

    fn preset_menu(&mut self, ui: &mut egui::Ui, label: egui::Id) {
        let current = presets().iter().find(|preset| {
            preset_motion(preset.name).is_ok_and(|motion| {
                motion.camera == self.project.camera
                    && motion.shake == self.project.shake
                    && motion.warp == self.project.warp
            })
        });
        let mut chosen = None;
        egui::ComboBox::from_id_salt("preset")
            .selected_text(current.map_or("Custom", |preset| preset.label))
            .width(190.0)
            .show_ui(ui, |ui| {
                for preset in presets() {
                    let label = if preset.requires_depth {
                        format!("{} (needs depth)", preset.label)
                    } else {
                        preset.label.to_owned()
                    };
                    if ui
                        .selectable_label(current == Some(preset), label)
                        .clicked()
                    {
                        chosen = Some(preset.name);
                    }
                }
            })
            .response
            .labelled_by(label)
            .on_hover_text("Presets replace camera, shake, and warp; timing and output stay");
        if let Some(name) = chosen
            && let Err(error) = self.project.apply_preset(name)
        {
            self.notice = Note::error(error.to_string());
        }
    }

    fn motion_inspector(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        for group in setting_groups() {
            inspector::group(ui, &mut self.project, &mut self.field_error, group);
            if group.id == "depth" {
                self.depth_controls(ui, frame);
            }
        }
        ui.add_space(12.0);
    }

    fn depth_status(&self) -> RichText {
        let image = self.source.as_ref().map(|source| source.image.size());
        match (&self.depth_task, &self.depth) {
            (DepthTask::Generating { progress }, _) => style::muted(match progress {
                None => "Starting depth generation…".to_owned(),
                Some(DepthProgress::CacheHit) => "Found a cached map.".to_owned(),
                Some(DepthProgress::Verifying) => "Verifying the depth model…".to_owned(),
                Some(DepthProgress::Downloading { received, total }) => format!(
                    "Downloading the depth model: {:.1} of {:.1} MB",
                    *received as f64 / 1e6,
                    *total as f64 / 1e6
                ),
                Some(DepthProgress::Loading) => "Loading ONNX Runtime and the model…".to_owned(),
                Some(DepthProgress::Inferring) => "Estimating depth…".to_owned(),
                Some(DepthProgress::Saving) => "Caching the map…".to_owned(),
                Some(DepthProgress::PersistenceWarning { .. }) => "Finishing…".to_owned(),
            }),
            (DepthTask::LookingUp, _) => style::muted("Checking the depth cache…"),
            (DepthTask::Loading(path), _) => {
                style::muted(format!("Loading {}…", display_name(path)))
            }
            (DepthTask::Idle, Some(depth)) => {
                let size = depth.map.size();
                let origin = match &depth.origin {
                    DepthOrigin::Cached => "Cached map for this image".to_owned(),
                    DepthOrigin::Generated => "Generated for this image".to_owned(),
                    DepthOrigin::Imported(path) => format!("Imported {}", display_name(path)),
                };
                RichText::new(format!("{origin}, {} × {}", size.width, size.height))
                    .color(style::TEXT)
            }
            (DepthTask::Idle, None) => style::muted(if image.is_some() {
                "No depth map yet. Load a grayscale PNG or generate one."
            } else {
                "Open an image to load or generate its depth map."
            }),
        }
    }

    fn depth_controls(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        let has_source = self.source.is_some();
        let generating = matches!(self.depth_task, DepthTask::Generating { .. });
        let busy = !matches!(self.depth_task, DepthTask::Idle);
        ui.horizontal(|ui| {
            if busy {
                ui.spinner();
            }
            ui.add(egui::Label::new(self.depth_status()).wrap());
        });
        ui.horizontal_wrapped(|ui| {
            let idle_dialog = self.dialog.is_none();
            if ui
                .add_enabled(has_source && idle_dialog, egui::Button::new("Load map…"))
                .on_hover_text("Use an 8- or 16-bit grayscale PNG where white is near")
                .clicked()
            {
                self.open_dialog(frame, Purpose::LoadDepth);
            }
            if generating {
                if ui.button("Cancel").clicked() {
                    self.workers.cancel_generation();
                    self.depth_note = Note::info("Cancelling after the current step…");
                }
            } else if ui
                .add_enabled(has_source, egui::Button::new("Generate"))
                .on_hover_text(
                    "Estimate depth with Depth Anything V2 Small on this computer. The first \
                     run downloads the 99 MB model; later runs reuse the cache.",
                )
                .clicked()
            {
                self.generate_depth();
            }
            if ui
                .add_enabled(
                    self.depth.is_some() && idle_dialog,
                    egui::Button::new("Save raw map…"),
                )
                .on_hover_text(
                    "Save the map exactly as loaded or generated, as a 16-bit PNG where white is \
                     near. Invert is not baked in: it stays in the project, so loading this file \
                     with the same project shows the same depth without inverting twice.",
                )
                .clicked()
            {
                self.open_dialog(frame, Purpose::SaveDepth);
            }
        });
        if let Some(note) = &self.depth_note {
            note.show(ui);
        }
    }

    fn transport(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let playing = self.playback.is_some();
            let label = if playing { "Pause" } else { "Play" };
            if ui
                .add_sized([84.0, 28.0], style::primary(label))
                .on_hover_text("Play or pause (Space)")
                .clicked()
            {
                self.toggle_playback();
            }
            ui.toggle_value(&mut self.repeat, "Repeat")
                .on_hover_text("Loop playback at the end of the timeline");
            ui.add_space(6.0);
            let frames = self.project.timeline.frames;
            let mut frame = self.frame;
            let enabled = frames > 1;
            let response = ui
                .add_enabled_ui(enabled, |ui| scrubber(ui, &mut frame, frames))
                .inner;
            if response.changed() || response.drag_started() {
                self.seek(frame);
            }
        });
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            let timeline = self.project.timeline;
            let last = self.last_frame();
            let label = ui.label(style::muted("Frame")).id;
            let mut frame = self.frame;
            if ui
                .add(
                    DragValue::new(&mut frame)
                        .range(0..=last)
                        .speed(0.25)
                        .update_while_editing(false),
                )
                .labelled_by(label)
                .on_hover_text("Frame index; the Left and Right arrow keys step one frame")
                .changed()
            {
                self.seek(frame);
            }
            ui.label(style::muted(format!("of 0–{last}")));
            let now = timeline
                .frame_time(self.frame.min(last))
                .map_or(0.0, |time| time.seconds);
            ui.label(
                RichText::new(format!(
                    "{} / {}",
                    format_time(now),
                    format_time(timeline.duration_seconds())
                ))
                .color(style::TEXT),
            );

            ui.separator();
            self.timing_fields(ui);

            if let Some(error) = Rejection::current(&self.timing_error, &self.project) {
                ui.label(style::error(error));
            }
        });
        if let Some(notice) = &self.notice {
            let mut dismiss = false;
            ui.horizontal(|ui| {
                dismiss = ui.small_button("Dismiss").clicked();
                notice.show(ui);
            });
            if dismiss {
                self.notice = None;
            }
        }
        ui.add_space(4.0);
    }

    fn timing_fields(&mut self, ui: &mut egui::Ui) {
        let timeline = self.project.timeline;
        let rate_label = ui.label(style::muted("Frame rate")).id;
        let mut num = timeline.fps.num;
        let mut den = timeline.fps.den;
        let numerator = ui
            .add(
                DragValue::new(&mut num)
                    .range(1..=u32::MAX)
                    .speed(0.05)
                    .update_while_editing(false),
            )
            .labelled_by(rate_label)
            .on_hover_text("Frames per second numerator, for example 30000 in 30000/1001");
        ui.label(style::muted("/"));
        let denominator = ui
            .add(
                DragValue::new(&mut den)
                    .range(1..=u32::MAX)
                    .speed(0.05)
                    .update_while_editing(false),
            )
            .on_hover_text("Frames per second denominator, for example 1001 in 30000/1001");
        if numerator.changed() || denominator.changed() {
            let duration = *self
                .fps_edit_duration
                .get_or_insert(timeline.duration_seconds());
            self.retime(FrameRate::new(num, den), duration);
        }
        let editing = [&numerator, &denominator]
            .iter()
            .any(|response| response.dragged() || response.has_focus());
        if !editing {
            self.fps_edit_duration = None;
        }

        let mut chosen = None;
        egui::ComboBox::from_id_salt("frame-rates")
            .selected_text(format!("{:.3} fps", timeline.fps.as_f64()))
            .width(96.0)
            .show_ui(ui, |ui| {
                for &(label, num, den) in FRAME_RATES {
                    let fps = FrameRate::new(num, den);
                    if ui
                        .selectable_label(timeline.fps == fps, format!("{label} ({num}/{den})"))
                        .clicked()
                    {
                        chosen = Some(fps);
                    }
                }
            })
            .response
            .on_hover_text("Common rates; changing the rate keeps the clip duration");
        if let Some(fps) = chosen {
            self.retime(fps, timeline.duration_seconds());
        }

        let label = ui.label(style::muted("Duration")).id;
        let mut seconds = timeline.duration_seconds();
        if ui
            .add(
                DragValue::new(&mut seconds)
                    .speed(0.05)
                    .max_decimals(3)
                    .suffix(" s")
                    .update_while_editing(false),
            )
            .labelled_by(label)
            .on_hover_text("Clip length; the frame count is rounded to the nearest frame")
            .changed()
        {
            self.retime(timeline.fps, seconds);
        }
        let label = ui.label(style::muted("Frames")).id;
        let mut frames = timeline.frames;
        if ui
            .add(
                DragValue::new(&mut frames)
                    .range(1..=u32::MAX)
                    .speed(0.5)
                    .update_while_editing(false),
            )
            .labelled_by(label)
            .on_hover_text("Total frame count at this rate")
            .changed()
        {
            let result = self.project.set_frames(frames);
            self.timing_error = Rejection::from_result(result, &self.project);
        }
    }

    fn export_inspector(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        style::section(ui, "Output size");
        let output = self.project.output;
        let mut width = output.width;
        let mut height = output.height;
        let mut changed = false;
        egui::Grid::new("output-size")
            .num_columns(2)
            .spacing([12.0, 7.0])
            .min_col_width(92.0)
            .show(ui, |ui| {
                let label = ui.label(style::muted("Width")).id;
                changed |= ui
                    .add(
                        DragValue::new(&mut width)
                            .range(1..=u32::MAX)
                            .speed(1.0)
                            .suffix(" px")
                            .update_while_editing(false),
                    )
                    .labelled_by(label)
                    .changed();
                ui.end_row();
                let label = ui.label(style::muted("Height")).id;
                changed |= ui
                    .add(
                        DragValue::new(&mut height)
                            .range(1..=u32::MAX)
                            .speed(1.0)
                            .suffix(" px")
                            .update_while_editing(false),
                    )
                    .labelled_by(label)
                    .changed();
                ui.end_row();
            });
        let mut chosen = None;
        let source_size = self.source.as_ref().map(|source| source.image.size());
        egui::ComboBox::from_id_salt("output-sizes")
            .selected_text("Common sizes")
            .width(200.0)
            .show_ui(ui, |ui| {
                for &(label, w, h) in OUTPUT_SIZES {
                    if ui
                        .selectable_label(false, format!("{label}, {w} × {h}"))
                        .clicked()
                    {
                        chosen = Some(Size::new(w, h));
                    }
                }
                if let Some(size) = source_size
                    && ui
                        .selectable_label(
                            false,
                            format!("Match source, {} × {}", size.width, size.height),
                        )
                        .clicked()
                {
                    chosen = Some(size);
                }
            });
        let target = chosen.or(changed.then_some(Size::new(width, height)));
        if let Some(size) = target {
            let result = self.project.set_output(size);
            self.output_error = Rejection::from_result(result, &self.project);
        }
        let preview = self.preview.output_size();
        ui.label(style::small(format!(
            "Preview renders at {} × {}. The image is cover-cropped to this aspect.",
            preview.width, preview.height
        )));
        if let Some(error) = Rejection::current(&self.output_error, &self.project) {
            ui.add(egui::Label::new(style::error(error)).wrap());
        }

        style::section(ui, "Movie");
        ui.label(style::small("H.264 MP4, encoded by FFmpeg"));
        egui::Grid::new("movie")
            .num_columns(2)
            .spacing([12.0, 7.0])
            .min_col_width(92.0)
            .show(ui, |ui| {
                let label = ui
                    .label(style::muted("Quality (CRF)"))
                    .on_hover_text("Lower is higher quality and larger files; 20 is the default")
                    .id;
                ui.add(egui::Slider::new(&mut self.crf, 0..=MAX_CRF).integer())
                    .labelled_by(label);
                ui.end_row();
            });
        self.path_row(ui, frame, Purpose::ChooseMovie);
        ui.checkbox(&mut self.overwrite, "Replace existing files")
            .on_hover_text("Off: rendering refuses to replace a file that already exists");

        let has_source = self.source.is_some();
        let state = self.export.as_ref().map(Export::state);
        let running = state.as_ref().is_some_and(|state| !state.is_finished());
        if running {
            let cancelling = self.export.as_ref().is_some_and(Export::is_cancelling);
            let (fraction, text) = match &state {
                Some(ExportState::Running { completed, total }) => (
                    *completed as f32 / (*total).max(1) as f32,
                    format!("Frame {completed} of {total}"),
                ),
                _ => (0.0, "Preparing…".to_owned()),
            };
            ui.add(egui::ProgressBar::new(fraction).text(text));
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!cancelling, egui::Button::new("Cancel render"))
                    .clicked()
                    && let Some(export) = &self.export
                {
                    export.cancel();
                }
                if cancelling {
                    ui.label(style::muted("Stopping FFmpeg…"));
                }
            });
            ui.label(style::small(
                "This render uses the settings from when it started. Edits apply to the next \
                 render.",
            ));
        } else {
            let ready = has_source && !self.movie_path.trim().is_empty();
            if ui
                .add_enabled(ready, style::primary("Render movie"))
                .on_disabled_hover_text(if has_source {
                    "Choose a file for the movie first"
                } else {
                    "Open an image first"
                })
                .clicked()
            {
                self.start_export();
            }
        }
        match (&state, &self.export) {
            (Some(ExportState::Done { output }), _) => {
                ui.add(egui::Label::new(style::ok(format!("Saved {}", output.display()))).wrap());
            }
            (Some(ExportState::Failed { message }), _) => {
                ui.add(egui::Label::new(style::error(message)).wrap());
            }
            (Some(ExportState::Cancelled), Some(export)) => {
                ui.add(
                    egui::Label::new(style::muted(format!(
                        "Render cancelled. {} was not written or changed.",
                        export.output.display()
                    )))
                    .wrap(),
                );
            }
            _ => {}
        }
        if let Some(note) = &self.export_note {
            note.show(ui);
        }

        style::section(ui, "Frame");
        ui.label(style::small(format!(
            "One full-quality PNG of frame {} at {} × {}",
            self.frame, self.project.output.width, self.project.output.height
        )));
        self.path_row(ui, frame, Purpose::ChooseFrame);
        if let Some(capture) = &self.capture {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(style::muted(format!("Saving frame {}…", capture.frame)));
                if ui
                    .add_enabled(!capture.is_cancelling(), egui::Button::new("Cancel"))
                    .clicked()
                {
                    capture.cancel();
                }
            });
        } else {
            let ready = has_source && !self.frame_path.trim().is_empty();
            if ui
                .add_enabled(
                    ready,
                    egui::Button::new(format!("Save frame {}", self.frame)),
                )
                .on_disabled_hover_text(if has_source {
                    "Choose a file for the frame first"
                } else {
                    "Open an image first"
                })
                .clicked()
            {
                self.start_capture();
            }
        }
        if let Some(note) = &self.capture_note {
            note.show(ui);
        }

        ui.add_space(14.0);
        ui.label(style::small(
            "Movies and frames always render the full-quality motion, whatever the preview \
             shows. Frames accept odd sizes; movies need an even width and height.",
        ));
    }

    fn path_row(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame, purpose: Purpose) {
        let idle = self.dialog.is_none();
        let mut choose = false;
        // Right to left: the button takes its size first and the field fills exactly what is
        // left, so the row never widens the panel.
        let row = Vec2::new(ui.available_width(), ui.spacing().interact_size.y);
        ui.allocate_ui_with_layout(row, Layout::right_to_left(Align::Center), |ui| {
            choose = ui.add_enabled(idle, egui::Button::new("Choose…")).clicked();
            let (text, hint) = match purpose {
                Purpose::ChooseMovie => (&mut self.movie_path, "Movie file path"),
                _ => (&mut self.frame_path, "Frame file path"),
            };
            let field = ui.add(
                TextEdit::singleline(text)
                    .hint_text(hint)
                    .desired_width(ui.available_width()),
            );
            ui.ctx()
                .accesskit_node_builder(field.id, |node| node.set_label(hint));
        });
        if choose {
            self.open_dialog(frame, purpose);
        }
    }

    // -----------------------------------------------------------------------------------------
    // Stage

    fn view_toggle(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            for (mode, label, help) in [
                (ViewMode::Motion, "Motion", "The animated result"),
                (
                    ViewMode::Source,
                    "Source",
                    "The cover-cropped source image, without motion",
                ),
                (
                    ViewMode::Depth,
                    "Depth",
                    "The effective depth map, unwarped: white is near",
                ),
            ] {
                if ui
                    .selectable_label(self.view == mode, label)
                    .on_hover_text(help)
                    .clicked()
                {
                    self.view = mode;
                }
            }
        });
    }

    fn plan(&self) -> (Option<RenderView>, Overlay) {
        match self.view {
            ViewMode::Motion => {
                if matches!(self.project.camera, CameraMotion::Depth { .. }) && self.depth.is_none()
                {
                    (Some(RenderView::Source), Overlay::NeedsDepth)
                } else {
                    (Some(RenderView::Motion), Overlay::None)
                }
            }
            ViewMode::Source => (Some(RenderView::Source), Overlay::None),
            ViewMode::Depth if self.depth.is_some() => {
                (Some(RenderView::Depth), Overlay::DepthLegend)
            }
            ViewMode::Depth => (None, Overlay::NeedsDepth),
        }
    }

    fn stage(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        let full = ui.max_rect();
        let bar = Rect::from_min_size(full.min, Vec2::new(full.width(), 40.0));
        ui.scope_builder(
            UiBuilder::new()
                .max_rect(bar.shrink2(Vec2::new(14.0, 8.0)))
                .layout(Layout::left_to_right(Align::Center)),
            |ui| {
                self.view_toggle(ui);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let preview = self.preview.output_size();
                    let output = self.project.output;
                    ui.label(style::small(format!(
                        "Preview {} × {} of {} × {}",
                        preview.width, preview.height, output.width, output.height
                    )));
                });
            },
        );
        let area = Rect::from_min_max(Pos2::new(full.left(), bar.bottom()), full.max)
            .shrink2(Vec2::new(18.0, 10.0));

        if self.source.is_none() {
            self.empty_state(ui, frame, area);
            self.drop_hint(ui, full);
            return;
        }

        let output = self.project.output;
        let aspect = output.width as f32 / output.height as f32;
        let fitted = fit(area, aspect);
        let (view, overlay) = self.plan();
        match view {
            Some(view) => match self.preview.render(&self.project, self.frame, view) {
                Ok(()) => {
                    self.preview_error = None;
                    ui.painter().image(
                        self.preview.texture(),
                        fitted,
                        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                        Color32::WHITE,
                    );
                }
                Err(error) => self.preview_error = Some(error.to_string()),
            },
            None => {
                ui.painter().rect_stroke(
                    fitted,
                    0.0,
                    Stroke::new(1.0, style::BORDER),
                    egui::StrokeKind::Inside,
                );
            }
        }
        if let Some(error) = &self.preview_error {
            ui.painter().text(
                fitted.center(),
                Align2::CENTER_CENTER,
                error,
                FontId::proportional(14.0),
                style::ERROR,
            );
        }
        match overlay {
            Overlay::None => {}
            Overlay::NeedsDepth => self.needs_depth_banner(ui, frame, fitted),
            Overlay::DepthLegend => self.depth_legend(ui, fitted),
        }
        self.drop_hint(ui, full);
    }

    fn empty_state(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame, area: Rect) {
        let card = Rect::from_center_size(area.center(), Vec2::new(420.0, 170.0));
        ui.scope_builder(
            UiBuilder::new()
                .max_rect(card)
                .layout(Layout::top_down(Align::Center)),
            |ui| {
                if let Some(path) = &self.loading {
                    ui.spinner();
                    ui.label(style::muted(format!("Opening {}…", display_name(path))));
                    return;
                }
                ui.label(
                    RichText::new("Animate one still image")
                        .size(22.0)
                        .color(style::TEXT),
                );
                ui.add_space(6.0);
                ui.label(style::muted(
                    "Open a PNG, JPEG, or WebP, or drop one anywhere in this window. Presets \
                     and settings are ready to explore now.",
                ));
                ui.add_space(12.0);
                if ui
                    .add_enabled(
                        self.dialog.is_none(),
                        style::primary("Open image…").min_size(Vec2::new(150.0, 32.0)),
                    )
                    .clicked()
                {
                    self.open_dialog(frame, Purpose::OpenImage);
                }
            },
        );
    }

    fn needs_depth_banner(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame, fitted: Rect) {
        let height = 92.0;
        let banner = Rect::from_min_max(
            Pos2::new(fitted.left(), (fitted.bottom() - height).max(fitted.top())),
            fitted.max,
        );
        ui.painter().rect_filled(
            banner,
            CornerRadius::ZERO,
            Color32::from_rgba_unmultiplied(0x2a, 0x27, 0x24, 236),
        );
        ui.scope_builder(
            UiBuilder::new()
                .max_rect(banner.shrink2(Vec2::new(16.0, 12.0)))
                .layout(Layout::top_down(Align::Min)),
            |ui| {
                let message = if self.view == ViewMode::Depth {
                    "This image has no depth map yet."
                } else {
                    "Depth parallax needs a depth map. Showing the unchanged source until you \
                     load or generate one."
                };
                ui.label(RichText::new(message).color(style::TEXT));
                ui.horizontal(|ui| {
                    let generating = matches!(self.depth_task, DepthTask::Generating { .. });
                    if !matches!(self.depth_task, DepthTask::Idle) {
                        ui.spinner();
                        ui.label(self.depth_status());
                    }
                    if ui
                        .add_enabled(self.dialog.is_none(), egui::Button::new("Load depth map…"))
                        .clicked()
                    {
                        self.open_dialog(frame, Purpose::LoadDepth);
                    }
                    if !generating
                        && ui
                            .add(style::primary("Generate depth"))
                            .on_hover_text(
                                "Runs Depth Anything V2 Small locally; the first run downloads \
                                 the 99 MB model",
                            )
                            .clicked()
                    {
                        self.generate_depth();
                    }
                });
                if let Some(note) = &self.depth_note {
                    note.show(ui);
                }
            },
        );
    }

    fn depth_legend(&self, ui: &mut egui::Ui, fitted: Rect) {
        let (Some(depth), Some(source)) = (&self.depth, &self.source) else {
            return;
        };
        let painter = ui.painter();
        let panel = Rect::from_min_size(
            Pos2::new(fitted.left() + 12.0, fitted.bottom() - 74.0),
            Vec2::new(fitted.width().min(560.0) - 24.0, 62.0),
        );
        painter.rect_filled(
            panel,
            CornerRadius::same(4),
            Color32::from_rgba_unmultiplied(0x2a, 0x27, 0x24, 230),
        );
        let bar = Rect::from_min_size(panel.min + Vec2::new(44.0, 12.0), Vec2::new(150.0, 10.0));
        let steps = 30;
        for i in 0..steps {
            let level = (i as f32 + 0.5) / steps as f32;
            let gray = (level * 255.0).round() as u8;
            let x0 = bar.left() + bar.width() * i as f32 / steps as f32;
            let x1 = bar.left() + bar.width() * (i + 1) as f32 / steps as f32;
            painter.rect_filled(
                Rect::from_min_max(Pos2::new(x0, bar.top()), Pos2::new(x1 + 0.5, bar.bottom())),
                CornerRadius::ZERO,
                Color32::from_gray(gray),
            );
        }
        painter.rect_stroke(
            bar,
            0.0,
            Stroke::new(1.0, style::BORDER),
            egui::StrokeKind::Outside,
        );
        let font = FontId::proportional(11.5);
        painter.text(
            Pos2::new(bar.left() - 8.0, bar.center().y),
            Align2::RIGHT_CENTER,
            "Far",
            font.clone(),
            style::MUTED,
        );
        painter.text(
            Pos2::new(bar.right() + 8.0, bar.center().y),
            Align2::LEFT_CENTER,
            "Near",
            font.clone(),
            style::MUTED,
        );
        let invert = if self.project.depth.invert {
            "Inverted from the raw map; the saved map stays raw"
        } else {
            "As stored in the raw map"
        };
        painter.text(
            Pos2::new(bar.right() + 52.0, bar.center().y),
            Align2::LEFT_CENTER,
            invert,
            font.clone(),
            style::TEXT,
        );
        let map = depth.map.size();
        let image = source.image.size();
        let output = self.project.output;
        painter.text(
            Pos2::new(panel.left() + 12.0, panel.bottom() - 12.0),
            Align2::LEFT_BOTTOM,
            format!(
                "Map {} × {} spans the full {} × {} source, then is cover-cropped to {} × {}",
                map.width, map.height, image.width, image.height, output.width, output.height
            ),
            font,
            style::MUTED,
        );
    }

    fn drop_hint(&self, ui: &mut egui::Ui, rect: Rect) {
        let hovering = ui.ctx().input(|input| !input.raw.hovered_files.is_empty());
        if !hovering {
            return;
        }
        let painter = ui.painter();
        painter.rect_filled(rect, CornerRadius::ZERO, Color32::from_black_alpha(190));
        painter.rect_stroke(
            rect.shrink(10.0),
            CornerRadius::same(6),
            Stroke::new(2.0, style::ACCENT),
            egui::StrokeKind::Inside,
        );
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            "Drop one image to open it",
            FontId::proportional(20.0),
            style::TEXT,
        );
    }
}

impl eframe::App for Workbench {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_dialog();
        while let Some(event) = self.workers.try_recv() {
            self.handle(event);
        }
        self.poll_capture();
        if self
            .export
            .as_ref()
            .is_some_and(|export| !export.state().is_finished())
        {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.handle_keys(&ctx);
        self.handle_drops(&ctx);
        self.advance_playback(&ctx);

        let bar = egui::Frame::new()
            .fill(style::PANEL)
            .inner_margin(Margin::symmetric(12, 6));
        let side = egui::Frame::new()
            .fill(style::PANEL)
            .inner_margin(Margin::symmetric(14, 6));
        egui::Panel::top("toolbar")
            .frame(bar)
            .exact_size(44.0)
            .show(ui, |ui| self.toolbar(ui, frame));
        egui::Panel::bottom("transport")
            .frame(bar)
            .show(ui, |ui| self.transport(ui));
        egui::Panel::left("motion")
            .frame(side)
            .default_size(320.0)
            .min_size(270.0)
            .max_size(440.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| self.motion_inspector(ui, frame));
            });
        egui::Panel::right("export")
            .frame(side)
            .default_size(300.0)
            .min_size(260.0)
            .max_size(440.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| self.export_inspector(ui, frame));
            });
        self.sync_timeline();
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(style::MATTE))
            .show(ui, |ui| self.stage(ui, frame));
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        match self.project.to_json() {
            Ok(json) => storage.set_string(KEY_PROJECT, json),
            Err(error) => self.notice = Note::error(format!("Could not save settings: {error}")),
        }
        storage.set_string(
            KEY_RECENT_SOURCE,
            self.recent_source
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_default(),
        );
    }

    fn on_exit(&mut self) {
        self.shutdown();
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        style::PANEL.to_normalized_gamma_f32()
    }

    fn persist_egui_memory(&self) -> bool {
        false
    }
}

impl Drop for Workbench {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The largest rect of `aspect` (width / height) centered in `area`.
fn fit(area: Rect, aspect: f32) -> Rect {
    let size = if area.width() / area.height().max(1.0) > aspect {
        Vec2::new(area.height() * aspect, area.height())
    } else {
        Vec2::new(area.width(), area.width() / aspect)
    };
    Rect::from_center_size(area.center(), size)
}

fn display_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// Adds `extension` when the chosen path has none.
fn with_extension(path: PathBuf, extension: &str) -> PathBuf {
    if path.extension().is_some() {
        path
    } else {
        path.with_extension(extension)
    }
}

/// `m:ss.ss`, or `h:mm:ss.ss` past an hour.
fn format_time(seconds: f64) -> String {
    let centis = (seconds.max(0.0) * 100.0).round() as u64;
    let (hours, rest) = (centis / 360_000, centis % 360_000);
    let (minutes, rest) = (rest / 6_000, rest % 6_000);
    let (secs, hundredths) = (rest / 100, rest % 100);
    if hours > 0 {
        format!("{hours}:{minutes:02}:{secs:02}.{hundredths:02}")
    } else {
        format!("{minutes}:{secs:02}.{hundredths:02}")
    }
}
