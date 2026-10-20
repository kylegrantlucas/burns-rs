//! Command implementations. Results go to stdout; progress, notes, and errors go to stderr.

use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread;
use std::time::Duration;

use burns::presets::DEFAULT_PRESET;
use burns::{
    CameraMotion, DepthMap, GpuContext, MotionProject, SourceImage, preset, presets,
    settings_schema,
};
use burns_depth::DepthService;
use burns_media::{
    FrameRequest, MediaError, RenderJob, RenderRequest, RenderStatus, load_depth, load_image,
    save_frame, write_depth_png,
};

use crate::cli::{Cli, Command, DepthArgs, DepthSource, FrameArgs, ProjectArgs, RenderArgs};
use crate::error::CliError;
use crate::interrupt;
use crate::progress::{FrameProgress, depth_reporter};
use crate::project::apply_overrides;

/// How often the export status is polled.
const POLL: Duration = Duration::from_millis(50);

pub fn run(cli: Cli) -> Result<(), CliError> {
    match cli.command {
        Command::Presets { json } => list_presets(json),
        Command::Schema => print_json(&settings_schema()),
        Command::Preset { name } => print_stdout(&preset(&name)?.to_json()?),
        Command::Render(args) => render(args),
        Command::Frame(args) => frame(args),
        Command::Depth(args) => depth(args),
    }
}

fn print_stdout(text: &str) -> Result<(), CliError> {
    let mut out = io::stdout().lock();
    writeln!(out, "{text}")
        .and_then(|()| out.flush())
        .map_err(CliError::Stdout)
}

fn print_json(value: &impl serde::Serialize) -> Result<(), CliError> {
    let json = serde_json::to_string_pretty(value).expect("schema types serialize to JSON");
    print_stdout(&json)
}

fn list_presets(json: bool) -> Result<(), CliError> {
    if json {
        return print_json(&presets());
    }
    let name_width = presets().iter().map(|p| p.name.len()).max().unwrap_or(0);
    let label_width = presets().iter().map(|p| p.label.len()).max().unwrap_or(0);
    let yes_no = |flag: bool| if flag { "yes" } else { "no" };
    let mut text = format!(
        "{:name_width$}  {:label_width$}  DEPTH  PERIODIC",
        "NAME", "LABEL"
    );
    for preset in presets() {
        text.push_str(&format!(
            "\n{:name_width$}  {:label_width$}  {:5}  {}",
            preset.name,
            preset.label,
            yes_no(preset.requires_depth),
            yes_no(preset.periodic)
        ));
    }
    print_stdout(&text)
}

/// Base project (selected preset or a strictly parsed file), then the command-line overrides.
fn load_project(args: &ProjectArgs) -> Result<MotionProject, CliError> {
    let base = match (&args.config, &args.preset) {
        (Some(path), _) => {
            let json = fs::read_to_string(path).map_err(|source| CliError::ReadProject {
                path: path.clone(),
                source,
            })?;
            MotionProject::from_json(&json).map_err(|source| CliError::ProjectFile {
                path: path.clone(),
                source,
            })?
        }
        (None, Some(name)) => preset(name)?,
        (None, None) => preset(DEFAULT_PRESET)?,
    };
    Ok(apply_overrides(base, &args.overrides())?)
}

/// Refuses an existing output before any decoding, inference, or GPU work. burns-media checks
/// again and publishes without clobbering, so this is only an early, cheap answer.
fn refuse_existing(output: &Path, overwrite: bool) -> Result<(), CliError> {
    if !overwrite && fs::symlink_metadata(output).is_ok() {
        return Err(MediaError::OutputExists {
            path: output.to_owned(),
        }
        .into());
    }
    Ok(())
}

fn uses_depth(project: &MotionProject) -> bool {
    matches!(project.camera, CameraMotion::Depth { .. })
}

/// The depth map a render needs: only the depth camera uses one. A supplied map wins; otherwise
/// `--auto-depth` generates (or reuses) one, and without it only an already cached map is used.
/// 2D cameras never construct the depth service, ONNX Runtime, or a model.
fn resolve_depth(
    project: &MotionProject,
    image: &SourceImage,
    input: &Path,
    source: &DepthSource,
) -> Result<Option<DepthMap>, CliError> {
    if !uses_depth(project) {
        let flag = match (&source.depth, source.auto_depth) {
            (Some(_), _) => Some("--depth"),
            (None, true) => Some("--auto-depth"),
            (None, false) => None,
        };
        if let Some(flag) = flag {
            eprintln!(
                "note: the {} camera does not use depth; ignoring {flag}",
                project.camera.kind().as_str()
            );
        }
        return Ok(None);
    }
    if let Some(path) = &source.depth {
        let map = load_depth(path)?;
        let size = map.size();
        eprintln!(
            "depth: using `{}` ({}x{})",
            path.display(),
            size.width,
            size.height
        );
        return Ok(Some(map));
    }
    let mut service = DepthService::new()?;
    if source.auto_depth {
        let cancel = interrupt::watch()?;
        return Ok(Some(service.generate(image, cancel, depth_reporter())?));
    }
    match service.cached(image)? {
        Some(map) => {
            eprintln!(
                "depth: using the cached map `{}`",
                service.depth_cache_path(image).display()
            );
            Ok(Some(map))
        }
        None => Err(CliError::DepthUnavailable {
            image: input.to_owned(),
        }),
    }
}

fn gpu() -> Result<GpuContext, CliError> {
    Ok(pollster::block_on(GpuContext::headless())?)
}

fn describe(project: &MotionProject) -> String {
    let timeline = project.timeline;
    format!(
        "{}x{}, {} frames ({:.3} s at {} fps)",
        project.output.width,
        project.output.height,
        timeline.frames,
        timeline.duration_seconds(),
        timeline.fps
    )
}

fn render(args: RenderArgs) -> Result<(), CliError> {
    let project = load_project(&args.project)?;
    let output = project.output;
    if output.width % 2 != 0 || output.height % 2 != 0 {
        return Err(MediaError::OddDimensions {
            width: output.width,
            height: output.height,
        }
        .into());
    }
    refuse_existing(&args.output, args.overwrite)?;
    let interrupted = interrupt::watch()?;
    let image = Arc::new(load_image(&args.input)?);
    let depth = resolve_depth(&project, &image, &args.input, &args.depth)?.map(Arc::new);
    interrupt::check(interrupted)?;
    let context = gpu()?;
    interrupt::check(interrupted)?;
    let summary = format!(
        "rendering {} to `{}` (CRF {})",
        describe(&project),
        args.output.display(),
        args.crf
    );
    let job = RenderJob::start(
        context,
        RenderRequest {
            project,
            image,
            depth,
            output: args.output,
            crf: args.crf,
            overwrite: args.overwrite,
        },
    )?;
    eprintln!("{summary}");
    let status = wait(job, interrupted);
    match status {
        RenderStatus::Done { output } => {
            eprintln!("wrote `{}`", output.display());
            Ok(())
        }
        // A terminal's Ctrl-C also signals FFmpeg, which can make the export fail before the
        // job is cancelled; the user asked to stop either way, and nothing was published.
        RenderStatus::Failed { .. } if interrupt::check(interrupted).is_err() => {
            Err(CliError::Interrupted)
        }
        RenderStatus::Failed { message } => Err(CliError::Export(message)),
        RenderStatus::Cancelled => Err(CliError::Interrupted),
        RenderStatus::Running { .. } => unreachable!("a joined job has finished"),
    }
}

/// Reports progress until the job finishes, cancelling it on Ctrl-C. If publication had already
/// begun, the job completes instead.
fn wait(job: RenderJob, interrupted: &AtomicBool) -> RenderStatus {
    let mut progress = FrameProgress::new();
    let mut cancelling = false;
    while let RenderStatus::Running { completed, total } = job.status() {
        if !cancelling && interrupt::check(interrupted).is_err() {
            cancelling = true;
            job.cancel();
            progress.finish();
            eprintln!("interrupted; stopping the export");
        }
        if !cancelling {
            progress.update(completed, total);
        }
        thread::sleep(POLL);
    }
    progress.finish();
    job.wait()
}

fn frame(args: FrameArgs) -> Result<(), CliError> {
    let project = load_project(&args.project)?;
    let time = project.timeline.frame_time(args.frame)?;
    refuse_existing(&args.output, args.overwrite)?;
    let interrupted = interrupt::watch()?;
    let image = load_image(&args.input)?;
    let depth = resolve_depth(&project, &image, &args.input, &args.depth)?;
    interrupt::check(interrupted)?;
    let context = gpu()?;
    save_frame(
        context,
        FrameRequest {
            project: &project,
            image: &image,
            depth: depth.as_ref(),
            frame: args.frame,
            output: &args.output,
            overwrite: args.overwrite,
            cancel: Some(interrupted),
        },
    )?;
    eprintln!(
        "wrote frame {} (t = {:.3} s) of {} to `{}`",
        args.frame,
        time.seconds,
        describe(&project),
        args.output.display()
    );
    Ok(())
}

fn depth(args: DepthArgs) -> Result<(), CliError> {
    refuse_existing(&args.output, args.overwrite)?;
    let image = load_image(&args.input)?;
    let interrupted = interrupt::watch()?;
    let mut service = DepthService::new()?;
    eprintln!(
        "depth: cache `{}`",
        service.depth_cache_path(&image).display()
    );
    let map = service.generate(&image, interrupted, depth_reporter())?;
    interrupt::check(interrupted)?;
    write_depth_png(&map, &args.output, args.overwrite, Some(interrupted))?;
    let size = map.size();
    eprintln!(
        "wrote {}x{} 16-bit depth map to `{}`",
        size.width,
        size.height,
        args.output.display()
    );
    Ok(())
}
