//! MP4 export and PNG frames on a real GPU with the system `ffmpeg`/`ffprobe`.
//!
//! Every test needs a Metal, Vulkan, or DirectX 12 adapter and FFmpeg with libx264 on `PATH`
//! (both come with `nix develop`), so they are ignored by default; run them with
//! `cargo test -p burns-media --test export -- --ignored`.

use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, LazyLock};
use std::thread;
use std::time::{Duration, Instant};

use burns::{DepthMap, FrameRate, GpuContext, MotionProject, Size, SourceImage, Timeline, preset};
use burns_media::{
    DEFAULT_CRF, FrameRequest, MediaError, RenderJob, RenderRequest, RenderStatus, save_frame,
};

static CONTEXT: LazyLock<GpuContext> = LazyLock::new(|| {
    pollster::block_on(GpuContext::headless())
        .unwrap_or_else(|error| panic!("GPU tests need a real adapter: {error}"))
});

fn context() -> GpuContext {
    CONTEXT.clone()
}

/// White two-pixel grid lines every 16 pixels over dark blue cells.
fn bright_grid(size: Size) -> SourceImage {
    let mut pixels = Vec::with_capacity(size.width as usize * size.height as usize * 4);
    for y in 0..size.height {
        for x in 0..size.width {
            if x % 16 < 2 || y % 16 < 2 {
                pixels.extend([255, 255, 255, 255]);
            } else {
                pixels.extend([16, 24, 96, 255]);
            }
        }
    }
    SourceImage::from_rgba8(size, pixels).unwrap()
}

fn project(name: &str, output: Size, fps: FrameRate, frames: u32) -> MotionProject {
    let mut project = preset(name).unwrap();
    project.output = output;
    project.timeline = Timeline::new(fps, frames).unwrap();
    project
}

fn request(project: MotionProject, output: PathBuf, overwrite: bool) -> RenderRequest {
    RenderRequest {
        project,
        image: Arc::new(bright_grid(Size::new(640, 360))),
        depth: None,
        output,
        crf: DEFAULT_CRF,
        overwrite,
    }
}

fn entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn wait_until_running(job: &RenderJob, frames: u32) {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        match job.status() {
            RenderStatus::Running { completed, .. } if completed >= frames => return,
            RenderStatus::Running { .. } => {}
            other => panic!("export stopped early: {other:?}"),
        }
        assert!(Instant::now() < deadline, "export made no progress");
        thread::sleep(Duration::from_millis(5));
    }
}

fn ffprobe(path: &Path) -> serde_json::Value {
    let output = Command::new("ffprobe")
        .args(["-v", "error", "-count_frames", "-show_entries"])
        .arg(
            "stream=codec_type,codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,\
             nb_read_frames,color_range,color_space,color_primaries,color_transfer",
        )
        .args(["-of", "json"])
        .arg(path)
        .output()
        .expect("ffprobe must be on PATH");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

/// Frame `index` decoded by FFmpeg as tightly packed RGB24.
fn decode_frame(path: &Path, index: u32) -> Vec<u8> {
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-vf", &format!("select=eq(n\\,{index})"), "-frames:v", "1"])
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        .output()
        .expect("ffmpeg must be on PATH");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

/// Decodes an 8-bit RGBA PNG into its size and pixels.
fn read_png(path: &Path) -> (Size, Vec<u8>) {
    let decoder = png::Decoder::new(Cursor::new(fs::read(path).unwrap()));
    let mut reader = decoder.read_info().unwrap();
    let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut pixels).unwrap();
    assert_eq!(info.color_type, png::ColorType::Rgba);
    assert_eq!(info.bit_depth, png::BitDepth::Eight);
    pixels.truncate(info.buffer_size());
    (Size::new(info.width, info.height), pixels)
}

/// Fractions of pixels that are bright (every channel above 200) and dark (every channel below 120).
fn bright_and_dark(pixels: &[u8], channels: usize) -> (f64, f64) {
    let count = pixels.len() / channels;
    let (mut bright, mut dark) = (0, 0);
    for pixel in pixels.chunks_exact(channels) {
        if pixel[..3].iter().all(|&c| c > 200) {
            bright += 1;
        }
        if pixel[..3].iter().all(|&c| c < 120) {
            dark += 1;
        }
    }
    (bright as f64 / count as f64, dark as f64 / count as f64)
}

#[test]
#[ignore = "requires a GPU adapter and FFmpeg with libx264"]
fn complete_export_is_a_tagged_video_only_mp4_with_every_frame() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("grid.mp4");
    let fps = FrameRate::new(30000, 1001);
    let project = project("kenburns", Size::new(320, 180), fps, 24);
    let job = RenderJob::start(context(), request(project.clone(), output.clone(), false)).unwrap();

    let mut last = 0;
    let status = loop {
        match job.status() {
            RenderStatus::Running { completed, total } => {
                assert_eq!(total, 24);
                assert!(
                    completed >= last && completed <= total,
                    "{completed}/{total}"
                );
                last = completed;
            }
            finished => break finished,
        }
        thread::sleep(Duration::from_millis(2));
    };
    assert_eq!(
        status,
        RenderStatus::Done {
            output: output.clone()
        }
    );
    assert_eq!(
        job.wait(),
        RenderStatus::Done {
            output: output.clone()
        }
    );
    assert_eq!(entries(dir.path()), ["grid.mp4"]);

    let probe = ffprobe(&output);
    let streams = probe["streams"].as_array().unwrap();
    assert_eq!(streams.len(), 1, "video only, no audio: {probe}");
    let video = &streams[0];
    assert_eq!(video["codec_type"], "video");
    assert_eq!(video["codec_name"], "h264");
    assert_eq!(video["width"], 320);
    assert_eq!(video["height"], 180);
    assert_eq!(video["pix_fmt"], "yuv420p");
    assert_eq!(video["r_frame_rate"], "30000/1001");
    assert_eq!(video["avg_frame_rate"], "30000/1001");
    assert_eq!(video["nb_read_frames"], "24");
    assert_eq!(video["color_range"], "tv");
    assert_eq!(video["color_space"], "bt709");
    assert_eq!(video["color_primaries"], "bt709");
    assert_eq!(video["color_transfer"], "bt709");

    // The decoded video matches the renderer's own final-quality frame.
    for index in [0, 23] {
        let png = dir.path().join(format!("frame{index}.png"));
        let image = bright_grid(Size::new(640, 360));
        save_frame(
            context(),
            FrameRequest {
                project: &project,
                image: &image,
                depth: None,
                frame: index,
                output: &png,
                overwrite: false,
                cancel: None,
            },
        )
        .unwrap();
        let (size, rendered) = read_png(&png);
        assert_eq!(size, Size::new(320, 180));
        let decoded = decode_frame(&output, index);
        assert_eq!(decoded.len(), 320 * 180 * 3);
        let (bright, dark) = bright_and_dark(&decoded, 3);
        assert!(bright > 0.05 && dark > 0.3, "bright {bright}, dark {dark}");
        let difference: u64 = rendered
            .chunks_exact(4)
            .zip(decoded.chunks_exact(3))
            .flat_map(|(a, b)| (0..3).map(move |c| u64::from(a[c].abs_diff(b[c]))))
            .sum();
        let mean = difference as f64 / (320.0 * 180.0 * 3.0);
        println!("frame {index}: bright {bright:.3}, dark {dark:.3}, mean |png - mp4| {mean:.2}");
        assert!(
            mean < 12.0,
            "frame {index} decoded far from the render: {mean}"
        );
    }
}

#[test]
#[ignore = "requires a GPU adapter and FFmpeg with libx264"]
fn cancelling_a_long_export_keeps_the_previous_file_and_no_partial() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("long.mp4");
    fs::write(&output, b"previous export").unwrap();
    let project = project("shake", Size::new(1280, 720), FrameRate::new(30, 1), 1800);
    let job = RenderJob::start(context(), request(project, output.clone(), true)).unwrap();
    wait_until_running(&job, 3);
    let started = Instant::now();
    job.cancel();
    assert_eq!(job.wait(), RenderStatus::Cancelled);
    println!("cancellation finished in {:?}", started.elapsed());
    assert!(started.elapsed() < Duration::from_secs(10));
    assert_eq!(fs::read(&output).unwrap(), b"previous export");
    assert_eq!(entries(dir.path()), ["long.mp4"]);
}

#[test]
#[ignore = "requires a GPU adapter and FFmpeg with libx264"]
fn dropping_a_running_export_stops_it_without_output() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("dropped.mp4");
    let project = project(
        "kenburns",
        Size::new(1280, 720),
        FrameRate::new(30, 1),
        1800,
    );
    let job = RenderJob::start(context(), request(project, output.clone(), false)).unwrap();
    wait_until_running(&job, 2);
    drop(job);
    assert!(entries(dir.path()).is_empty());
}

#[test]
#[ignore = "requires a GPU adapter and FFmpeg with libx264"]
fn existing_output_and_dangling_symlink_are_refused_before_rendering() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("taken.mp4");
    fs::write(&output, b"previous export").unwrap();
    let project = project("kenburns", Size::new(64, 36), FrameRate::new(30, 1), 4);
    let error =
        RenderJob::start(context(), request(project.clone(), output.clone(), false)).unwrap_err();
    assert!(matches!(error, MediaError::OutputExists { .. }), "{error}");
    assert_eq!(fs::read(&output).unwrap(), b"previous export");

    #[cfg(unix)]
    {
        let link = dir.path().join("link.mp4");
        std::os::unix::fs::symlink(dir.path().join("nowhere.mp4"), &link).unwrap();
        let error = RenderJob::start(context(), request(project, link.clone(), false)).unwrap_err();
        assert!(matches!(error, MediaError::OutputExists { .. }), "{error}");
        assert!(
            fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(entries(dir.path()), ["link.mp4", "taken.mp4"]);
    }
}

#[test]
#[ignore = "requires a GPU adapter and FFmpeg with libx264"]
fn odd_size_is_refused_for_mp4_but_saved_as_png() {
    let dir = tempfile::tempdir().unwrap();
    let project = project("kenburns", Size::new(321, 181), FrameRate::new(30, 1), 4);
    let error = RenderJob::start(
        context(),
        request(project.clone(), dir.path().join("odd.mp4"), false),
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            MediaError::OddDimensions {
                width: 321,
                height: 181
            }
        ),
        "{error}"
    );
    assert!(error.to_string().contains("320x180"), "{error}");
    assert!(entries(dir.path()).is_empty());

    let png = dir.path().join("odd.png");
    let image = bright_grid(Size::new(640, 360));
    save_frame(
        context(),
        FrameRequest {
            project: &project,
            image: &image,
            depth: None,
            frame: 3,
            output: &png,
            overwrite: false,
            cancel: None,
        },
    )
    .unwrap();
    let (size, pixels) = read_png(&png);
    assert_eq!(size, Size::new(321, 181));
    let (bright, dark) = bright_and_dark(&pixels, 4);
    assert!(bright > 0.05 && dark > 0.3, "bright {bright}, dark {dark}");
    assert!(pixels.chunks_exact(4).all(|p| p[3] == 255));
    assert_eq!(entries(dir.path()), ["odd.png"]);

    let error = save_frame(
        context(),
        FrameRequest {
            project: &project,
            image: &image,
            depth: None,
            frame: 3,
            output: &png,
            overwrite: false,
            cancel: None,
        },
    )
    .unwrap_err();
    assert!(matches!(error, MediaError::OutputExists { .. }), "{error}");
    let error = save_frame(
        context(),
        FrameRequest {
            project: &project,
            image: &image,
            depth: None,
            frame: 4,
            output: &dir.path().join("late.png"),
            overwrite: false,
            cancel: None,
        },
    )
    .unwrap_err();
    assert!(matches!(error, MediaError::Config(_)), "{error}");
    assert_eq!(entries(dir.path()), ["odd.png"]);
}

#[test]
#[ignore = "requires a GPU adapter and FFmpeg with libx264"]
fn invalid_crf_and_depth_inputs_are_refused_without_output() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("out.mp4");
    let kenburns = project("kenburns", Size::new(64, 36), FrameRate::new(30, 1), 4);

    let mut bad_crf = request(kenburns, output.clone(), false);
    bad_crf.crf = 52;
    let error = RenderJob::start(context(), bad_crf).unwrap_err();
    assert!(
        matches!(error, MediaError::InvalidCrf { crf: 52 }),
        "{error}"
    );

    let orbital = project("orbital", Size::new(64, 36), FrameRate::new(30, 1), 4);
    let error =
        RenderJob::start(context(), request(orbital.clone(), output.clone(), false)).unwrap_err();
    assert!(matches!(error, MediaError::MissingDepth), "{error}");

    let image = bright_grid(Size::new(640, 360));
    let error = save_frame(
        context(),
        FrameRequest {
            project: &orbital,
            image: &image,
            depth: None,
            frame: 0,
            output: &output,
            overwrite: false,
            cancel: None,
        },
    )
    .unwrap_err();
    assert!(matches!(error, MediaError::MissingDepth), "{error}");

    assert!(entries(dir.path()).is_empty());
}

#[test]
#[ignore = "requires a GPU adapter and FFmpeg with libx264"]
fn imported_depth_grids_use_the_full_uv_extent_at_any_resolution() {
    let dir = tempfile::tempdir().unwrap();
    let project = project("orbital", Size::new(128, 72), FrameRate::new(24, 1), 6);
    let image = Arc::new(bright_grid(Size::new(640, 360)));
    let wide = DepthMap::from_f32(Size::new(32, 18), vec![0.5; 32 * 18]).unwrap();
    let square = Arc::new(DepthMap::from_f32(Size::new(16, 16), vec![0.5; 16 * 16]).unwrap());
    let wide_png = dir.path().join("wide.png");
    let square_png = dir.path().join("square.png");
    save_frame(
        context(),
        FrameRequest {
            project: &project,
            image: &image,
            depth: Some(&wide),
            frame: 2,
            output: &wide_png,
            overwrite: false,
            cancel: None,
        },
    )
    .unwrap();
    save_frame(
        context(),
        FrameRequest {
            project: &project,
            image: &image,
            depth: Some(&square),
            frame: 2,
            output: &square_png,
            overwrite: false,
            cancel: None,
        },
    )
    .unwrap();
    assert_eq!(read_png(&square_png), read_png(&wide_png));
    let output = dir.path().join("square-depth.mp4");
    let job = RenderJob::start(
        context(),
        RenderRequest {
            project,
            image,
            depth: Some(square),
            output: output.clone(),
            crf: DEFAULT_CRF,
            overwrite: false,
        },
    )
    .unwrap();
    assert_eq!(
        job.wait(),
        RenderStatus::Done {
            output: output.clone()
        }
    );
    assert_eq!(ffprobe(&output)["streams"][0]["nb_read_frames"], "6");
}
