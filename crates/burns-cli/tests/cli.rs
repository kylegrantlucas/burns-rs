//! Failures the `burns` binary must report before touching the GPU, FFmpeg, or ONNX Runtime,
//! without writing to stdout or disturbing the output path.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use tempfile::TempDir;

struct Sandbox {
    dir: TempDir,
}

impl Sandbox {
    fn new() -> Self {
        let sandbox = Self {
            dir: tempfile::tempdir().unwrap(),
        };
        fs::create_dir(sandbox.path("home")).unwrap();
        write_png(&sandbox.path("image.png"), 8, 6);
        sandbox
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    /// Runs `burns` with an empty home/cache and no usable ONNX Runtime, only for the child.
    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_burns"))
            .args(args)
            .current_dir(self.dir.path())
            .env("HOME", self.path("home"))
            .env("XDG_CACHE_HOME", self.path("home/cache"))
            .env("ORT_DYLIB_PATH", self.path("no-onnxruntime"))
            .output()
            .unwrap()
    }
}

fn write_png(path: &Path, width: u32, height: u32) {
    let file = fs::File::create(path).unwrap();
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i * 7) as u8).collect();
    encoder
        .write_header()
        .unwrap()
        .write_image_data(&pixels)
        .unwrap();
}

fn assert_failed(output: &Output) {
    assert_eq!(
        output.status.code(),
        Some(1),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty(), "stdout must stay clean");
    assert!(!output.stderr.is_empty(), "the error is reported on stderr");
}

/// No published output and no leftover temporary in the directory.
fn assert_nothing_written(sandbox: &Sandbox, name: &str) {
    let entries: Vec<_> = fs::read_dir(sandbox.dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|entry| entry.contains(name))
        .collect();
    assert!(entries.is_empty(), "unexpected outputs: {entries:?}");
}

#[test]
fn existing_output_is_preserved_without_overwrite() {
    let sandbox = Sandbox::new();
    for (command, name) in [
        ("render", "clip.mp4"),
        ("frame", "frame.png"),
        ("depth", "depth.png"),
    ] {
        let existing = sandbox.path(name);
        fs::write(&existing, b"keep me").unwrap();
        let mut args = vec![command, "-i", "image.png", "-o", name];
        if command == "frame" {
            args.extend(["--frame", "0"]);
        }
        assert_failed(&sandbox.run(&args));
        assert_eq!(fs::read(&existing).unwrap(), b"keep me", "{command}");
    }
}

#[test]
fn out_of_range_frame_is_rejected() {
    let sandbox = Sandbox::new();
    // The final timeline is 2 s at 10 fps: frames 0..=19.
    for frame in ["20", "4294967295"] {
        let output = sandbox.run(&[
            "frame",
            "-i",
            "image.png",
            "-o",
            "frame.png",
            "--frame",
            frame,
            "--fps",
            "10",
            "--duration",
            "2",
        ]);
        assert_failed(&output);
    }
    assert_nothing_written(&sandbox, "frame.png");
}

#[test]
fn odd_mp4_dimensions_are_rejected() {
    let sandbox = Sandbox::new();
    let output = sandbox.run(&[
        "render",
        "-i",
        "image.png",
        "-o",
        "clip.mp4",
        "--width",
        "641",
        "--height",
        "360",
    ]);
    assert_failed(&output);
    assert_nothing_written(&sandbox, "clip.mp4");
}

#[test]
fn depth_motion_without_a_map_or_cache_fails_without_auto_depth() {
    let sandbox = Sandbox::new();
    let project = ["--preset", "orbital", "--width", "64", "--height", "36"];
    let render = [
        &["render", "-i", "image.png", "-o", "out.file"][..],
        &project,
    ]
    .concat();
    let frame = [
        &["frame", "-i", "image.png", "-o", "out.file", "--frame", "0"][..],
        &project,
    ]
    .concat();
    for args in [render, frame] {
        assert_failed(&sandbox.run(&args));
    }
    assert_nothing_written(&sandbox, "out.file");
    // The cache lookup must not create the cache, model, or runtime state.
    assert_eq!(fs::read_dir(sandbox.path("home")).unwrap().count(), 0);
}

#[test]
fn project_files_are_strict() {
    let sandbox = Sandbox::new();
    let valid = burns::preset("kenburns").unwrap().to_json().unwrap();
    let mut document: serde_json::Value = serde_json::from_str(&valid).unwrap();
    document["shake"]["sped"] = serde_json::json!(1.0);
    fs::write(sandbox.path("unknown.json"), document.to_string()).unwrap();
    let mut document: serde_json::Value = serde_json::from_str(&valid).unwrap();
    document["warp"].as_object_mut().unwrap().remove("enabled");
    fs::write(sandbox.path("missing.json"), document.to_string()).unwrap();
    let mut document: serde_json::Value = serde_json::from_str(&valid).unwrap();
    document["shake"]["speed"] = serde_json::json!(40.0);
    fs::write(sandbox.path("range.json"), document.to_string()).unwrap();

    for config in ["unknown.json", "missing.json", "range.json", "absent.json"] {
        let output = sandbox.run(&[
            "frame",
            "-i",
            "image.png",
            "-o",
            "frame.png",
            "--frame",
            "0",
            "--config",
            config,
        ]);
        assert_failed(&output);
    }
    assert_nothing_written(&sandbox, "frame.png");
}

#[cfg(unix)]
#[test]
#[ignore = "requires a real GPU adapter with a 3840-pixel texture limit"]
fn sigint_during_png_encoding_preserves_the_prior_file_and_removes_the_temporary() {
    use std::io::Read;
    use std::process::{Child, Stdio};
    use std::thread;
    use std::time::{Duration, Instant};

    struct OwnedChild(Child);
    impl Drop for OwnedChild {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let sandbox = Sandbox::new();
    let mut random = 0x9173_5041u32;
    let pixels: Vec<u8> = (0..1280 * 720 * 3)
        .map(|_| {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            random as u8
        })
        .collect();
    let mut encoder = png::Encoder::new(
        fs::File::create(sandbox.path("image.png")).unwrap(),
        1280,
        720,
    );
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .unwrap()
        .write_image_data(&pixels)
        .unwrap();
    fs::write(sandbox.path("frame.png"), b"prior capture").unwrap();
    let mut process = OwnedChild(
        Command::new(env!("CARGO_BIN_EXE_burns"))
            .args([
                "frame",
                "-i",
                "image.png",
                "-o",
                "frame.png",
                "--frame",
                "0",
                "--overwrite",
            ])
            .current_dir(sandbox.dir.path())
            .env("HOME", sandbox.path("home"))
            .env("ORT_DYLIB_PATH", sandbox.path("no-onnxruntime"))
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let writing = fs::read_dir(sandbox.dir.path()).unwrap().any(|entry| {
            let entry = entry.unwrap();
            entry.file_name() != "image.png"
                && entry.file_name() != "frame.png"
                && entry
                    .metadata()
                    .is_ok_and(|meta| meta.is_file() && meta.len() > 4096)
        });
        if writing {
            break;
        }
        assert!(
            process.0.try_wait().unwrap().is_none(),
            "PNG command exited before encoding"
        );
        assert!(Instant::now() < deadline, "PNG encoding never began");
        thread::sleep(Duration::from_millis(1));
    }
    assert!(
        Command::new("kill")
            .args(["-INT", &process.0.id().to_string()])
            .status()
            .unwrap()
            .success()
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = process.0.try_wait().unwrap() {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "PNG cancellation did not complete"
        );
        thread::sleep(Duration::from_millis(2));
    };
    let mut stderr = String::new();
    process
        .0
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    assert_eq!(status.code(), Some(130), "{stderr}");
    assert_eq!(
        fs::read(sandbox.path("frame.png")).unwrap(),
        b"prior capture"
    );
    let mut entries: Vec<_> = fs::read_dir(sandbox.dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    entries.sort();
    assert_eq!(entries, ["frame.png", "home", "image.png"]);
}
