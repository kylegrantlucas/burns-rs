//! Output publishing, asset loading, and depth saving; no GPU or FFmpeg needed.

use std::fs;
use std::io::Write;
use std::path::Path;

use burns::{DepthMap, Size};
use burns_media::{MediaError, load_depth, load_image, safe_write, write_depth_png};

fn entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn write_bytes(bytes: &'static [u8]) -> impl FnOnce(&mut fs::File) -> Result<(), MediaError> {
    move |file| {
        file.write_all(bytes).unwrap();
        Ok(())
    }
}

#[test]
fn writes_a_new_file_without_leftovers() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("project.json");
    safe_write(&output, false, write_bytes(b"{}")).unwrap();
    assert_eq!(fs::read(&output).unwrap(), b"{}");
    assert_eq!(entries(dir.path()), ["project.json"]);
}

#[test]
fn existing_file_is_refused_by_default_and_keeps_its_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("out.png");
    fs::write(&output, b"previous").unwrap();
    let mut ran = false;
    let error = safe_write(&output, false, |_| {
        ran = true;
        Ok(())
    })
    .unwrap_err();
    assert!(matches!(error, MediaError::OutputExists { .. }), "{error}");
    assert!(!ran, "nothing may be written for a refused destination");
    assert_eq!(fs::read(&output).unwrap(), b"previous");
    assert_eq!(entries(dir.path()), ["out.png"]);
}

#[cfg(unix)]
#[test]
fn dangling_symlink_is_refused_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("missing-target");
    let output = dir.path().join("out.png");
    std::os::unix::fs::symlink(&target, &output).unwrap();
    let error = safe_write(&output, false, write_bytes(b"new")).unwrap_err();
    assert!(matches!(error, MediaError::OutputExists { .. }), "{error}");
    assert!(
        fs::symlink_metadata(&output)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert!(!target.exists(), "the link target must not be created");
    assert_eq!(entries(dir.path()), ["out.png"]);
}

#[cfg(unix)]
#[test]
fn overwrite_replaces_the_symlink_entry_without_following_it() {
    let outside = tempfile::tempdir().unwrap();
    let target = outside.path().join("elsewhere.png");
    fs::write(&target, b"keep me").unwrap();
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("out.png");
    std::os::unix::fs::symlink(&target, &output).unwrap();

    safe_write(&output, true, write_bytes(b"new")).unwrap();

    let meta = fs::symlink_metadata(&output).unwrap();
    assert!(meta.file_type().is_file(), "the link itself is replaced");
    assert_eq!(fs::read(&output).unwrap(), b"new");
    assert_eq!(fs::read(&target).unwrap(), b"keep me");
    assert_eq!(entries(dir.path()), ["out.png"]);
}

#[test]
fn overwrite_replaces_an_existing_file() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("out.png");
    fs::write(&output, b"previous").unwrap();
    safe_write(&output, true, write_bytes(b"new")).unwrap();
    assert_eq!(fs::read(&output).unwrap(), b"new");
    assert_eq!(entries(dir.path()), ["out.png"]);
}

#[test]
fn destination_created_during_the_write_is_never_clobbered() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("out.png");
    let racer = output.clone();
    let error = safe_write(&output, false, move |file| {
        fs::write(&racer, b"racer").unwrap();
        file.write_all(b"ours").unwrap();
        Ok(())
    })
    .unwrap_err();
    assert!(matches!(error, MediaError::OutputExists { .. }), "{error}");
    assert_eq!(fs::read(&output).unwrap(), b"racer");
    assert_eq!(entries(dir.path()), ["out.png"]);
}

#[test]
fn failed_write_leaves_the_previous_file_and_no_partial() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("out.png");
    fs::write(&output, b"previous").unwrap();
    let error = safe_write(&output, true, |file| {
        file.write_all(b"half").unwrap();
        Err(MediaError::MissingDepth)
    })
    .unwrap_err();
    assert!(matches!(error, MediaError::MissingDepth), "{error}");
    assert_eq!(fs::read(&output).unwrap(), b"previous");
    assert_eq!(entries(dir.path()), ["out.png"]);
}

#[test]
fn missing_output_directory_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("missing").join("out.png");
    let error = safe_write(&output, false, write_bytes(b"new")).unwrap_err();
    let MediaError::OutputDirectory { directory, .. } = &error else {
        panic!("expected OutputDirectory, got {error}");
    };
    assert_eq!(directory, &dir.path().join("missing"));
    assert!(error.to_string().contains("does not exist"), "{error}");
}

#[test]
fn directory_paths_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let sub = dir.path().join("sub");
    fs::create_dir(&sub).unwrap();

    let error = safe_write(&sub, true, write_bytes(b"new")).unwrap_err();
    assert!(
        matches!(error, MediaError::OutputIsDirectory { .. }),
        "{error}"
    );

    let mut trailing = sub.into_os_string();
    trailing.push("/");
    let error = safe_write(Path::new(&trailing), true, write_bytes(b"new")).unwrap_err();
    assert!(
        matches!(error, MediaError::InvalidOutputPath { .. }),
        "{error}"
    );

    let error = safe_write(Path::new(""), true, write_bytes(b"new")).unwrap_err();
    assert!(
        matches!(error, MediaError::InvalidOutputPath { .. }),
        "{error}"
    );
    assert_eq!(entries(dir.path()), ["sub"]);
    assert!(entries(&dir.path().join("sub")).is_empty());
}

#[test]
fn depth_png_round_trips_through_disk() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("depth.png");
    let values = (0..12).map(|i| i as f32 / 11.0).collect();
    let depth = DepthMap::from_f32(Size::new(4, 3), values).unwrap();

    write_depth_png(&depth, &output, false, None).unwrap();
    let loaded = load_depth(&output).unwrap();
    assert_eq!(loaded.size(), Size::new(4, 3));
    for (a, b) in loaded.values().iter().zip(depth.values()) {
        assert!((a - b).abs() <= 0.5 / 65535.0, "{a} vs {b}");
    }

    let error = write_depth_png(&depth, &output, false, None).unwrap_err();
    assert!(matches!(error, MediaError::OutputExists { .. }), "{error}");
    assert_eq!(entries(dir.path()), ["depth.png"]);
}

#[test]
fn load_errors_name_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing.png");
    let error = load_image(&missing).unwrap_err();
    assert!(matches!(error, MediaError::Io { .. }), "{error}");
    assert!(error.to_string().contains("missing.png"), "{error}");

    let garbage = dir.path().join("garbage.png");
    fs::write(&garbage, b"not an image").unwrap();
    let error = load_image(&garbage).unwrap_err();
    assert!(matches!(error, MediaError::Asset { .. }), "{error}");
    assert!(error.to_string().contains("garbage.png"), "{error}");

    let error = load_depth(&garbage).unwrap_err();
    assert!(matches!(error, MediaError::Asset { .. }), "{error}");
}

#[test]
fn load_image_decodes_png() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.png");
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, 3, 2);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer
            .write_image_data(&[
                255, 0, 0, 0, 255, 0, 0, 0, 255, 9, 9, 9, 99, 99, 99, 200, 200, 200,
            ])
            .unwrap();
    }
    fs::write(&path, bytes).unwrap();
    let image = load_image(&path).unwrap();
    assert_eq!(image.size(), Size::new(3, 2));
    assert_eq!(&image.pixels()[..8], &[255, 0, 0, 255, 0, 255, 0, 255]);
}
