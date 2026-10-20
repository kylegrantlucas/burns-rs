//! `burns-gui`: the native burns workbench. Opens one still image, previews its motion live on
//! the GPU, manages its depth map, and exports MP4 movies and PNG frames.

mod app;
mod dialogs;
mod inspector;
mod preview;
mod scrubber;
mod style;
mod workers;

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use eframe::egui;
use egui_wgpu::wgpu;

const USAGE: &str = "\
Usage: burns-gui [IMAGE]

Opens the burns workbench. IMAGE is an optional PNG, JPEG, or WebP still to open; without it,
the most recent image is reopened when it still exists.

Options:
  -h, --help     Print this help
  -V, --version  Print the version
";

enum Args {
    Run(Option<PathBuf>),
    Exit(ExitCode),
}

fn parse(args: impl IntoIterator<Item = OsString>) -> Args {
    let mut image = None;
    let mut options_done = false;
    for arg in args {
        if !options_done {
            match arg.to_str() {
                Some("-h" | "--help") => {
                    print!("{USAGE}");
                    return Args::Exit(ExitCode::SUCCESS);
                }
                Some("-V" | "--version") => {
                    println!("burns-gui {}", env!("CARGO_PKG_VERSION"));
                    return Args::Exit(ExitCode::SUCCESS);
                }
                Some("--") => {
                    options_done = true;
                    continue;
                }
                Some(flag) if flag.starts_with('-') && flag.len() > 1 => {
                    eprintln!("error: unknown option `{flag}`\n\n{USAGE}");
                    return Args::Exit(ExitCode::from(2));
                }
                _ => {}
            }
        }
        if image.is_some() {
            eprintln!("error: open one image at a time; burns animates a single still\n\n{USAGE}");
            return Args::Exit(ExitCode::from(2));
        }
        image = Some(PathBuf::from(arg));
    }
    Args::Run(image)
}

/// The window's device uses the same backends and limits as burns' headless renderer, so
/// anything the CLI can render, the workbench can preview and export.
fn wgpu_options() -> egui_wgpu::WgpuConfiguration {
    let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.instance_descriptor.backends = burns::renderer::HEADLESS_BACKENDS;
    setup.device_descriptor = Arc::new(|adapter: &wgpu::Adapter| {
        let supported = adapter.limits();
        let mut limits = wgpu::Limits::default().using_resolution(supported.clone());
        limits.max_buffer_size = supported.max_buffer_size;
        wgpu::DeviceDescriptor {
            label: Some("burns-gui"),
            required_limits: limits,
            ..Default::default()
        }
    });
    egui_wgpu::WgpuConfiguration {
        wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(setup),
        ..Default::default()
    }
}

fn main() -> ExitCode {
    let image = match parse(std::env::args_os().skip(1)) {
        Args::Run(image) => image,
        Args::Exit(code) => return code,
    };
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: wgpu_options(),
        viewport: egui::ViewportBuilder::default()
            .with_title("burns")
            .with_app_id("burns")
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([1040.0, 640.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    let result = eframe::run_native(
        "burns",
        options,
        Box::new(move |cc| {
            app::Workbench::new(cc, image)
                .map(|app| Box::new(app) as Box<dyn eframe::App>)
                .map_err(Into::into)
        }),
    );
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
