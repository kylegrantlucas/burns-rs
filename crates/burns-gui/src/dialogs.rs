//! Native file dialogs that never block the UI. The dialog future is created on the UI thread
//! (where macOS presents it as a window sheet) and polled each frame with a waker that asks
//! egui to repaint when the user answers.

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

use eframe::egui;

/// What the chosen path is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    OpenImage,
    LoadDepth,
    SaveDepth,
    LoadProject,
    SaveProject,
    ChooseMovie,
    ChooseFrame,
}

type DialogFuture = Pin<Box<dyn Future<Output = Option<rfd::FileHandle>> + Send>>;

pub struct PendingDialog {
    pub purpose: Purpose,
    future: DialogFuture,
    waker: Waker,
}

struct Repaint(egui::Context);

impl Wake for Repaint {
    fn wake(self: Arc<Self>) {
        self.0.request_repaint();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.request_repaint();
    }
}

const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp"];

impl PendingDialog {
    /// Opens the dialog for `purpose`, attached to the workbench window.
    pub fn open(
        ctx: &egui::Context,
        frame: &eframe::Frame,
        purpose: Purpose,
        directory: Option<PathBuf>,
        file_name: Option<String>,
    ) -> Self {
        let mut dialog = rfd::AsyncFileDialog::new().set_parent(frame);
        if let Some(directory) = directory {
            dialog = dialog.set_directory(directory);
        }
        if let Some(name) = file_name {
            dialog = dialog.set_file_name(name);
        }
        let future: DialogFuture = match purpose {
            Purpose::OpenImage => Box::pin(
                dialog
                    .set_title("Open image")
                    .add_filter("Images (PNG, JPEG, WebP)", IMAGE_EXTENSIONS)
                    .pick_file(),
            ),
            Purpose::LoadDepth => Box::pin(
                dialog
                    .set_title("Load depth map")
                    .add_filter("Grayscale PNG", &["png"])
                    .pick_file(),
            ),
            Purpose::SaveDepth => Box::pin(
                dialog
                    .set_title("Save raw depth map")
                    .add_filter("16-bit grayscale PNG", &["png"])
                    .save_file(),
            ),
            Purpose::LoadProject => Box::pin(
                dialog
                    .set_title("Open project")
                    .add_filter("burns project", &["json"])
                    .pick_file(),
            ),
            Purpose::SaveProject => Box::pin(
                dialog
                    .set_title("Save project")
                    .add_filter("burns project", &["json"])
                    .save_file(),
            ),
            Purpose::ChooseMovie => Box::pin(
                dialog
                    .set_title("Render movie to")
                    .add_filter("MP4 movie", &["mp4"])
                    .save_file(),
            ),
            Purpose::ChooseFrame => Box::pin(
                dialog
                    .set_title("Save frame to")
                    .add_filter("PNG image", &["png"])
                    .save_file(),
            ),
        };
        Self {
            purpose,
            future,
            waker: Waker::from(Arc::new(Repaint(ctx.clone()))),
        }
    }

    /// `Ready(None)` when the user dismissed the dialog.
    pub fn poll(&mut self) -> Poll<Option<PathBuf>> {
        let mut context = Context::from_waker(&self.waker);
        self.future
            .as_mut()
            .poll(&mut context)
            .map(|handle| handle.map(|handle| handle.path().to_owned()))
    }
}
