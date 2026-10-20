//! The `ffmpeg` child process: availability probe, exact arguments, and stderr draining.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, Command, Stdio};
use std::thread::{self, JoinHandle};

use burns::{Size, Timeline};

use crate::MediaError;

/// Most stderr bytes kept from FFmpeg; older output is discarded but still drained.
pub(crate) const STDERR_LIMIT: usize = 64 * 1024;

/// An FFmpeg executable, run directly (never through a shell).
#[derive(Debug, Clone)]
pub(crate) struct Encoder {
    program: PathBuf,
}

impl Encoder {
    /// `ffmpeg` resolved through `PATH`.
    pub(crate) fn system() -> Self {
        Self {
            program: PathBuf::from("ffmpeg"),
        }
    }

    /// A specific executable; lets tests exercise missing and failing encoders without
    /// touching the process-wide `PATH`.
    #[cfg(test)]
    pub(crate) fn at(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
        }
    }

    pub(crate) fn program(&self) -> &Path {
        &self.program
    }

    /// Checks that the executable runs and lists the libx264 encoder.
    pub(crate) fn probe(&self) -> Result<(), MediaError> {
        let unavailable = |reason: String| MediaError::EncoderUnavailable {
            program: self.program.clone(),
            reason,
        };
        let listing = Command::new(&self.program)
            .args(["-hide_banner", "-encoders"])
            .stdin(Stdio::null())
            .stderr(Stdio::piped())
            .stdout(Stdio::piped())
            .output()
            .map_err(|error| {
                unavailable(format!(
                    "it could not be run ({error}); install FFmpeg with libx264 and put `ffmpeg` \
                     on PATH"
                ))
            })?;
        if !listing.status.success() {
            return Err(unavailable(format!(
                "`ffmpeg -encoders` exited with {}: {}",
                listing.status,
                String::from_utf8_lossy(&listing.stderr).trim()
            )));
        }
        let lists_x264 = String::from_utf8_lossy(&listing.stdout)
            .lines()
            .any(|line| line.split_whitespace().nth(1) == Some("libx264"));
        if !lists_x264 {
            return Err(unavailable(
                "this FFmpeg build has no libx264 encoder; install a build configured with \
                 --enable-gpl --enable-libx264"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    /// Starts FFmpeg reading tightly packed RGBA8 frames on stdin and writing MP4 to `temp`.
    pub(crate) fn spawn(
        &self,
        size: Size,
        timeline: Timeline,
        crf: u8,
        temp: &Path,
    ) -> Result<Child, MediaError> {
        Command::new(&self.program)
            .args(arguments(size, timeline, crf, temp))
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| MediaError::EncoderUnavailable {
                program: self.program.clone(),
                reason: format!("it could not be started ({error})"),
            })
    }
}

/// FFmpeg arguments for one export. `-y` only ever overwrites the owned temporary, and the
/// `file:` prefix keeps a path starting with `-` or containing `:` from being parsed as an
/// option or protocol.
pub(crate) fn arguments(size: Size, timeline: Timeline, crf: u8, temp: &Path) -> Vec<OsString> {
    let Size { width, height } = size;
    let fps = timeline.fps;
    let mut output = OsString::from("file:");
    output.push(temp);
    let mut args: Vec<OsString> = [
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgba",
        "-video_size",
        &format!("{width}x{height}"),
        "-framerate",
        &format!("{}/{}", fps.num, fps.den),
        "-i",
        "pipe:0",
        "-frames:v",
        &timeline.frames.to_string(),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        &crf.to_string(),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-vf",
        // `setparams` tags the frames themselves: FFmpeg 9 lets untagged frames override the
        // `-color_primaries`/`-color_trc` output options, leaving both `unknown` in the stream.
        "scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p,\
         setparams=color_primaries=bt709:color_trc=bt709",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-f",
        "mp4",
    ]
    .into_iter()
    .map(OsString::from)
    .collect();
    args.push(output);
    args
}

/// Reads `stderr` to the end on its own thread so FFmpeg never blocks on a full pipe, keeping
/// the last [`STDERR_LIMIT`] bytes.
pub(crate) fn drain(stderr: ChildStderr) -> io::Result<JoinHandle<String>> {
    thread::Builder::new()
        .name("burns ffmpeg stderr".to_owned())
        .spawn(move || drain_to_tail(stderr))
}

fn drain_to_tail(mut stderr: impl Read) -> String {
    let mut tail = VecDeque::with_capacity(STDERR_LIMIT);
    let mut truncated = false;
    let mut chunk = [0; 8192];
    loop {
        match stderr.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => {
                tail.extend(&chunk[..read]);
                if tail.len() > STDERR_LIMIT {
                    tail.drain(..tail.len() - STDERR_LIMIT);
                    truncated = true;
                }
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
    let text = String::from_utf8_lossy(tail.make_contiguous()).into_owned();
    if truncated {
        format!("[earlier output truncated]\n{text}")
    } else {
        text
    }
}
