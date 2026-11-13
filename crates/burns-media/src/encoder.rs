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

/// Video codec and container of an export.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Codec {
    /// libx264 H.264 in MP4 at the request's CRF, `medium` preset, with faststart.
    #[default]
    H264,
    /// Lossless FFV1 version 3 in Matroska: every frame a keyframe, 4 slices with CRCs. The
    /// CRF is ignored.
    Ffv1,
}

impl Codec {
    /// The FFmpeg encoder name that must appear in `ffmpeg -encoders`.
    pub fn encoder(self) -> &'static str {
        match self {
            Self::H264 => "libx264",
            Self::Ffv1 => "ffv1",
        }
    }
}

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

    /// Checks that the executable runs and lists the encoder `codec` needs.
    pub(crate) fn probe(&self, codec: Codec) -> Result<(), MediaError> {
        let unavailable = |reason: String| MediaError::EncoderUnavailable {
            program: self.program.clone(),
            reason,
        };
        let name = codec.encoder();
        let listing = Command::new(&self.program)
            .args(["-hide_banner", "-encoders"])
            .stdin(Stdio::null())
            .stderr(Stdio::piped())
            .stdout(Stdio::piped())
            .output()
            .map_err(|error| {
                unavailable(format!(
                    "it could not be run ({error}); install FFmpeg with {name} and put `ffmpeg` \
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
        let listed = String::from_utf8_lossy(&listing.stdout)
            .lines()
            .any(|line| line.split_whitespace().nth(1) == Some(name));
        if !listed {
            let remedy = match codec {
                Codec::H264 => "; install a build configured with --enable-gpl --enable-libx264",
                Codec::Ffv1 => "",
            };
            return Err(unavailable(format!(
                "this FFmpeg build has no {name} encoder{remedy}"
            )));
        }
        Ok(())
    }

    /// Starts FFmpeg reading tightly packed RGBA8 frames on stdin and writing `codec`'s
    /// container to `temp`.
    pub(crate) fn spawn(
        &self,
        size: Size,
        timeline: Timeline,
        codec: Codec,
        crf: u8,
        temp: &Path,
    ) -> Result<Child, MediaError> {
        Command::new(&self.program)
            .args(arguments(size, timeline, codec, crf, temp))
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
pub(crate) fn arguments(
    size: Size,
    timeline: Timeline,
    codec: Codec,
    crf: u8,
    temp: &Path,
) -> Vec<OsString> {
    let Size { width, height } = size;
    let fps = timeline.fps;
    let video_size = format!("{width}x{height}");
    let framerate = format!("{}/{}", fps.num, fps.den);
    let frames = timeline.frames.to_string();
    let crf = crf.to_string();
    let mut output = OsString::from("file:");
    output.push(temp);
    let mut args = vec![
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgba",
        "-video_size",
        &video_size,
        "-framerate",
        &framerate,
        "-i",
        "pipe:0",
        "-frames:v",
        &frames,
        "-an",
        "-c:v",
    ];
    match codec {
        Codec::H264 => args.extend(["libx264", "-preset", "medium", "-crf", &crf]),
        Codec::Ffv1 => args.extend([
            "ffv1",
            "-level",
            "3",
            "-g",
            "1",
            "-slices",
            "4",
            "-slicecrc",
            "1",
        ]),
    }
    args.extend(["-pix_fmt", "yuv420p"]);
    if codec == Codec::H264 {
        args.extend(["-movflags", "+faststart"]);
    }
    args.extend([
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
        match codec {
            Codec::H264 => "mp4",
            Codec::Ffv1 => "matroska",
        },
    ]);
    let mut args: Vec<OsString> = args.into_iter().map(OsString::from).collect();
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

#[cfg(test)]
mod tests {
    use burns::FrameRate;

    use super::*;

    fn args(codec: Codec) -> Vec<String> {
        let timeline = Timeline::new(FrameRate::new(30, 1), 30).unwrap();
        arguments(
            Size::new(320, 180),
            timeline,
            codec,
            20,
            Path::new("out.tmp"),
        )
        .into_iter()
        .map(|arg| arg.into_string().unwrap())
        .collect()
    }

    #[test]
    fn ffv1_writes_matroska_without_x264_options() {
        let ffv1 = args(Codec::Ffv1);
        for expected in ["ffv1", "matroska"] {
            assert!(ffv1.iter().any(|arg| arg == expected), "{ffv1:?}");
        }
        for refused in ["libx264", "+faststart"] {
            assert!(!ffv1.iter().any(|arg| arg == refused), "{ffv1:?}");
        }
    }
}
