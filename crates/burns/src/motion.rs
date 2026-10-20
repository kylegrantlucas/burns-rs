//! Per-frame motion evaluation: the single inverse map from an output pixel to
//! the virtual output frame that is then cover-cropped into the source.
//!
//! The visual stack is camera/depth, then shake, then warp. Rendering inverts it
//! in one pass per pixel:
//!
//! 1. **Warp**: add the inverse warp displacement to the output UV.
//! 2. **Shake**: `R(-roll) * (q / k + delta)` in metric coordinates
//!    `q = (A * (u - 0.5), v - 0.5)` (y down), where `k` is a constant safety
//!    scale that keeps every warp and shake displacement inside the frame.
//! 3. **Camera**: a planar view/tilt map that stays inside the virtual frame, or
//!    the depth heightfield ray traversal, whose result is mirrored.
//!
//! The virtual frame has the project output aspect `A`; [`crate::CoverMap`] maps it
//! into the source exactly once. Geometry depends only on the project and the frame
//! index, never on the render target size or on previously evaluated frames.
//!
//! Positive roll is clockwise on screen. All trigonometry that depends only on
//! time runs here, once per frame; the renderer evaluates only the spatial warp
//! sines and the depth traversal per pixel.

use std::f64::consts::{PI, TAU};

use crate::settings::{
    CameraMotion, ConfigError, DepthMotion, Easing, MotionProject, ShakeProfile, WarpSettings,
};
use crate::timeline::FrameTime;

/// Row-major 3x3 matrix acting on homogeneous UV column vectors `[u, v, 1]`.
pub type Mat3 = [[f64; 3]; 3];

/// Shake offsets are authored in pixels of a 3840x2160 reference frame.
pub const SHAKE_REFERENCE_SIZE: [f64; 2] = [3840.0, 2160.0];

/// Seed component id of camera shake harmonics.
pub const SHAKE_COMPONENT: u32 = 1;
/// Seed component id of the ripple warp.
pub const RIPPLE_COMPONENT: u32 = 2;
/// Seed component id of the heat-shimmer warp.
pub const SHIMMER_COMPONENT: u32 = 3;

/// Coarse depth-traversal intervals between the near slab plane and the far plane.
pub const DEPTH_COARSE_STEPS: u32 = 64;
/// Bisection iterations refining the first coarse surface crossing.
pub const DEPTH_REFINE_STEPS: u32 = 8;

/// Identity transform.
pub const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

// ---------------------------------------------------------------------------
// Shake spectra

/// One sinusoidal shake term: `amplitude * sin(2 pi * frequency_hz * t + phase)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Harmonic {
    /// Reference pixels for x/y; radians for roll.
    pub amplitude: f64,
    pub frequency_hz: f64,
    /// Base phase in radians, before the seed offset.
    pub phase: f64,
}

const fn harmonic(amplitude: f64, frequency_hz: f64, phase: f64) -> Harmonic {
    Harmonic {
        amplitude,
        frequency_hz,
        phase,
    }
}

/// Harmonic content of a shake profile at `amount = 1`, `speed = 1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShakeSpectrum {
    /// Nominal safety scale `k` while this profile is active.
    pub nominal_scale: f64,
    /// Horizontal offset in reference pixels.
    pub x: &'static [Harmonic],
    /// Vertical offset in reference pixels (down is positive).
    pub y: &'static [Harmonic],
    /// Roll in radians (clockwise is positive).
    pub roll: &'static [Harmonic],
}

impl ShakeSpectrum {
    const fn axes(&self) -> [&'static [Harmonic]; 3] {
        [self.x, self.y, self.roll]
    }
}

static SHAKE_SPECTRUM: ShakeSpectrum = ShakeSpectrum {
    nominal_scale: 1.08,
    x: &[
        harmonic(48.0, 0.31, 0.0),
        harmonic(22.0, 0.83, 0.7),
        harmonic(6.0, 3.1, 2.1),
    ],
    y: &[
        harmonic(36.0, 0.27, 1.9),
        harmonic(18.0, 0.71, 0.3),
        harmonic(5.0, 3.7, 0.9),
    ],
    roll: &[harmonic(0.0035, 0.23, 1.1), harmonic(0.0015, 1.3, 0.0)],
};

static VIBRATE_SPECTRUM: ShakeSpectrum = ShakeSpectrum {
    nominal_scale: 1.14,
    x: &[
        harmonic(30.0, 0.9, 0.0),
        harmonic(28.0, 2.9, 0.7),
        harmonic(26.0, 7.7, 2.1),
        harmonic(22.0, 11.3, 0.2),
        harmonic(14.0, 13.1, 1.6),
    ],
    y: &[
        harmonic(26.0, 0.7, 1.9),
        harmonic(24.0, 3.3, 0.3),
        harmonic(22.0, 8.3, 0.9),
        harmonic(18.0, 10.7, 2.6),
        harmonic(12.0, 12.7, 0.5),
    ],
    roll: &[harmonic(0.004, 1.7, 0.4), harmonic(0.003, 9.1, 1.3)],
};

static VIBRATE_TIGHT_SPECTRUM: ShakeSpectrum = ShakeSpectrum {
    nominal_scale: 1.03,
    x: &[
        harmonic(9.0, 13.7, 0.2),
        harmonic(7.0, 12.1, 1.6),
        harmonic(5.0, 9.7, 2.1),
        harmonic(3.0, 6.3, 0.7),
    ],
    y: &[
        harmonic(7.0, 12.9, 2.6),
        harmonic(6.0, 11.3, 0.5),
        harmonic(4.0, 8.9, 0.9),
        harmonic(2.0, 5.9, 0.3),
    ],
    roll: &[harmonic(0.0006, 11.9, 0.8)],
};

impl ShakeProfile {
    /// Harmonic table of this profile.
    pub fn spectrum(self) -> &'static ShakeSpectrum {
        match self {
            Self::Shake => &SHAKE_SPECTRUM,
            Self::Vibrate => &VIBRATE_SPECTRUM,
            Self::VibrateTight => &VIBRATE_TIGHT_SPECTRUM,
        }
    }
}

// ---------------------------------------------------------------------------
// Seeds

/// One SplitMix64 output for input state `x`.
pub const fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// High 53 bits as a uniform value in `[0, 1)`.
fn unit(hash: u64) -> f64 {
    (hash >> 11) as f64 / (1u64 << 53) as f64
}

/// Deterministic phase offset in radians for one harmonic of one component.
///
/// `key = (component << 32) | harmonic`; the offset is
/// `2 pi * (unit(splitmix64(key ^ seed)) - unit(splitmix64(key)))`, so seed `0`
/// yields exactly `0` for every harmonic.
pub fn seed_phase(seed: u32, component: u32, harmonic: u32) -> f64 {
    let key = (u64::from(component) << 32) | u64::from(harmonic);
    TAU * (unit(splitmix64(key ^ u64::from(seed))) - unit(splitmix64(key)))
}

/// Seed harmonic id of shake term `term` on `axis` (`0` x, `1` y, `2` roll).
pub const fn shake_harmonic_id(profile: ShakeProfile, axis: u32, term: u32) -> u32 {
    profile.index() * 32 + axis * 8 + term
}

// ---------------------------------------------------------------------------
// Pure geometry helpers

/// Easing curve applied to endpoint-inclusive progress `u`.
pub fn ease(easing: Easing, progress: f64) -> f64 {
    match easing {
        Easing::Cosine => (1.0 - (PI * progress).cos()) / 2.0,
        Easing::Linear => progress,
    }
}

/// Triangle-wave mirror `m(x) = 1 - |1 - (x - 2 floor(x / 2))|`: identity on
/// `[0, 1]`, reflecting beyond it.
pub fn mirror(x: f64) -> f64 {
    1.0 - (1.0 - (x - 2.0 * (x / 2.0).floor())).abs()
}

/// Applies a projective matrix to a UV point.
pub fn apply(matrix: &Mat3, uv: [f64; 2]) -> [f64; 2] {
    let row = |r: usize| matrix[r][0] * uv[0] + matrix[r][1] * uv[1] + matrix[r][2];
    let w = row(2);
    [row(0) / w, row(1) / w]
}

/// Matrix product `a * b` (apply `b` first).
pub fn multiply(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[r][k] * b[k][c]).sum();
        }
    }
    out
}

/// Tilt quad corners `[TL, TR, BL, BR]` in virtual-frame UV for eased progress
/// `p`, interpolated from the unit-square corners by `amount`.
pub fn tilt_corners(amount: f64, p: f64) -> [[f64; 2]; 4] {
    let q = 1.0 - 2.0 * p;
    let target = [
        [0.06 + 0.03 * p, 0.07 + 0.03 * p + 0.06 * q],
        [0.94 - 0.03 * p, 0.07 + 0.03 * p - 0.06 * q],
        [0.06 + 0.03 * p + 0.02 * q, 0.93 - 0.03 * p - 0.06 * q],
        [0.94 - 0.03 * p + 0.02 * q, 0.93 - 0.03 * p + 0.06 * q],
    ];
    let unit = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    std::array::from_fn(|i| {
        [
            unit[i][0] + amount * (target[i][0] - unit[i][0]),
            unit[i][1] + amount * (target[i][1] - unit[i][1]),
        ]
    })
}

/// Projective homography mapping the unit square onto the quad `[TL, TR, BL, BR]`:
/// `(0,0)`, `(1,0)`, `(0,1)`, `(1,1)` land exactly on those corners.
pub fn square_to_quad(corners: [[f64; 2]; 4]) -> Mat3 {
    let [p0, p1, p3, p2] = corners;
    let sx = p0[0] - p1[0] + p2[0] - p3[0];
    let sy = p0[1] - p1[1] + p2[1] - p3[1];
    let (dx1, dx2) = (p1[0] - p2[0], p3[0] - p2[0]);
    let (dy1, dy2) = (p1[1] - p2[1], p3[1] - p2[1]);
    let det = dx1 * dy2 - dx2 * dy1;
    let g = (sx * dy2 - dx2 * sy) / det;
    let h = (dx1 * sy - sx * dy1) / det;
    [
        [p1[0] - p0[0] + g * p1[0], p3[0] - p0[0] + h * p3[0], p0[0]],
        [p1[1] - p0[1] + g * p1[1], p3[1] - p0[1] + h * p3[1], p0[1]],
        [g, h, 1.0],
    ]
}

/// Constant safety scale `k` that keeps warp and shake inside the frame.
///
/// `shake_x`/`shake_y` are summed absolute shake amplitudes in metric units
/// (image heights), `roll_bound` the summed absolute shake roll in radians, and
/// `warp_bound` the active warp amplitude in image heights. Errors when the
/// displacement cannot be covered at this aspect.
pub fn safety_scale(
    aspect: f64,
    nominal: f64,
    shake_x: f64,
    shake_y: f64,
    roll_bound: f64,
    warp_bound: f64,
) -> Result<f64, ConfigError> {
    let (c, s) = (1.0, roll_bound.sin());
    let denominator_x = aspect - 2.0 * (c * shake_x + s * shake_y);
    let denominator_y = 1.0 - 2.0 * (s * shake_x + c * shake_y);
    if denominator_x <= 0.0 || denominator_y <= 0.0 {
        return Err(ConfigError::invalid(
            "shake.amount",
            format!(
                "unsupported displacement: shake of up to {shake_x:.4} x {shake_y:.4} image heights \
                 with {roll_bound:.4} rad roll cannot stay inside a {aspect:.4}:1 frame; lower \
                 shake.amount or use a wider/taller output"
            ),
        ));
    }
    let kx = (aspect * c + s + 2.0 * warp_bound * (c + s)) / denominator_x;
    let ky = (aspect * s + c + 2.0 * warp_bound * (s + c)) / denominator_y;
    Ok(nominal.max(kx).max(ky))
}

/// Geometry feasibility shared by configuration commits and the evaluator.
pub(crate) fn project_safety_scale(
    project: &MotionProject,
    aspect: f64,
) -> Result<f64, ConfigError> {
    let shake = &project.shake;
    let (nominal, bounds) = if shake.enabled && shake.amount > 0.0 {
        let spectrum = shake.profile.spectrum();
        let amount = f64::from(shake.amount);
        let sum = |terms: &[Harmonic]| {
            terms
                .iter()
                .map(|term| (amount * term.amplitude).abs())
                .sum::<f64>()
        };
        (
            spectrum.nominal_scale,
            [
                aspect * sum(spectrum.x) / SHAKE_REFERENCE_SIZE[0],
                sum(spectrum.y) / SHAKE_REFERENCE_SIZE[1],
                sum(spectrum.roll),
            ],
        )
    } else {
        (1.0, [0.0; 3])
    };
    let warp = match project.warp {
        WarpSettings::Ripple {
            enabled: true,
            amplitude,
            ..
        }
        | WarpSettings::HeatShimmer {
            enabled: true,
            amplitude,
            ..
        } => f64::from(amplitude),
        _ => 0.0,
    };
    safety_scale(aspect, nominal, bounds[0], bounds[1], bounds[2], warp)
}

/// Metric depth-plane coordinates `g = ((2u - 1) A, 1 - 2v)` (y up) of a UV point.
pub fn uv_to_depth_xy(uv: [f64; 2], aspect: f64) -> [f64; 2] {
    [(2.0 * uv[0] - 1.0) * aspect, 1.0 - 2.0 * uv[1]]
}

/// Inverse of [`uv_to_depth_xy`]: `((x / A + 1) / 2, (1 - y) / 2)`.
pub fn depth_xy_to_uv(xy: [f64; 2], aspect: f64) -> [f64; 2] {
    [(xy[0] / aspect + 1.0) / 2.0, (1.0 - xy[1]) / 2.0]
}

// ---------------------------------------------------------------------------
// Evaluated poses

/// Resolved 2D camera view for one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedView {
    /// Effective zoom, at least the project's roll-coverage minimum.
    pub zoom: f64,
    /// Crop center in virtual-frame UV, clamped so the rolled crop stays inside.
    pub center: [f64; 2],
    /// Clockwise roll in radians.
    pub roll: f64,
}

impl ResolvedView {
    /// Inverse map from output UV to virtual-frame UV.
    pub fn matrix(&self, aspect: f64) -> Mat3 {
        let (s, c) = self.roll.sin_cos();
        let z = self.zoom;
        let k = [[c / z, s / (aspect * z)], [-aspect * s / z, c / z]];
        [
            [k[0][0], k[0][1], self.center[0] - 0.5 * (k[0][0] + k[0][1])],
            [k[1][0], k[1][1], self.center[1] - 0.5 * (k[1][0] + k[1][1])],
            [0.0, 0.0, 1.0],
        ]
    }
}

/// Depth camera state for one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthPose {
    /// Heightfield relief `h`.
    pub height: f64,
    /// Plane `f` that the projection mix does not move.
    pub focus: f64,
    /// Plane `s` that the offset does not move.
    pub steady: f64,
    pub zoom: f64,
    /// Projection mix `i` after blending by `amount`.
    pub projection_mix: f64,
    /// Camera offset `P` after scaling by `amount`.
    pub offset: [f64; 2],
    /// Treat `1 - raw` as the depth.
    pub invert: bool,
}

/// Scalar coefficients of the depth ray: `O = P + origin_scale g`,
/// `E = far_scale g + P offset_far`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthCoefficients {
    pub origin_scale: f64,
    pub far_scale: f64,
    pub offset_far: f64,
}

impl DepthPose {
    pub fn coefficients(&self) -> DepthCoefficients {
        let focus_plane = 1.0 - self.height * self.focus;
        let steady_plane = 1.0 - self.height * self.steady;
        let i = self.projection_mix;
        DepthCoefficients {
            origin_scale: self.zoom * i,
            far_scale: self.zoom * (i + (1.0 - i) / focus_plane),
            offset_far: 1.0 - 1.0 / steady_plane,
        }
    }

    /// Ray origin `O` and far-plane point `E` for depth-plane coordinates `g`.
    pub fn ray(&self, g: [f64; 2]) -> ([f64; 2], [f64; 2]) {
        let k = self.coefficients();
        let p = self.offset;
        (
            [p[0] + k.origin_scale * g[0], p[1] + k.origin_scale * g[1]],
            [
                k.far_scale * g[0] + p[0] * k.offset_far,
                k.far_scale * g[1] + p[1] * k.offset_far,
            ],
        )
    }

    /// Heightfield traversal for metric `g`, returning the hit point in depth-plane
    /// coordinates. `raw_depth` receives mirrored virtual-frame UV and returns the
    /// raw (uninverted) map value there.
    ///
    /// Scans `Z` from `1 - h` to `1` in [`DEPTH_COARSE_STEPS`] intervals, accepting
    /// a crossing at the starting endpoint, then bisects [`DEPTH_REFINE_STEPS`]
    /// times. `Z = 1` always crosses, so the far plane is a valid hit. With `h = 0`
    /// the result is the flat projection `E`.
    pub fn trace(
        &self,
        aspect: f64,
        g: [f64; 2],
        mut raw_depth: impl FnMut([f64; 2]) -> f64,
    ) -> [f64; 2] {
        let (origin, far) = self.ray(g);
        let height = self.height;
        if height <= 0.0 {
            return far;
        }
        let direction = [far[0] - origin[0], far[1] - origin[1]];
        let at = |z: f64| [origin[0] + z * direction[0], origin[1] + z * direction[1]];
        let mut reached = |z: f64| {
            let uv = depth_xy_to_uv(at(z), aspect);
            let raw = raw_depth([mirror(uv[0]), mirror(uv[1])]);
            let depth = if self.invert { 1.0 - raw } else { raw };
            z >= 1.0 - height * depth
        };
        let start = 1.0 - height;
        if reached(start) {
            return at(start);
        }
        let mut near = start;
        let mut hit = 1.0;
        for step in 1..=DEPTH_COARSE_STEPS {
            let z = if step == DEPTH_COARSE_STEPS {
                1.0
            } else {
                start + height * f64::from(step) / f64::from(DEPTH_COARSE_STEPS)
            };
            if reached(z) {
                hit = z;
                break;
            }
            near = z;
        }
        for _ in 0..DEPTH_REFINE_STEPS {
            let mid = 0.5 * (near + hit);
            if reached(mid) {
                hit = mid;
            } else {
                near = mid;
            }
        }
        at(hit)
    }
}

/// Camera stage of one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraPose {
    /// Zoom, crop center, and roll (still, transform, breathing, figure eight).
    View(ResolvedView),
    /// Perspective tilt: output square to quad `[TL, TR, BL, BR]`.
    Tilt {
        corners: [[f64; 2]; 4],
        homography: Mat3,
    },
    /// Depth heightfield parallax; needs a depth map.
    Depth(DepthPose),
}

impl CameraPose {
    /// Planar inverse map, or `None` for the depth camera.
    pub fn matrix(&self, aspect: f64) -> Option<Mat3> {
        match self {
            Self::View(view) => Some(view.matrix(aspect)),
            Self::Tilt { homography, .. } => Some(*homography),
            Self::Depth(_) => None,
        }
    }
}

/// Shake stage of one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShakePose {
    /// Constant safety scale `k` of the project.
    pub scale: f64,
    /// Inverse offset `delta` in metric units (image heights).
    pub offset: [f64; 2],
    /// Clockwise roll in radians.
    pub roll: f64,
}

impl ShakePose {
    /// Inverse map `R(-roll) (q / k + delta)` expressed on UV.
    pub fn matrix(&self, aspect: f64) -> Mat3 {
        let (s, c) = self.roll.sin_cos();
        let k = self.scale;
        let m = [[c / k, s / (aspect * k)], [-aspect * s / k, c / k]];
        let [dx, dy] = self.offset;
        let t = [(c * dx + s * dy) / aspect, -s * dx + c * dy];
        [
            [m[0][0], m[0][1], 0.5 - 0.5 * (m[0][0] + m[0][1]) + t[0]],
            [m[1][0], m[1][1], 0.5 - 0.5 * (m[1][0] + m[1][1]) + t[1]],
            [0.0, 0.0, 1.0],
        ]
    }
}

/// Warp stage of one frame, with its time-dependent phases folded in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WarpField {
    None,
    /// Inverse offset `amplitude * smoothstep(0, wavelength / 4, r) *
    /// sin(2 pi r / wavelength - phase) * rvec / r` (metric, x divided by `A`).
    Ripple {
        center: [f64; 2],
        amplitude: f64,
        wavelength: f64,
        /// `2 pi * cycles * tau - seed phase`, wrapped to `[0, 2 pi)`.
        phase: f64,
    },
    /// Rising heat shimmer, strongest at the bottom of the frame.
    HeatShimmer {
        amplitude: f64,
        spatial_scale: f64,
        /// Temporal plus seed phases of the 6k, 11k (vertical) and 4k
        /// (horizontal) terms, wrapped to `[0, 2 pi)`.
        phases: [f64; 3],
    },
}

impl WarpField {
    /// Inverse UV displacement at output `uv`.
    pub fn offset(&self, uv: [f64; 2], aspect: f64) -> [f64; 2] {
        match *self {
            Self::None => [0.0, 0.0],
            Self::Ripple {
                center,
                amplitude,
                wavelength,
                phase,
            } => {
                let radial = [(uv[0] - center[0]) * aspect, uv[1] - center[1]];
                let radius = radial[0].hypot(radial[1]);
                if radius <= 0.0 {
                    return [0.0, 0.0];
                }
                let wave = amplitude
                    * smoothstep(0.0, wavelength / 4.0, radius)
                    * (TAU * radius / wavelength - phase).sin()
                    / radius;
                [radial[0] * wave / aspect, radial[1] * wave]
            }
            Self::HeatShimmer {
                amplitude,
                spatial_scale: k,
                phases,
            } => {
                let envelope = 0.25 + 0.75 * uv[1] * uv[1];
                let dy = amplitude
                    * envelope
                    * (0.65 * (TAU * 6.0 * k * uv[0] + phases[0]).sin()
                        + 0.35 * (TAU * 11.0 * k * uv[0] + phases[1]).sin());
                let dx = 0.35 * amplitude * envelope * (TAU * 4.0 * k * uv[1] + phases[2]).sin();
                [dx / aspect, dy]
            }
        }
    }
}

fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Complete motion state of one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvaluatedMotion {
    pub time: FrameTime,
    /// Project output aspect `A = width / height`.
    pub aspect: f64,
    pub warp: WarpField,
    pub shake: ShakePose,
    pub camera: CameraPose,
}

impl EvaluatedMotion {
    /// Combined planar inverse map applied after the warp: camera after shake,
    /// or shake alone for the depth camera (whose traversal follows).
    pub fn frame_matrix(&self) -> Mat3 {
        let shake = self.shake.matrix(self.aspect);
        match self.camera.matrix(self.aspect) {
            Some(camera) => multiply(&camera, &shake),
            None => shake,
        }
    }

    /// Maps output UV to the virtual-frame UV whose cover-mapped source pixel is
    /// shown there. Planar cameras stay inside `[0, 1]`; the depth result is
    /// mirrored. `raw_depth` is queried only for the depth camera, with mirrored
    /// virtual-frame UV, and returns the raw map value.
    pub fn map_uv(&self, uv: [f64; 2], raw_depth: impl FnMut([f64; 2]) -> f64) -> [f64; 2] {
        let offset = self.warp.offset(uv, self.aspect);
        let warped = [uv[0] + offset[0], uv[1] + offset[1]];
        let framed = apply(&self.frame_matrix(), warped);
        match &self.camera {
            CameraPose::Depth(depth) => {
                let hit = depth.trace(self.aspect, uv_to_depth_xy(framed, self.aspect), raw_depth);
                let virtual_uv = depth_xy_to_uv(hit, self.aspect);
                [mirror(virtual_uv[0]), mirror(virtual_uv[1])]
            }
            _ => framed,
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluator

#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct SeededTerm {
    amplitude: f64,
    angular: f64,
    phase: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SeededAxis {
    terms: [SeededTerm; 5],
    len: usize,
}

impl SeededAxis {
    fn terms(&self) -> &[SeededTerm] {
        &self.terms[..self.len]
    }
}

/// Project-constant motion state: validated settings, seeded phases, safety scale
/// and roll coverage. Build once per project change; [`Self::evaluate`] is
/// stateless and accepts frames in any order.
#[derive(Debug, Clone, PartialEq)]
pub struct MotionEvaluator {
    project: MotionProject,
    aspect: f64,
    safety_scale: f64,
    min_camera_zoom: f64,
    shake: Option<[SeededAxis; 3]>,
    warp_phases: [f64; 3],
}

impl MotionEvaluator {
    /// Validates `project` and precomputes everything that does not depend on the frame.
    pub fn new(project: &MotionProject) -> Result<Self, ConfigError> {
        let (aspect, safety_scale) = project.validated_geometry()?;

        let settings = project.shake;
        let shake: Option<[SeededAxis; 3]> =
            (settings.enabled && settings.amount > 0.0).then(|| {
                let amount = f64::from(settings.amount);
                let speed = f64::from(settings.speed);
                let axes = settings.profile.spectrum().axes();
                std::array::from_fn(|axis| {
                    let mut seeded = SeededAxis {
                        terms: [SeededTerm::default(); 5],
                        len: axes[axis].len(),
                    };
                    assert!(
                        seeded.len <= seeded.terms.len(),
                        "shake spectrum exceeds fixed storage"
                    );
                    for (term, h) in axes[axis].iter().enumerate() {
                        seeded.terms[term] = SeededTerm {
                            amplitude: amount * h.amplitude,
                            angular: TAU * h.frequency_hz * speed,
                            phase: h.phase
                                + seed_phase(
                                    project.seed,
                                    SHAKE_COMPONENT,
                                    shake_harmonic_id(settings.profile, axis as u32, term as u32),
                                ),
                        };
                    }
                    seeded
                })
            });

        let max_roll = match project.camera {
            CameraMotion::Still { view } => f64::from(view.roll_degrees).abs(),
            CameraMotion::Transform { start, end, .. } => f64::from(start.roll_degrees)
                .abs()
                .max(f64::from(end.roll_degrees).abs()),
            _ => 0.0,
        }
        .to_radians();
        let sin = max_roll.sin();
        let min_camera_zoom = (1.0 + sin / aspect).max(1.0 + aspect * sin);

        let warp_phases = match project.warp {
            WarpSettings::Ripple { .. } => {
                [seed_phase(project.seed, RIPPLE_COMPONENT, 0), 0.0, 0.0]
            }
            WarpSettings::HeatShimmer { .. } => {
                std::array::from_fn(|i| seed_phase(project.seed, SHIMMER_COMPONENT, i as u32))
            }
        };

        Ok(Self {
            project: project.clone(),
            aspect,
            safety_scale,
            min_camera_zoom,
            shake,
            warp_phases,
        })
    }

    /// The validated project this evaluator was built from.
    pub fn project(&self) -> &MotionProject {
        &self.project
    }

    /// Output aspect `A`.
    pub fn aspect(&self) -> f64 {
        self.aspect
    }

    /// Constant shake/warp safety scale `k`.
    pub fn safety_scale(&self) -> f64 {
        self.safety_scale
    }

    /// Minimum camera zoom that keeps the largest camera roll covered.
    pub fn min_camera_zoom(&self) -> f64 {
        self.min_camera_zoom
    }

    /// Motion of frame `frame`; rejects `frame >= timeline.frames`.
    pub fn evaluate(&self, frame: u32) -> Result<EvaluatedMotion, ConfigError> {
        let time = self.project.timeline.frame_time(frame)?;
        Ok(EvaluatedMotion {
            time,
            aspect: self.aspect,
            warp: self.warp_field(time.phase),
            shake: self.shake_pose(time.seconds),
            camera: self.camera_pose(time),
        })
    }

    fn shake_pose(&self, seconds: f64) -> ShakePose {
        let Some(axes) = &self.shake else {
            return ShakePose {
                scale: self.safety_scale,
                offset: [0.0, 0.0],
                roll: 0.0,
            };
        };
        let sum = |axis: &SeededAxis| -> f64 {
            axis.terms()
                .iter()
                .map(|t| t.amplitude * (t.angular * seconds + t.phase).sin())
                .sum()
        };
        ShakePose {
            scale: self.safety_scale,
            offset: [
                self.aspect * sum(&axes[0]) / SHAKE_REFERENCE_SIZE[0],
                sum(&axes[1]) / SHAKE_REFERENCE_SIZE[1],
            ],
            roll: sum(&axes[2]),
        }
    }

    fn warp_field(&self, tau: f64) -> WarpField {
        match self.project.warp {
            WarpSettings::Ripple {
                enabled: true,
                amplitude,
                wavelength,
                center,
                cycles,
            } if amplitude > 0.0 => WarpField::Ripple {
                center: center.map(f64::from),
                amplitude: f64::from(amplitude),
                wavelength: f64::from(wavelength),
                phase: (TAU * f64::from(cycles) * tau - self.warp_phases[0]).rem_euclid(TAU),
            },
            WarpSettings::HeatShimmer {
                enabled: true,
                amplitude,
                spatial_scale,
                cycles,
            } if amplitude > 0.0 => {
                let b = f64::from(cycles) * tau;
                let temporal = [TAU * b, -2.0 * TAU * b, TAU * b];
                WarpField::HeatShimmer {
                    amplitude: f64::from(amplitude),
                    spatial_scale: f64::from(spatial_scale),
                    phases: std::array::from_fn(|i| {
                        (temporal[i] + self.warp_phases[i]).rem_euclid(TAU)
                    }),
                }
            }
            _ => WarpField::None,
        }
    }

    fn camera_pose(&self, time: FrameTime) -> CameraPose {
        let progress = time.progress;
        let c = TAU * time.phase;
        let view = |zoom: f64, x: f64, y: f64, roll_degrees: f64| {
            CameraPose::View(self.resolve_view(zoom, x, y, roll_degrees))
        };
        match self.project.camera {
            CameraMotion::Still { view: v } => {
                view(v.zoom.into(), v.x.into(), v.y.into(), v.roll_degrees.into())
            }
            CameraMotion::Transform { start, end, easing } => {
                let p = ease(easing, progress);
                let lerp = |a: f32, b: f32| f64::from(a) + p * (f64::from(b) - f64::from(a));
                view(
                    lerp(start.zoom, end.zoom),
                    lerp(start.x, end.x),
                    lerp(start.y, end.y),
                    lerp(start.roll_degrees, end.roll_degrees),
                )
            }
            CameraMotion::Tilt { amount, easing } => {
                let corners = tilt_corners(amount.into(), ease(easing, progress));
                CameraPose::Tilt {
                    corners,
                    homography: square_to_quad(corners),
                }
            }
            CameraMotion::Breathing {
                base_zoom,
                amplitude,
                cycles,
            } => {
                let zoom = f64::from(base_zoom)
                    + f64::from(amplitude) * (0.5 - 0.5 * (c * f64::from(cycles)).cos());
                view(zoom, 0.5, 0.5, 0.0)
            }
            CameraMotion::FigureEight {
                zoom,
                amplitude,
                cycles,
            } => {
                let angle = c * f64::from(cycles);
                view(
                    zoom.into(),
                    0.5 + f64::from(amplitude[0]) * angle.sin(),
                    0.5 + f64::from(amplitude[1]) * (2.0 * angle).sin(),
                    0.0,
                )
            }
            CameraMotion::Depth {
                motion,
                height,
                steady,
                focus,
                zoom,
                amount,
                cycles,
            } => {
                let c = c * f64::from(cycles);
                let (mix, offset) = match motion {
                    DepthMotion::Orbital => (0.75 + 0.50 * c.cos(), [0.50 * c.sin(), 0.0]),
                    DepthMotion::Circle => (0.60, [0.35 * c.cos(), 0.35 * c.sin()]),
                    DepthMotion::Dolly => (0.50 * (1.0 - c.cos()), [0.0, 0.0]),
                    DepthMotion::DollyOut => (0.50 * (1.0 + (PI * time.phase).cos()), [0.0, 0.0]),
                };
                let amount = f64::from(amount);
                CameraPose::Depth(DepthPose {
                    height: height.into(),
                    focus: focus.into(),
                    steady: steady.into(),
                    zoom: zoom.into(),
                    projection_mix: 1.0 + amount * (mix - 1.0),
                    offset: offset.map(|o| amount * o),
                    invert: self.project.depth.invert,
                })
            }
        }
    }

    fn resolve_view(&self, zoom: f64, x: f64, y: f64, roll_degrees: f64) -> ResolvedView {
        let zoom = zoom.max(self.min_camera_zoom);
        let roll = roll_degrees.to_radians();
        let (sin, cos) = (roll.sin().abs(), roll.cos().abs());
        let half = [
            (cos + sin / self.aspect) / (2.0 * zoom),
            (cos + self.aspect * sin) / (2.0 * zoom),
        ];
        let requested = [
            0.5 / zoom + x * (1.0 - 1.0 / zoom),
            0.5 / zoom + y * (1.0 - 1.0 / zoom),
        ];
        // max-then-min never panics, even if rounding makes `half` exceed 0.5.
        let center = std::array::from_fn(|i| requested[i].max(half[i]).min(1.0 - half[i]));
        ResolvedView { zoom, center, roll }
    }
}

impl MotionProject {
    /// Motion of frame `frame`. Validates the project and rebuilds the seeded
    /// state on every call; keep a [`MotionEvaluator`] to evaluate many frames.
    pub fn evaluate(&self, frame: u32) -> Result<EvaluatedMotion, ConfigError> {
        MotionEvaluator::new(self)?.evaluate(frame)
    }
}
