use std::f64::consts::{PI, TAU};

use burns::motion::{
    self, CameraPose, DepthPose, EvaluatedMotion, MotionEvaluator, WarpField, depth_xy_to_uv,
    mirror, seed_phase, splitmix64, uv_to_depth_xy,
};
use burns::{
    CameraMotion, DepthMotion, Easing, FrameRate, MotionProject, ShakeProfile, ShakeSettings, Size,
    Timeline, View, WarpSettings, preset,
};

const EPS: f64 = 1e-9;

fn assert_uv(actual: [f64; 2], expected: [f64; 2], eps: f64) {
    assert!(
        (actual[0] - expected[0]).abs() <= eps && (actual[1] - expected[1]).abs() <= eps,
        "expected {expected:?}, got {actual:?} (eps {eps})"
    );
}

fn planar_uv(motion: &EvaluatedMotion, uv: [f64; 2]) -> [f64; 2] {
    motion.map_uv(uv, |_| unreachable!("planar cameras never sample depth"))
}

fn view(motion: &EvaluatedMotion) -> motion::ResolvedView {
    match motion.camera {
        CameraPose::View(view) => view,
        other => panic!("expected a view camera, got {other:?}"),
    }
}

fn depth_pose(motion: &EvaluatedMotion) -> DepthPose {
    match motion.camera {
        CameraPose::Depth(pose) => pose,
        other => panic!("expected a depth camera, got {other:?}"),
    }
}

fn still(zoom: f32, x: f32, y: f32, roll_degrees: f32) -> MotionProject {
    MotionProject {
        camera: CameraMotion::Still {
            view: View::new(zoom, x, y, roll_degrees),
        },
        ..MotionProject::default()
    }
}

/// Output-frame sample points: a dense perimeter plus an interior grid.
fn probe_points() -> Vec<[f64; 2]> {
    let steps = 48;
    let mut points = Vec::new();
    for i in 0..=steps {
        let t = f64::from(i) / f64::from(steps);
        points.extend([[t, 0.0], [t, 1.0], [0.0, t], [1.0, t]]);
        for j in 0..=8 {
            points.push([t, f64::from(j) / 8.0]);
        }
    }
    points
}

fn assert_inside_frame(motion: &EvaluatedMotion, label: &str) {
    for uv in probe_points() {
        let mapped = planar_uv(motion, uv);
        for value in mapped {
            assert!(
                (-EPS..=1.0 + EPS).contains(&value),
                "{label} frame {}: output {uv:?} samples {mapped:?} outside the virtual frame",
                motion.time.index
            );
        }
    }
}

#[test]
fn frame_timing_is_rational_and_single_frame_progress_is_zero() {
    let mut project = MotionProject {
        timeline: Timeline::new(FrameRate::new(30000, 1001), 300).unwrap(),
        ..MotionProject::default()
    };
    let motion = project.evaluate(7).unwrap();
    assert_eq!(motion.time.seconds, 7.0 * 1001.0 / 30000.0);
    assert_eq!(motion.time.progress, 7.0 / 299.0);
    assert_eq!(motion.time.phase, 7.0 / 300.0);
    assert_eq!(project.evaluate(300).unwrap_err().field(), "frame");

    project.timeline = Timeline::new(FrameRate::new(24, 1), 1).unwrap();
    let only = project.evaluate(0).unwrap();
    assert_eq!((only.time.progress, only.time.phase), (0.0, 0.0));
    // N = 1 shows the start view.
    let CameraMotion::Transform { start, .. } = project.camera else {
        unreachable!()
    };
    assert_eq!(view(&only).zoom, f64::from(start.zoom));
}

#[test]
fn evaluation_is_stateless_in_any_frame_order() {
    let mut project = preset("vibrate").unwrap();
    project.seed = 7;
    project.warp = WarpSettings::default_for(burns::WarpKind::HeatShimmer, true);
    project.camera = CameraMotion::default_for(burns::CameraKind::FigureEight);
    let evaluator = MotionEvaluator::new(&project).unwrap();
    let frames = project.timeline.frames;
    let forward: Vec<_> = (0..frames)
        .map(|n| evaluator.evaluate(n).unwrap())
        .collect();
    for n in (0..frames).rev().step_by(7) {
        assert_eq!(evaluator.evaluate(n).unwrap(), forward[n as usize]);
    }
    for n in [131, 3, 239, 0, 64] {
        assert_eq!(project.evaluate(n).unwrap(), forward[n as usize]);
        assert_eq!(evaluator.evaluate(n).unwrap(), forward[n as usize]);
    }
}

#[test]
fn transform_interpolates_eased_views_between_exact_endpoints() {
    let mut project = MotionProject::default();
    let CameraMotion::Transform { start, end, .. } = project.camera else {
        unreachable!()
    };
    let center = |v: View| {
        let z = f64::from(v.zoom);
        [
            0.5 / z + f64::from(v.x) * (1.0 - 1.0 / z),
            0.5 / z + f64::from(v.y) * (1.0 - 1.0 / z),
        ]
    };
    let last = project.timeline.frames - 1;
    let first = view(&project.evaluate(0).unwrap());
    let final_view = view(&project.evaluate(last).unwrap());
    assert_eq!(first.zoom, f64::from(start.zoom));
    assert_uv(first.center, center(start), 1e-15);
    assert_eq!(final_view.zoom, f64::from(end.zoom));
    assert_uv(final_view.center, center(end), 1e-12);

    // Five frames put frame 1 at u = 0.25.
    project.timeline = Timeline::new(FrameRate::new(30, 1), 5).unwrap();
    let mut quarter = |easing| {
        let CameraMotion::Transform { easing: e, .. } = &mut project.camera else {
            unreachable!()
        };
        *e = easing;
        view(&project.evaluate(1).unwrap()).zoom
    };
    let lerp = |p: f64| f64::from(start.zoom) + p * f64::from(end.zoom - start.zoom);
    assert!((quarter(Easing::Linear) - lerp(0.25)).abs() < 1e-7);
    assert!((quarter(Easing::Cosine) - lerp((1.0 - (PI / 4.0).cos()) / 2.0)).abs() < 1e-7);
}

#[test]
fn view_position_spans_the_full_crop_travel() {
    let corners = |project: &MotionProject| {
        let motion = project.evaluate(0).unwrap();
        (
            planar_uv(&motion, [0.0, 0.0]),
            planar_uv(&motion, [1.0, 1.0]),
        )
    };
    let (top_left, bottom_right) = corners(&still(2.0, 0.0, 0.0, 0.0));
    assert_uv(top_left, [0.0, 0.0], EPS);
    assert_uv(bottom_right, [0.5, 0.5], EPS);
    let (top_left, bottom_right) = corners(&still(2.0, 1.0, 1.0, 0.0));
    assert_uv(top_left, [0.5, 0.5], EPS);
    assert_uv(bottom_right, [1.0, 1.0], EPS);
    let (top_left, bottom_right) = corners(&still(4.0, 0.5, 0.0, 0.0));
    assert_uv(top_left, [0.375, 0.0], EPS);
    assert_uv(bottom_right, [0.625, 0.25], EPS);
}

#[test]
fn camera_roll_raises_zoom_to_cover_and_turns_clockwise() {
    let project = still(1.0, 0.0, 1.0, 10.0);
    let aspect = 1920.0 / 1080.0;
    let sin = 10f64.to_radians().sin();
    let motion = project.evaluate(0).unwrap();
    let resolved = view(&motion);
    assert!((resolved.zoom - (1.0 + aspect * sin)).abs() < 1e-12);
    assert_inside_frame(&motion, "still roll");
    // The rolled crop touches the frame: minimum zoom is tight.
    let touches = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        .into_iter()
        .flat_map(|uv| planar_uv(&motion, uv))
        .any(|value| value.abs() < 1e-9 || (value - 1.0).abs() < 1e-9);
    assert!(touches, "minimum roll zoom should be tight");
    // Clockwise: the left edge shows content from below the center.
    let centered = still(1.0, 0.5, 0.5, 10.0).evaluate(0).unwrap();
    assert!(planar_uv(&centered, [0.0, 0.5])[1] > 0.5);
    assert!(planar_uv(&centered, [1.0, 0.5])[1] < 0.5);

    for portrait in [false, true] {
        let mut project = preset("roll").unwrap();
        if portrait {
            project.output = Size::new(1080, 1920);
        }
        let evaluator = MotionEvaluator::new(&project).unwrap();
        for frame in (0..project.timeline.frames).step_by(13) {
            assert_inside_frame(&evaluator.evaluate(frame).unwrap(), "roll preset");
        }
    }
}

#[test]
fn tilt_maps_output_corners_exactly_onto_a_projective_quad() {
    let mut project = preset("tilt").unwrap();
    project.camera = CameraMotion::Tilt {
        amount: 1.0,
        easing: Easing::Linear,
    };
    project.timeline = Timeline::new(FrameRate::new(30, 1), 5).unwrap();
    let cases = [
        (0, [[0.06, 0.13], [0.94, 0.01], [0.08, 0.87], [0.96, 0.99]]),
        (4, [[0.09, 0.04], [0.91, 0.16], [0.07, 0.96], [0.89, 0.84]]),
    ];
    let unit = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    for (frame, quad) in cases {
        let motion = project.evaluate(frame).unwrap();
        for (corner, expected) in unit.into_iter().zip(quad) {
            assert_uv(planar_uv(&motion, corner), expected, 1e-12);
        }
        // A homography sends the square's center to the quad's diagonal
        // intersection, unlike a bilinear map (corner average).
        let [tl, tr, bl, br] = quad;
        let (d1, d2) = (
            [br[0] - tl[0], br[1] - tl[1]],
            [bl[0] - tr[0], bl[1] - tr[1]],
        );
        let t =
            ((tr[0] - tl[0]) * d2[1] - (tr[1] - tl[1]) * d2[0]) / (d1[0] * d2[1] - d1[1] * d2[0]);
        let crossing = [tl[0] + t * d1[0], tl[1] + t * d1[1]];
        assert_uv(planar_uv(&motion, [0.5, 0.5]), crossing, 1e-12);
    }

    project.camera = CameraMotion::Tilt {
        amount: 0.5,
        easing: Easing::Linear,
    };
    let half = project.evaluate(0).unwrap();
    assert_uv(planar_uv(&half, [0.0, 0.0]), [0.03, 0.065], 1e-12);
    assert_uv(planar_uv(&half, [1.0, 1.0]), [0.98, 0.995], 1e-12);
}

#[test]
fn breathing_and_figure_eight_follow_their_periodic_curves() {
    let mut project = preset("breathing").unwrap();
    let CameraMotion::Breathing {
        base_zoom,
        amplitude,
        ..
    } = project.camera
    else {
        unreachable!()
    };
    let half = project.timeline.frames / 2;
    let peak = view(&project.evaluate(half).unwrap());
    assert!((peak.zoom - f64::from(base_zoom) - f64::from(amplitude)).abs() < 1e-12);
    assert_uv(peak.center, [0.5, 0.5], 1e-12);
    assert!((view(&project.evaluate(0).unwrap()).zoom - f64::from(base_zoom)).abs() < 1e-12);

    project = preset("figure-eight").unwrap();
    let CameraMotion::FigureEight {
        zoom, amplitude, ..
    } = project.camera
    else {
        unreachable!()
    };
    let quarter = view(&project.evaluate(project.timeline.frames / 4).unwrap());
    let z = f64::from(zoom);
    let x = 0.5 + f64::from(amplitude[0]);
    // sin(2 * pi / 2) = 0, so y stays centered.
    assert_eq!(quarter.zoom, z);
    assert_uv(quarter.center, [0.5 / z + x * (1.0 - 1.0 / z), 0.5], 1e-12);
}

#[test]
fn seed_phases_are_deterministic_and_zero_for_seed_zero() {
    assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF);
    for (component, harmonic) in [(1, 0), (1, 75), (2, 0), (3, 2)] {
        assert_eq!(seed_phase(0, component, harmonic), 0.0);
    }
    let phase = seed_phase(42, 1, 3);
    assert_ne!(phase, 0.0);
    assert!(phase.abs() < TAU);
    assert_eq!(phase, seed_phase(42, 1, 3));
    assert_ne!(seed_phase(42, 2, 0), seed_phase(43, 2, 0));

    let mut project = preset("shake").unwrap();
    let base = project.evaluate(40).unwrap().shake;
    project.seed = 9;
    let seeded = project.evaluate(40).unwrap().shake;
    assert_ne!(base.offset, seeded.offset);
    assert_eq!(seeded, project.evaluate(40).unwrap().shake);
}

#[test]
fn safety_scale_keeps_profile_defaults_and_grows_only_when_needed() {
    let scale = |project: &MotionProject| MotionEvaluator::new(project).unwrap().safety_scale();
    assert_eq!(scale(&MotionProject::default()), 1.0);
    assert_eq!(scale(&preset("shake").unwrap()), 1.08);
    assert_eq!(scale(&preset("vibrate").unwrap()), 1.14);
    assert_eq!(scale(&preset("vibrate-tight").unwrap()), 1.03);

    // A warp alone still gets a constant margin before the camera.
    let rippled = preset("ripple").unwrap();
    assert!((scale(&rippled) - (1.0 + 2.0 * 0.006)).abs() < 1e-6);

    let mut strong = preset("vibrate").unwrap();
    strong.shake.amount = 2.0;
    let k = scale(&strong);
    assert!(
        k > 1.14,
        "doubled vibrate needs more than its nominal scale, got {k}"
    );
    // Constant over time: no zoom pumping.
    let evaluator = MotionEvaluator::new(&strong).unwrap();
    for frame in [0, 50, 199] {
        assert_eq!(evaluator.evaluate(frame).unwrap().shake.scale, k);
    }
}

#[test]
fn shake_and_warp_never_sample_outside_the_frame() {
    for (profile, warp, output) in [
        (
            ShakeProfile::Vibrate,
            burns::WarpKind::Ripple,
            Size::new(1920, 1080),
        ),
        (
            ShakeProfile::Shake,
            burns::WarpKind::HeatShimmer,
            Size::new(1080, 1920),
        ),
        (
            ShakeProfile::VibrateTight,
            burns::WarpKind::Ripple,
            Size::new(1000, 1000),
        ),
    ] {
        let mut project = preset("roll").unwrap();
        project.output = output;
        project.seed = 1234;
        project.shake = ShakeSettings {
            enabled: true,
            profile,
            amount: 2.0,
            speed: 1.0,
        };
        project.warp = match WarpSettings::default_for(warp, true) {
            WarpSettings::Ripple { center, cycles, .. } => WarpSettings::Ripple {
                enabled: true,
                amplitude: 0.02,
                wavelength: 0.05,
                center,
                cycles,
            },
            WarpSettings::HeatShimmer { cycles, .. } => WarpSettings::HeatShimmer {
                enabled: true,
                amplitude: 0.02,
                spatial_scale: 4.0,
                cycles,
            },
        };
        let evaluator = MotionEvaluator::new(&project).unwrap();
        for frame in (0..project.timeline.frames).step_by(5) {
            assert_inside_frame(&evaluator.evaluate(frame).unwrap(), &format!("{profile:?}"));
        }
    }
}

#[test]
fn uncoverable_shake_displacement_is_rejected() {
    let mut project = preset("vibrate").unwrap();
    project.shake.amount = 2.0;
    project.output = Size::new(1, 2000);
    let error = MotionEvaluator::new(&project).unwrap_err();
    assert_eq!(error.field(), "shake.amount");
    assert!(error.to_string().contains("unsupported displacement"));
}

#[test]
fn ripple_is_zero_at_its_center_and_bounded_by_its_amplitude() {
    let mut project = preset("ripple").unwrap();
    project.seed = 5;
    let motion = project.evaluate(17).unwrap();
    let WarpField::Ripple {
        center, amplitude, ..
    } = motion.warp
    else {
        panic!("ripple should be active")
    };
    assert_eq!(motion.warp.offset(center, motion.aspect), [0.0, 0.0]);
    for uv in probe_points() {
        let [dx, dy] = motion.warp.offset(uv, motion.aspect);
        assert!((dx * motion.aspect).hypot(dy) <= amplitude + 1e-12);
    }
    project.warp = WarpSettings::default_for(burns::WarpKind::Ripple, false);
    assert_eq!(project.evaluate(17).unwrap().warp, WarpField::None);
}

fn depth_project(motion: DepthMotion) -> MotionProject {
    let mut project = preset(match motion {
        DepthMotion::Orbital => "orbital",
        DepthMotion::Circle => "circle",
        DepthMotion::Dolly => "dolly",
        DepthMotion::DollyOut => "dolly-out",
    })
    .unwrap();
    project.output = Size::new(1600, 900);
    project
}

/// Hit of a flat plane at `depth`: `Z = 1 - h d`, `XY = O + Z (E - O)`.
fn flat_plane_uv(pose: &DepthPose, aspect: f64, uv: [f64; 2], depth: f64) -> [f64; 2] {
    let (origin, far) = pose.ray(uv_to_depth_xy(uv, aspect));
    let z = 1.0 - pose.height * depth;
    let hit = [
        origin[0] + z * (far[0] - origin[0]),
        origin[1] + z * (far[1] - origin[1]),
    ];
    let virtual_uv = depth_xy_to_uv(hit, aspect);
    [mirror(virtual_uv[0]), mirror(virtual_uv[1])]
}

#[test]
fn depth_flat_planes_hit_their_analytic_intersections() {
    let project = depth_project(DepthMotion::Orbital);
    let motion = project.evaluate(project.timeline.frames / 4).unwrap();
    let pose = depth_pose(&motion);
    // c = pi / 2: i = 0.75, P = (0.5, 0).
    assert!((pose.projection_mix - 0.75).abs() < 1e-12);
    assert_uv(pose.offset, [0.5, 0.0], 1e-12);
    for depth in [0.0, 0.4, 1.0] {
        for uv in [[0.1, 0.2], [0.5, 0.5], [0.93, 0.71], [0.0, 1.0]] {
            let traced = motion.map_uv(uv, |_| depth);
            assert_uv(traced, flat_plane_uv(&pose, motion.aspect, uv, depth), 1e-4);
        }
    }
}

#[test]
fn depth_parallax_depends_on_depth_and_the_steady_plane_is_anchored() {
    let project = depth_project(DepthMotion::Circle);
    let CameraMotion::Depth { steady, .. } = project.camera else {
        unreachable!()
    };
    let evaluator = MotionEvaluator::new(&project).unwrap();
    let frames = [0, 60, 120, 180].map(|n| evaluator.evaluate(n).unwrap());
    let uv = [0.37, 0.61];
    let anchored: Vec<_> = frames
        .iter()
        .map(|m| m.map_uv(uv, |_| f64::from(steady)))
        .collect();
    for other in &anchored[1..] {
        assert_uv(*other, anchored[0], 1e-4);
    }
    let shift = |depth: f64| {
        let a = frames[0].map_uv(uv, |_| depth);
        let b = frames[1].map_uv(uv, |_| depth);
        [b[0] - a[0], b[1] - a[1]]
    };
    let (near, far) = (shift(1.0), shift(0.0));
    assert!(near[0].hypot(near[1]) > 0.01 && far[0].hypot(far[1]) > 0.01);
    // Planes in front of and behind the steady plane move in opposite directions.
    assert!(near[0] * far[0] + near[1] * far[1] < 0.0);
}

#[test]
fn zero_height_is_the_flat_zoomed_projection() {
    let mut project = depth_project(DepthMotion::Orbital);
    let CameraMotion::Depth { height, zoom, .. } = &mut project.camera else {
        unreachable!()
    };
    *height = 0.0;
    *zoom = 1.5;
    for frame in [0, 37, 120] {
        let motion = project.evaluate(frame).unwrap();
        for uv in [[0.0, 0.0], [0.25, 0.8], [0.5, 0.5], [1.0, 0.3]] {
            let expected = [
                mirror(0.5 + 1.5 * (uv[0] - 0.5)),
                mirror(0.5 + 1.5 * (uv[1] - 0.5)),
            ];
            let mapped = motion.map_uv(uv, |_| unreachable!("h = 0 never samples depth"));
            assert_uv(mapped, expected, 1e-12);
        }
    }
}

#[test]
fn inverted_depth_equals_the_complemented_raw_map() {
    let mut project = depth_project(DepthMotion::Dolly);
    let raw = |uv: [f64; 2]| (0.3 + 0.6 * uv[0] * uv[1]).clamp(0.0, 1.0);
    let frame = 90;
    project.depth.invert = true;
    let inverted = project.evaluate(frame).unwrap();
    project.depth.invert = false;
    let complemented = project.evaluate(frame).unwrap();
    for uv in [[0.2, 0.3], [0.5, 0.5], [0.8, 0.1]] {
        assert_eq!(
            inverted.map_uv(uv, raw),
            complemented.map_uv(uv, |p| 1.0 - raw(p))
        );
    }
}

#[test]
fn depth_paths_blend_by_amount_and_dolly_out_ignores_cycles() {
    let mut project = depth_project(DepthMotion::DollyOut);
    let pose = |project: &MotionProject, frame| depth_pose(&project.evaluate(frame).unwrap());
    assert_eq!(pose(&project, 0).projection_mix, 1.0);
    let half = project.timeline.frames / 2;
    // Half a cosine over the loop phase: i = 0.5 (1 + cos(pi / 2)) at tau = 0.5.
    assert!((pose(&project, half).projection_mix - 0.5).abs() < 1e-12);
    let before = pose(&project, 77);
    let set = |project: &mut MotionProject, new_cycles: u32, new_amount: f32| {
        let CameraMotion::Depth { cycles, amount, .. } = &mut project.camera else {
            unreachable!()
        };
        (*cycles, *amount) = (new_cycles, new_amount);
    };
    set(&mut project, 5, 1.0);
    assert_eq!(pose(&project, 77), before);
    set(&mut project, 5, 0.0);
    for frame in [0, 77, 200] {
        let still = pose(&project, frame);
        assert_eq!((still.projection_mix, still.offset), (1.0, [0.0, 0.0]));
    }
}
