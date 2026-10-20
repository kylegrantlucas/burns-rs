//! GPU renderer behavior on a real adapter.
//!
//! Every test needs a Metal, Vulkan, or DirectX 12 adapter and is ignored by
//! default; run them with `cargo test -p burns --test render -- --ignored`. A
//! missing adapter fails the test instead of skipping it.
#![cfg(feature = "gpu")]

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::LazyLock;

use burns::motion::{CameraPose, depth_xy_to_uv, mirror, uv_to_depth_xy};
use burns::renderer::PREVIEW_MAX_EDGE;
use burns::{
    CameraKind, CameraMotion, CoverMap, DepthMap, GpuContext, MotionEvaluator, MotionProject,
    RenderError, RenderQuality, RenderView, Renderer, ShakeProfile, ShakeSettings, Size,
    SourceImage, View, WarpKind, WarpSettings, preset,
};
use pollster::block_on;

static CONTEXT: LazyLock<GpuContext> = LazyLock::new(|| {
    block_on(GpuContext::headless())
        .unwrap_or_else(|error| panic!("GPU tests need a real adapter: {error}"))
});

fn context() -> GpuContext {
    CONTEXT.clone()
}

fn renderer(size: Size, quality: RenderQuality) -> Renderer {
    Renderer::new(context(), size, quality).unwrap()
}

fn read(renderer: &mut Renderer) -> Vec<u8> {
    let size = renderer.output_size();
    let stride = size.width as usize * 4;
    let mut pixels = vec![0; stride * size.height as usize];
    block_on(renderer.read_rgba8(&mut pixels, stride)).unwrap();
    pixels
}

fn render(
    renderer: &mut Renderer,
    project: &MotionProject,
    frame: u32,
    view: RenderView,
) -> Vec<u8> {
    renderer.render(project, frame, view).unwrap();
    read(renderer)
}

/// Red encodes source u and green source v at texel centers; blue is constant.
fn uv_gradient(size: Size) -> SourceImage {
    let mut pixels = Vec::with_capacity(size.width as usize * size.height as usize * 4);
    for y in 0..size.height {
        for x in 0..size.width {
            let u = 255.0 * (f64::from(x) + 0.5) / f64::from(size.width);
            let v = 255.0 * (f64::from(y) + 0.5) / f64::from(size.height);
            pixels.extend([u.round() as u8, v.round() as u8, 96, 255]);
        }
    }
    SourceImage::from_rgba8(size, pixels).unwrap()
}

fn constant_depth(value: f32) -> DepthMap {
    let size = Size::new(32, 18);
    DepthMap::from_f32(size, vec![value; 32 * 18]).unwrap()
}

fn pixel_uv(x: u32, y: u32, size: Size) -> [f64; 2] {
    [
        (f64::from(x) + 0.5) / f64::from(size.width),
        (f64::from(y) + 0.5) / f64::from(size.height),
    ]
}

fn rgba(pixels: &[u8], size: Size, x: u32, y: u32) -> [u8; 4] {
    let i = (y as usize * size.width as usize + x as usize) * 4;
    pixels[i..i + 4].try_into().unwrap()
}

/// Gradient source probe: checks that output pixel `(x, y)` sampled the
/// virtual-frame point `virtual_uv`, after the cover crop and bilinear filtering.
struct Probe {
    source: Size,
    cover: [f64; 2],
    output: Size,
}

impl Probe {
    fn new(source: Size, project_output: Size, render_output: Size) -> Self {
        let cover = CoverMap::new(source, project_output).unwrap().scale();
        Self {
            source,
            cover: cover.map(f64::from),
            output: render_output,
        }
    }

    fn expected(&self, virtual_uv: [f64; 2]) -> [f64; 2] {
        let texels = [self.source.width, self.source.height];
        std::array::from_fn(|axis| {
            let source_uv = 0.5 + (virtual_uv[axis] - 0.5) * self.cover[axis];
            let half = 0.5 / f64::from(texels[axis]);
            255.0 * source_uv.clamp(half, 1.0 - half)
        })
    }

    fn check(
        &self,
        pixels: &[u8],
        x: u32,
        y: u32,
        virtual_uv: [f64; 2],
        tolerance: f64,
        label: &str,
    ) {
        let [r, g, ..] = rgba(pixels, self.output, x, y);
        let expected = self.expected(virtual_uv);
        assert!(
            (f64::from(r) - expected[0]).abs() <= tolerance
                && (f64::from(g) - expected[1]).abs() <= tolerance,
            "{label}: pixel ({x}, {y}) is ({r}, {g}), expected {expected:?} for virtual uv {virtual_uv:?}"
        );
    }
}

fn with_output(mut project: MotionProject, output: Size) -> MotionProject {
    project.output = output;
    project
}

#[test]
#[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
fn cover_crops_wide_and_portrait_sources_around_the_center() {
    for (source_size, output) in [
        (Size::new(400, 100), Size::new(160, 90)),
        (Size::new(100, 400), Size::new(90, 160)),
        (Size::new(300, 300), Size::new(160, 90)),
        (Size::new(240, 300), Size::new(90, 160)),
    ] {
        let label = format!("{source_size:?} into {output:?}");
        let mut project = with_output(MotionProject::default(), output);
        project.camera = CameraMotion::Still {
            view: View::centered(1.0),
        };
        let mut renderer = renderer(output, RenderQuality::Preview);
        renderer.set_source(&uv_gradient(source_size)).unwrap();
        let inspected = render(&mut renderer, &project, 0, RenderView::Source);
        let still = render(&mut renderer, &project, 0, RenderView::Motion);
        assert_eq!(
            inspected, still,
            "{label}: a static camera equals source inspection"
        );

        let probe = Probe::new(source_size, output, output);
        let (w, h) = (output.width - 1, output.height - 1);
        for (x, y) in [
            (0, 0),
            (w, 0),
            (0, h),
            (w, h),
            (w / 2, h / 2),
            (w / 3, h / 5),
        ] {
            probe.check(&inspected, x, y, pixel_uv(x, y, output), 1.5, &label);
        }
        // The crop is centered and keeps orientation: left/top corners carry the
        // smaller source coordinates.
        let [left, top, ..] = rgba(&inspected, output, 0, 0);
        let [right, bottom, ..] = rgba(&inspected, output, w, h);
        assert!(left < right && top < bottom, "{label}: flipped crop");
        assert!(
            (i32::from(left) + i32::from(right) - 255).abs() <= 2
                && (i32::from(top) + i32::from(bottom) - 255).abs() <= 2,
            "{label}: crop is not centered"
        );
    }
}

#[test]
#[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
fn planar_motion_matches_the_cpu_inverse_map() {
    let source = Size::new(640, 480);
    let project_output = Size::new(1920, 1080);
    let target = Size::new(192, 108);
    let mut renderer = renderer(target, RenderQuality::Preview);
    renderer.set_source(&uv_gradient(source)).unwrap();
    let probe = Probe::new(source, project_output, target);

    let mut rippled = preset("vibrate").unwrap();
    rippled.seed = 3;
    rippled.warp = WarpSettings::Ripple {
        enabled: true,
        amplitude: 0.01,
        wavelength: 0.1,
        center: [0.4, 0.6],
        cycles: 2,
    };
    let mut shimmering = preset("figure-eight").unwrap();
    shimmering.seed = 8;
    shimmering.shake.enabled = true;
    shimmering.warp = WarpSettings::default_for(WarpKind::HeatShimmer, true);
    let cases = [
        ("kenburns", preset("kenburns").unwrap(), 37),
        ("roll", preset("roll").unwrap(), 10),
        ("tilt", preset("tilt").unwrap(), 50),
        ("vibrate + ripple", rippled, 17),
        ("figure-eight + shake + shimmer", shimmering, 80),
    ];
    for (label, project, frame) in cases {
        let project = with_output(project, project_output);
        let pixels = render(&mut renderer, &project, frame, RenderView::Motion);
        let motion = project.evaluate(frame).unwrap();
        for y in (0..target.height).step_by(6) {
            for x in (0..target.width).step_by(6) {
                let mapped = motion.map_uv(pixel_uv(x, y, target), |_| unreachable!());
                probe.check(&pixels, x, y, mapped, 2.0, label);
            }
        }
    }
}

#[test]
#[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
fn bright_grid_shows_no_dark_wedges_under_roll_shake_and_warp() {
    // Same aspect as the output, so the virtual frame is exactly the source and
    // its one-pixel black border is the frame edge.
    let source_size = Size::new(640, 360);
    let mut pixels = Vec::with_capacity(640 * 360 * 4);
    for y in 0..source_size.height {
        for x in 0..source_size.width {
            let border =
                x == 0 || y == 0 || x == source_size.width - 1 || y == source_size.height - 1;
            let line = x % 16 < 2 || y % 16 < 2;
            pixels.extend(match (border, line) {
                (true, _) => [0, 0, 0, 255],
                (false, true) => [255, 200, 150, 255],
                (false, false) => [170, 210, 255, 255],
            });
        }
    }
    let source = SourceImage::from_rgba8(source_size, pixels).unwrap();

    let mut project = with_output(preset("roll").unwrap(), Size::new(1920, 1080));
    project.camera = CameraMotion::Transform {
        start: View::new(1.0, 0.2, 0.8, -12.0),
        end: View::new(1.0, 0.8, 0.2, 12.0),
        easing: burns::Easing::Linear,
    };
    project.seed = 99;
    project.shake = ShakeSettings {
        enabled: true,
        profile: ShakeProfile::Vibrate,
        amount: 2.0,
        speed: 1.0,
    };
    project.warp = WarpSettings::Ripple {
        enabled: true,
        amplitude: 0.02,
        wavelength: 0.05,
        center: [0.3, 0.7],
        cycles: 2,
    };

    let target = Size::new(320, 180);
    let mut renderer = renderer(target, RenderQuality::Preview);
    renderer.set_source(&source).unwrap();
    for frame in [0, 31, 60, 120, 181, 239] {
        let frame_pixels = render(&mut renderer, &project, frame, RenderView::Motion);
        for y in 2..target.height - 2 {
            for x in 2..target.width - 2 {
                let [r, g, b, a] = rgba(&frame_pixels, target, x, y);
                assert!(
                    r.min(g).min(b) >= 64 && a == 255,
                    "frame {frame}: dark pixel ({r}, {g}, {b}) at ({x}, {y}) shows the frame edge"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
fn frames_are_deterministic_across_order_repeats_and_fresh_renderers() {
    let mut project = with_output(preset("breathing").unwrap(), Size::new(1280, 720));
    project.seed = 11;
    project.shake = ShakeSettings {
        enabled: true,
        profile: ShakeProfile::Vibrate,
        amount: 1.0,
        speed: 1.0,
    };
    project.warp = WarpSettings::default_for(WarpKind::HeatShimmer, true);
    let source = uv_gradient(Size::new(300, 200));
    let target = Size::new(160, 90);
    let frames = [0, 7, 150, 239];

    let mut first = renderer(target, RenderQuality::Preview);
    first.set_source(&source).unwrap();
    let forward: Vec<_> = frames
        .iter()
        .map(|&n| render(&mut first, &project, n, RenderView::Motion))
        .collect();
    for (i, &n) in frames.iter().enumerate().rev() {
        assert_eq!(
            render(&mut first, &project, n, RenderView::Motion),
            forward[i]
        );
    }
    assert_ne!(forward[0], forward[1], "frames should move");

    let mut fresh = renderer(target, RenderQuality::Preview);
    fresh.set_source(&source).unwrap();
    for (i, &n) in frames.iter().enumerate().rev() {
        assert_eq!(
            render(&mut fresh, &project, n, RenderView::Motion),
            forward[i]
        );
    }
}

#[test]
#[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
fn preview_caps_the_long_edge_and_final_box_resolves_a_2x_render() {
    let mut capped = renderer(Size::new(1920, 1080), RenderQuality::Preview);
    assert_eq!(capped.output_size(), Size::new(PREVIEW_MAX_EDGE, 540));
    capped
        .resize(Size::new(1000, 3000), RenderQuality::Preview)
        .unwrap();
    assert_eq!(capped.output_size(), Size::new(320, 960));
    capped
        .resize(Size::new(64, 36), RenderQuality::Final)
        .unwrap();
    assert_eq!(capped.output_size(), Size::new(64, 36));

    // High-frequency source so the box filter is visible.
    let size = Size::new(257, 143);
    let mut pixels = Vec::with_capacity(257 * 143 * 4);
    for y in 0..size.height {
        for x in 0..size.width {
            let hash = (x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)) % 251;
            pixels.extend([
                hash as u8,
                ((x * 7 + y * 3) % 256) as u8,
                if (x + y) % 2 == 0 { 230 } else { 20 },
                255,
            ]);
        }
    }
    let source = SourceImage::from_rgba8(size, pixels).unwrap();
    let project = with_output(preset("kenburns").unwrap(), Size::new(64, 36));

    let mut double = renderer(Size::new(128, 72), RenderQuality::Preview);
    double.set_source(&source).unwrap();
    let reference = render(&mut double, &project, 3, RenderView::Motion);
    let mut fin = renderer(Size::new(64, 36), RenderQuality::Final);
    fin.set_source(&source).unwrap();
    let resolved = render(&mut fin, &project, 3, RenderView::Motion);

    let big = Size::new(128, 72);
    for y in 0..36 {
        for x in 0..64 {
            let quad = [(0, 0), (1, 0), (0, 1), (1, 1)]
                .map(|(dx, dy)| rgba(&reference, big, 2 * x + dx, 2 * y + dy));
            let actual = rgba(&resolved, Size::new(64, 36), x, y);
            for channel in 0..3 {
                let mean = quad.iter().map(|p| f64::from(p[channel])).sum::<f64>() / 4.0;
                assert!(
                    (f64::from(actual[channel]) - mean).abs() <= 1.0,
                    "pixel ({x}, {y}) channel {channel}: {} vs 2x2 mean {mean}",
                    actual[channel]
                );
            }
            assert_eq!(actual[3], 255);
        }
    }
}

fn depth_renderer(output: Size, target: Size) -> (Renderer, Probe) {
    let source = output;
    let mut renderer = renderer(target, RenderQuality::Preview);
    renderer.set_source(&uv_gradient(source)).unwrap();
    (renderer, Probe::new(source, output, target))
}

#[test]
#[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
fn depth_flat_plane_matches_the_analytic_intersection() {
    let output = Size::new(800, 450);
    let target = Size::new(160, 90);
    let (mut renderer, probe) = depth_renderer(output, target);
    let project = with_output(preset("orbital").unwrap(), output);
    let frame = project.timeline.frames / 4;
    let pose = match project.evaluate(frame).unwrap().camera {
        CameraPose::Depth(pose) => pose,
        other => panic!("expected depth camera, got {other:?}"),
    };
    let aspect = 800.0 / 450.0;
    for depth in [0.0f32, 0.4, 1.0] {
        renderer.set_depth(Some(&constant_depth(depth))).unwrap();
        let pixels = render(&mut renderer, &project, frame, RenderView::Motion);
        let z = 1.0 - pose.height * f64::from(depth);
        for y in (0..target.height).step_by(5) {
            for x in (0..target.width).step_by(5) {
                let (origin, far) = pose.ray(uv_to_depth_xy(pixel_uv(x, y, target), aspect));
                let hit = [
                    origin[0] + z * (far[0] - origin[0]),
                    origin[1] + z * (far[1] - origin[1]),
                ];
                let uv = depth_xy_to_uv(hit, aspect);
                probe.check(
                    &pixels,
                    x,
                    y,
                    uv.map(mirror),
                    2.0,
                    &format!("plane at {depth}"),
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
fn depth_parallax_depends_on_the_plane_and_the_steady_plane_is_anchored() {
    let output = Size::new(800, 450);
    let target = Size::new(160, 90);
    let (mut renderer, _) = depth_renderer(output, target);
    let mut project = with_output(preset("circle").unwrap(), output);
    let CameraMotion::Depth { height, steady, .. } = &mut project.camera else {
        unreachable!()
    };
    *height = 0.6;
    let steady = *steady;
    let quarter = project.timeline.frames / 4;

    renderer.set_depth(Some(&constant_depth(steady))).unwrap();
    let a = render(&mut renderer, &project, 0, RenderView::Motion);
    let b = render(&mut renderer, &project, quarter, RenderView::Motion);
    let worst = a.iter().zip(&b).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
    assert!(worst <= 1, "the steady plane moved by {worst} levels");

    let shift = |renderer: &mut Renderer, depth: f32| {
        renderer.set_depth(Some(&constant_depth(depth))).unwrap();
        let a = render(renderer, &project, 0, RenderView::Motion);
        let b = render(renderer, &project, quarter, RenderView::Motion);
        let [ar, ag, ..] = rgba(&a, target, 80, 45);
        let [br, bg, ..] = rgba(&b, target, 80, 45);
        [f64::from(br) - f64::from(ar), f64::from(bg) - f64::from(ag)]
    };
    let near = shift(&mut renderer, 1.0);
    let far = shift(&mut renderer, 0.0);
    assert!(
        near[0].hypot(near[1]) >= 4.0 && far[0].hypot(far[1]) >= 4.0,
        "near {near:?}, far {far:?}"
    );
    assert!(
        near[0] * far[0] + near[1] * far[1] < 0.0,
        "near {near:?} and far {far:?} should move oppositely"
    );
    assert!(near[0].hypot(near[1]) > far[0].hypot(far[1]));
}

#[test]
#[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
fn zero_height_depth_is_the_flat_zoomed_projection() {
    let output = Size::new(800, 450);
    let target = Size::new(160, 90);
    let (mut renderer, probe) = depth_renderer(output, target);
    let noise: Vec<f32> = (0..64 * 36)
        .map(|i| ((i * 37) % 101) as f32 / 100.0)
        .collect();
    renderer
        .set_depth(Some(&DepthMap::from_f32(Size::new(64, 36), noise).unwrap()))
        .unwrap();
    let mut project = with_output(preset("orbital").unwrap(), output);
    let CameraMotion::Depth { height, zoom, .. } = &mut project.camera else {
        unreachable!()
    };
    *height = 0.0;
    *zoom = 1.5;
    let pixels = render(&mut renderer, &project, 37, RenderView::Motion);
    for y in (0..target.height).step_by(4) {
        for x in (0..target.width).step_by(4) {
            let uv = pixel_uv(x, y, target).map(|c| mirror(0.5 + 1.5 * (c - 0.5)));
            probe.check(&pixels, x, y, uv, 2.0, "h = 0");
        }
    }
}

#[test]
#[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
fn depth_inspection_shows_the_effective_map_and_inversion_reproduces_raw() {
    let output = Size::new(800, 450);
    let target = Size::new(160, 90);
    let (mut renderer, _) = depth_renderer(output, target);
    let map_size = Size::new(64, 36);
    let raw: Vec<f32> = (0..36)
        .flat_map(|_| (0..64).map(|x| x as f32 / 63.0))
        .collect();
    let complement: Vec<f32> = raw.iter().map(|v| 1.0 - v).collect();
    renderer
        .set_depth(Some(&DepthMap::from_f32(map_size, raw).unwrap()))
        .unwrap();

    let mut project = with_output(preset("dolly").unwrap(), output);
    let plain = render(&mut renderer, &project, 0, RenderView::Depth);
    project.depth.invert = true;
    let inverted = render(&mut renderer, &project, 0, RenderView::Depth);
    for y in [0, 45, 89] {
        for x in [0, 10, 80, 159] {
            // Same aspect: the cover crop is the identity.
            let texel = (pixel_uv(x, y, target)[0] * 64.0 - 0.5).clamp(0.0, 63.0);
            let expected = 255.0 * texel / 63.0;
            let [p, pg, pb, _] = rgba(&plain, target, x, y);
            let [i, ..] = rgba(&inverted, target, x, y);
            assert!(
                (f64::from(p) - expected).abs() <= 1.0,
                "raw depth {p} vs {expected}"
            );
            assert_eq!((p, p), (pg, pb), "depth inspection is gray");
            assert!(
                (f64::from(i) - (255.0 - expected)).abs() <= 1.0,
                "inverted depth {i}"
            );
        }
    }

    let inverted_motion = render(&mut renderer, &project, 90, RenderView::Motion);
    project.depth.invert = false;
    renderer
        .set_depth(Some(&DepthMap::from_f32(map_size, complement).unwrap()))
        .unwrap();
    let complemented_motion = render(&mut renderer, &project, 90, RenderView::Motion);
    assert_eq!(inverted_motion, complemented_motion);
}

#[test]
#[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
fn missing_assets_and_bad_frames_report_actionable_errors() {
    let output = Size::new(160, 90);
    let mut renderer = renderer(output, RenderQuality::Preview);
    let depth_project = with_output(preset("orbital").unwrap(), output);
    assert!(matches!(
        renderer.render(&depth_project, 0, RenderView::Source),
        Err(RenderError::MissingSource)
    ));

    renderer
        .set_source(&uv_gradient(Size::new(64, 64)))
        .unwrap();
    let error = renderer
        .render(&depth_project, 0, RenderView::Motion)
        .unwrap_err();
    assert!(matches!(error, RenderError::MissingDepth));
    assert!(error.to_string().contains("RenderView::Source"));
    renderer
        .render(&depth_project, 0, RenderView::Source)
        .unwrap();
    assert!(matches!(
        renderer.render(&depth_project, 0, RenderView::Depth),
        Err(RenderError::MissingDepth)
    ));

    renderer.set_depth(Some(&constant_depth(0.5))).unwrap();
    renderer
        .render(&depth_project, 0, RenderView::Motion)
        .unwrap();
    renderer.set_depth(None).unwrap();
    assert!(matches!(
        renderer.render(&depth_project, 0, RenderView::Motion),
        Err(RenderError::MissingDepth)
    ));

    let planar = with_output(MotionProject::default(), output);
    let frames = planar.timeline.frames;
    match renderer.render(&planar, frames, RenderView::Motion) {
        Err(RenderError::Config(error)) => assert_eq!(error.field(), "frame"),
        other => panic!("expected a frame range error, got {other:?}"),
    }
}

#[test]
#[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
fn readback_validates_the_destination_and_preserves_padding() {
    let output = Size::new(33, 17);
    let mut renderer = renderer(output, RenderQuality::Preview);
    renderer
        .set_source(&uv_gradient(Size::new(50, 40)))
        .unwrap();
    let project = with_output(preset("kenburns").unwrap(), output);
    let tight = render(&mut renderer, &project, 5, RenderView::Motion);

    let row = 33 * 4;
    let stride = row + 12;
    let required = stride * 16 + row;
    let mut padded = vec![0xA5; required + 20];
    block_on(renderer.read_rgba8(&mut padded, stride)).unwrap();
    for y in 0..17 {
        let line = &padded[y * stride..y * stride + row];
        assert_eq!(line, &tight[y * row..(y + 1) * row], "row {y}");
        if y < 16 {
            assert!(
                padded[y * stride + row..(y + 1) * stride]
                    .iter()
                    .all(|&b| b == 0xA5)
            );
        }
    }
    assert!(
        padded[required..].iter().all(|&b| b == 0xA5),
        "bytes after the last row changed"
    );

    let mut short = vec![0xA5; required - 1];
    match block_on(renderer.read_rgba8(&mut short, stride)) {
        Err(RenderError::DestinationTooShort {
            len,
            required: needed,
            ..
        }) => {
            assert_eq!((len, needed), (required - 1, required));
        }
        other => panic!("expected DestinationTooShort, got {other:?}"),
    }
    assert!(short.iter().all(|&b| b == 0xA5));
    assert!(matches!(
        block_on(renderer.read_rgba8(&mut padded, row - 1)),
        Err(RenderError::DestinationStride { .. })
    ));

    let (pitch, len) = block_on(renderer.with_rgba8(|bytes, pitch| (pitch, bytes.len()))).unwrap();
    assert!(pitch >= row && pitch % 256 == 0);
    assert_eq!(len, pitch * 16 + row);

    // A panicking consumer still unmaps the buffer, so readback keeps working.
    let panicked = catch_unwind(AssertUnwindSafe(|| {
        block_on(renderer.with_rgba8(|_, _| panic!("consumer failed")))
    }));
    assert!(panicked.is_err());
    assert_eq!(read(&mut renderer), tight);
}

#[test]
#[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
fn oversized_targets_and_sources_report_texture_limits() {
    let max = context().device.limits().max_texture_dimension_2d;
    let too_wide = Size::new(max / 2 + 1, 16);
    match Renderer::new(context(), too_wide, RenderQuality::Final) {
        Err(RenderError::TextureLimit { max: limit, .. }) => assert_eq!(limit, max),
        other => panic!("expected TextureLimit, got {other:?}"),
    }
    // Preview caps the long edge, so the same request fits.
    let mut renderer = renderer(too_wide, RenderQuality::Preview);
    assert_eq!(renderer.output_size().width, PREVIEW_MAX_EDGE);

    renderer
        .resize(Size::new(64, 36), RenderQuality::Final)
        .unwrap();
    assert!(matches!(
        renderer.resize(too_wide, RenderQuality::Final),
        Err(RenderError::TextureLimit { .. })
    ));
    assert_eq!(renderer.output_size(), Size::new(64, 36));

    let wide_source =
        SourceImage::from_rgba8(Size::new(max + 1, 1), vec![255; (max as usize + 1) * 4]).unwrap();
    assert!(matches!(
        renderer.set_source(&wide_source),
        Err(RenderError::TextureLimit { .. })
    ));
    renderer
        .set_source(&uv_gradient(Size::new(64, 36)))
        .unwrap();
    let project = with_output(MotionProject::default(), Size::new(64, 36));
    renderer.render(&project, 0, RenderView::Motion).unwrap();
}

fn read_texture(context: &GpuContext, texture: &wgpu::Texture) -> Vec<u8> {
    let size = texture.size();
    let row = size.width * 4;
    let padded = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test readback"),
        size: u64::from(padded * size.height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = context.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(size.height),
            },
        },
        size,
    );
    context.queue.submit([encoder.finish()]);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |result| result.unwrap());
    context
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let mapped = buffer.slice(..).get_mapped_range().unwrap();
    mapped
        .chunks(padded as usize)
        .flat_map(|line| &line[..row as usize])
        .copied()
        .collect()
}

#[test]
#[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
fn present_blits_output_bytes_to_unorm_and_srgb_targets() {
    let output = Size::new(48, 27);
    let mut renderer = renderer(output, RenderQuality::Preview);
    renderer
        .set_source(&uv_gradient(Size::new(48, 27)))
        .unwrap();
    let project = with_output(MotionProject::default(), output);
    let expected = render(&mut renderer, &project, 0, RenderView::Source);
    let context = context();
    for format in [
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rgba8UnormSrgb,
    ] {
        let texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test present target"),
            size: wgpu::Extent3d {
                width: output.width,
                height: output.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        renderer
            .present_to(&texture.create_view(&Default::default()), format)
            .unwrap();
        let presented = read_texture(&context, &texture);
        let worst = presented
            .iter()
            .zip(&expected)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(worst <= 1, "{format:?} presentation differs by {worst}");
    }
}

#[test]
#[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
fn geometry_follows_the_project_aspect_not_the_target() {
    // The same project rendered into a square target is a resampled, not
    // re-cropped, version of the matching-aspect render.
    let mut project = with_output(MotionProject::default(), Size::new(1920, 1080));
    project.camera = CameraMotion::default_for(CameraKind::Still);
    let source = Size::new(400, 400);
    let mut square = renderer(Size::new(90, 90), RenderQuality::Preview);
    square.set_source(&uv_gradient(source)).unwrap();
    let pixels = render(&mut square, &project, 0, RenderView::Motion);
    let probe = Probe::new(source, project.output, Size::new(90, 90));
    let evaluator = MotionEvaluator::new(&project).unwrap();
    let motion = evaluator.evaluate(0).unwrap();
    for (x, y) in [(0, 0), (89, 89), (45, 10)] {
        let mapped = motion.map_uv(pixel_uv(x, y, Size::new(90, 90)), |_| unreachable!());
        probe.check(&pixels, x, y, mapped, 1.5, "square target");
    }
}

#[test]
#[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
fn shared_device_loss_is_reported_by_every_renderer() {
    let size = Size::new(32, 18);
    let other_context = block_on(GpuContext::headless()).unwrap();
    let mut other = Renderer::new(other_context, size, RenderQuality::Preview).unwrap();
    other.set_source(&uv_gradient(size)).unwrap();
    let project = with_output(MotionProject::default(), size);
    let unaffected = render(&mut other, &project, 0, RenderView::Source);
    let context = block_on(GpuContext::headless()).unwrap();
    let mut preview = Renderer::new(context.clone(), size, RenderQuality::Preview).unwrap();
    let mut export = Renderer::new(context.clone(), size, RenderQuality::Final).unwrap();
    // Destruction resolves device loss only after queued texture uploads finish.
    context.queue.submit([]);
    context
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();
    context.device.destroy();
    context.device.poll(wgpu::PollType::Poll).unwrap();
    let preview_result = preview.resize(size, RenderQuality::Preview);
    let export_result = export.resize(size, RenderQuality::Final);
    assert!(matches!(
        preview_result,
        Err(RenderError::DeviceLost { .. })
    ));
    assert!(matches!(export_result, Err(RenderError::DeviceLost { .. })));
    assert_eq!(
        render(&mut other, &project, 0, RenderView::Source),
        unaffected
    );
}

#[test]
#[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
fn preview_rejects_oversized_output_instead_of_silently_proxying() {
    let context = context();
    let max = context.device.limits().max_texture_dimension_2d;
    let output = Size::new(max + 1, 16);
    assert!(matches!(
        Renderer::new(context.clone(), output, RenderQuality::Preview),
        Err(RenderError::TextureLimit { .. })
    ));
    let mut renderer = Renderer::new(context, Size::new(64, 36), RenderQuality::Preview).unwrap();
    renderer
        .set_source(&uv_gradient(Size::new(64, 36)))
        .unwrap();
    let project = with_output(MotionProject::default(), output);
    assert!(matches!(
        renderer.render(&project, 0, RenderView::Motion),
        Err(RenderError::TextureLimit { .. })
    ));
}

#[test]
#[ignore = "requires a Metal, Vulkan, or DirectX 12 GPU adapter"]
fn depth_projection_covers_wide_and_portrait_sources_without_a_second_aspect_correction() {
    for (source, output) in [
        (Size::new(400, 100), Size::new(160, 90)),
        (Size::new(100, 400), Size::new(160, 90)),
        (Size::new(400, 100), Size::new(90, 160)),
        (Size::new(100, 400), Size::new(90, 160)),
    ] {
        let mut project = with_output(preset("orbital").unwrap(), output);
        if let CameraMotion::Depth { amount, zoom, .. } = &mut project.camera {
            *amount = 0.0;
            *zoom = 1.0;
        }
        let mut renderer = renderer(output, RenderQuality::Preview);
        renderer.set_source(&uv_gradient(source)).unwrap();
        renderer.set_depth(Some(&constant_depth(0.4))).unwrap();
        let pixels = render(&mut renderer, &project, 117, RenderView::Motion);
        let probe = Probe::new(source, output, output);
        let (w, h) = (output.width - 1, output.height - 1);
        for (x, y) in [(0, 0), (w, 0), (0, h), (w, h), (w / 2, h / 2)] {
            probe.check(&pixels, x, y, pixel_uv(x, y, output), 1.5, "depth cover");
        }
    }
}
