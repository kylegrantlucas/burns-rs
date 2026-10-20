// Single inverse map from an output pixel to one source sample:
// warp -> shake/camera (one projective matrix) -> optional depth traversal ->
// mirror (depth only) -> cover -> source. All time-dependent terms are evaluated
// on the CPU once per frame; this shader evaluates only spatial terms.

const TAU: f32 = 6.283185307179586;

const MODE_PLANAR: u32 = 0u;
const MODE_DEPTH: u32 = 1u;
const MODE_SOURCE: u32 = 2u;
const MODE_DEPTH_VIEW: u32 = 3u;

const WARP_RIPPLE: u32 = 1u;
const WARP_SHIMMER: u32 = 2u;

const COARSE_STEPS: i32 = 64;
const REFINE_STEPS: i32 = 8;

struct Params {
    // Rows of the 3x3 inverse map applied to [u, v, 1] after the warp.
    frame_map: array<vec4<f32>, 3>,
    // Cover scale x, cover scale y, output aspect A, unused.
    cover: vec4<f32>,
    // Ripple: center x, center y, amplitude, 1 / wavelength.
    // Shimmer: amplitude, 2 pi 6k, 2 pi 11k, 2 pi 4k.
    warp_a: vec4<f32>,
    // Ripple: wavelength / 4, temporal phase, unused, unused.
    // Shimmer: phase 6k, phase 11k, phase 4k, unused.
    warp_b: vec4<f32>,
    // Height, far-plane scale, origin scale, offset far-plane factor.
    depth_a: vec4<f32>,
    // Offset P x, offset P y, unused, unused.
    depth_b: vec4<f32>,
    // Mode, warp kind, invert depth, unused.
    modes: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var source_image: texture_2d<f32>;
@group(0) @binding(2) var source_sampler: sampler;
@group(0) @binding(3) var depth_map: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle; uv (0, 0) is the top-left corner.
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var output: VertexOutput;
    output.position = vec4<f32>(corner.x * 2.0 - 1.0, 1.0 - corner.y * 2.0, 0.0, 1.0);
    output.uv = corner;
    return output;
}

fn cover(uv: vec2<f32>) -> vec2<f32> {
    return 0.5 + (uv - 0.5) * params.cover.xy;
}

fn mirror(uv: vec2<f32>) -> vec2<f32> {
    return 1.0 - abs(1.0 - (uv - 2.0 * floor(uv * 0.5)));
}

fn source_color(virtual_uv: vec2<f32>) -> vec4<f32> {
    let rgb = textureSampleLevel(source_image, source_sampler, cover(virtual_uv), 0.0).rgb;
    return vec4<f32>(rgb, 1.0);
}

fn depth_texel(texel: vec2<i32>) -> f32 {
    let raw = textureLoad(depth_map, texel, 0).r;
    return select(raw, 1.0 - raw, params.modes.z != 0u);
}

// Effective depth at a virtual-frame UV: cover once, then bilinear from four
// clamped texel loads (R32Float is not filterable).
fn effective_depth(virtual_uv: vec2<f32>) -> f32 {
    let size = vec2<i32>(textureDimensions(depth_map));
    let position = cover(virtual_uv) * vec2<f32>(size) - 0.5;
    let base = floor(position);
    let weight = position - base;
    let low = clamp(vec2<i32>(base), vec2<i32>(0), size - 1);
    let high = clamp(vec2<i32>(base) + 1, vec2<i32>(0), size - 1);
    let top = mix(depth_texel(low), depth_texel(vec2<i32>(high.x, low.y)), weight.x);
    let bottom = mix(depth_texel(vec2<i32>(low.x, high.y)), depth_texel(high), weight.x);
    return mix(top, bottom, weight.y);
}

fn warp_offset(uv: vec2<f32>) -> vec2<f32> {
    let aspect = params.cover.z;
    let kind = params.modes.y;
    if kind == WARP_RIPPLE {
        let radial = vec2<f32>((uv.x - params.warp_a.x) * aspect, uv.y - params.warp_a.y);
        let radius = length(radial);
        if radius <= 0.0 {
            return vec2<f32>(0.0);
        }
        let wave = params.warp_a.z * smoothstep(0.0, params.warp_b.x, radius)
            * sin(TAU * radius * params.warp_a.w - params.warp_b.y);
        let metric = radial * (wave / radius);
        return vec2<f32>(metric.x / aspect, metric.y);
    }
    if kind == WARP_SHIMMER {
        let amplitude = params.warp_a.x;
        let envelope = 0.25 + 0.75 * uv.y * uv.y;
        let dy = amplitude * envelope * (0.65 * sin(params.warp_a.y * uv.x + params.warp_b.x)
            + 0.35 * sin(params.warp_a.z * uv.x + params.warp_b.y));
        let dx = 0.35 * amplitude * envelope * sin(params.warp_a.w * uv.y + params.warp_b.z);
        return vec2<f32>(dx / aspect, dy);
    }
    return vec2<f32>(0.0);
}

fn depth_xy_to_uv(xy: vec2<f32>) -> vec2<f32> {
    return vec2<f32>((xy.x / params.cover.z + 1.0) * 0.5, (1.0 - xy.y) * 0.5);
}

fn surface_reached(xy: vec2<f32>, z: f32) -> bool {
    return z >= 1.0 - params.depth_a.x * effective_depth(mirror(depth_xy_to_uv(xy)));
}

// Heightfield traversal: first crossing of the ray with Z = 1 - h * depth while
// scanning Z from 1 - h (accepted at the endpoint) to the far plane Z = 1, which
// always crosses. Returns the hit in virtual-frame UV, before mirroring.
fn trace_depth(uv: vec2<f32>) -> vec2<f32> {
    let aspect = params.cover.z;
    let g = vec2<f32>((2.0 * uv.x - 1.0) * aspect, 1.0 - 2.0 * uv.y);
    let height = params.depth_a.x;
    let offset = params.depth_b.xy;
    let far = params.depth_a.y * g + offset * params.depth_a.w;
    if height <= 0.0 {
        return depth_xy_to_uv(far);
    }
    let origin = offset + params.depth_a.z * g;
    let direction = far - origin;
    let start = 1.0 - height;
    if surface_reached(origin + start * direction, start) {
        return depth_xy_to_uv(origin + start * direction);
    }
    var near = start;
    var hit = 1.0;
    for (var step = 1; step <= COARSE_STEPS; step += 1) {
        let z = select(start + height * f32(step) / f32(COARSE_STEPS), 1.0, step == COARSE_STEPS);
        if surface_reached(origin + z * direction, z) {
            hit = z;
            break;
        }
        near = z;
    }
    for (var refine = 0; refine < REFINE_STEPS; refine += 1) {
        let mid = 0.5 * (near + hit);
        if surface_reached(origin + mid * direction, mid) {
            hit = mid;
        } else {
            near = mid;
        }
    }
    return depth_xy_to_uv(origin + hit * direction);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let mode = params.modes.x;
    if mode == MODE_SOURCE {
        return source_color(input.uv);
    }
    if mode == MODE_DEPTH_VIEW {
        let depth = effective_depth(input.uv);
        return vec4<f32>(depth, depth, depth, 1.0);
    }
    let warped = vec3<f32>(input.uv + warp_offset(input.uv), 1.0);
    let mapped = vec3<f32>(
        dot(params.frame_map[0].xyz, warped),
        dot(params.frame_map[1].xyz, warped),
        dot(params.frame_map[2].xyz, warped),
    );
    let framed = mapped.xy / mapped.z;
    if mode == MODE_DEPTH {
        return source_color(mirror(trace_depth(framed)));
    }
    return source_color(framed);
}
