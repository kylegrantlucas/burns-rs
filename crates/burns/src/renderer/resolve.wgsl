// Final-quality 2x2 box resolve and presentation blits of the rendered output.
// Images are sRGB-encoded bytes in Rgba8Unorm textures: no transfer function is
// applied, except decoding before an sRGB render target re-encodes on store.

@group(0) @binding(0) var color_image: texture_2d<f32>;
@group(0) @binding(1) var color_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var output: VertexOutput;
    output.position = vec4<f32>(corner.x * 2.0 - 1.0, 1.0 - corner.y * 2.0, 0.0, 1.0);
    output.uv = corner;
    return output;
}

// Equal-weight average of the 2x2 supersampled texels covering this pixel.
@fragment
fn fs_resolve(input: VertexOutput) -> @location(0) vec4<f32> {
    let base = vec2<i32>(floor(input.position.xy)) * 2;
    let sum = textureLoad(color_image, base, 0)
        + textureLoad(color_image, base + vec2<i32>(1, 0), 0)
        + textureLoad(color_image, base + vec2<i32>(0, 1), 0)
        + textureLoad(color_image, base + vec2<i32>(1, 1), 0);
    return vec4<f32>(sum.rgb * 0.25, 1.0);
}

@fragment
fn fs_present(input: VertexOutput) -> @location(0) vec4<f32> {
    let rgb = textureSampleLevel(color_image, color_sampler, input.uv, 0.0).rgb;
    return vec4<f32>(rgb, 1.0);
}

fn srgb_to_linear(encoded: vec3<f32>) -> vec3<f32> {
    let low = encoded / 12.92;
    let high = pow((encoded + 0.055) / 1.055, vec3<f32>(2.4));
    return select(high, low, encoded <= vec3<f32>(0.04045));
}

// For *Srgb targets: decode so the hardware encode stores the original bytes.
@fragment
fn fs_present_srgb(input: VertexOutput) -> @location(0) vec4<f32> {
    let rgb = textureSampleLevel(color_image, color_sampler, input.uv, 0.0).rgb;
    return vec4<f32>(srgb_to_linear(rgb), 1.0);
}
