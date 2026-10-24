# Technical reference

[Back to burns](../README.md)

Deterministic single-image motion with portable JSON projects and a shared Rust processing core.

## Development

```sh
nix develop
just presets
cargo test -p burns --no-default-features
```

The locked project shell supplies Rust 1.97.0, the `wasm32-unknown-unknown` target,
wasm-bindgen 0.2.129, Node 24, pnpm, FFmpeg, and ONNX Runtime. It does not install
machine-wide tools or require credentials. Keep `Cargo.lock`, `flake.lock`, and the
site's pnpm lockfile with the project.

`just presets` regenerates `presets/*.json` from the Rust constructors; those files
are not a second source of defaults. The settings registry supplies the same
stable IDs, ranges, labels, choices, and variant visibility to every interface.

## Headless CLI

Run commands inside `nix develop`. `just cli` preserves individual arguments,
including image paths containing spaces.

```sh
just cli presets
just cli presets --json
just cli schema
just cli preset kenburns > project.json
just cli render -i "still image.png" -o clip.mp4 --preset kenburns
just cli render -i image.png -o depth-motion.mp4 --preset dolly-out --duration 20 --depth depth.png
just cli frame -i image.png -o frame.png --config project.json --depth depth.png --frame 37
just cli depth -i image.png -o depth.png
```

The 16 names are `kenburns`, `shake`, `vibrate`, `vibrate-tight`, `tilt`,
`orbital`, `circle`, `dolly`, `dolly-out`, `pan`, `pull-back`, `roll`, `breathing`,
`figure-eight`, `ripple`, and `heat-shimmer`. The four depth presets need a
supplied map or a matching inferred cache entry. `--auto-depth` explicitly
permits generation/download when needed; selecting a depth preset alone never
downloads anything. A supplied map wins and may use any grid dimensions.
`burns depth` needs neither a rendering GPU nor FFmpeg.

Overrides: defaults → selected preset or strict saved config → ordered
`--set ID=VALUE` assignments → dedicated output/timing/seed flags. For example:

```sh
just cli render -i image.png -o buzz.mp4 --preset vibrate-tight \
  --set shake.speed=2 --fps 60 --duration 2 --seed 42 --crf 20
```

Final FPS is applied before duration-to-frame conversion. FPS alone preserves
the saved clip duration; explicit duration overrides it. `--set` is limited to
motion/depth/shake/warp/seed descriptors; use `--width`, `--height`, `--fps`, and
`--duration` for output and timing. Dedicated `--seed` wins over `--set seed=…`.
Assignments are one atomic transaction with final cross-field validation.

Existing outputs are refused unless `--overwrite` is explicit. SIGINT cancels
movie, PNG, and depth work without publishing a partial destination (atomic
publication already in progress completes). Progress/errors go to stderr;
project/schema/metadata JSON stays on stdout. MP4 defaults to x264 CRF 20;
browser quality is a separate encoder setting, not “CRF 20.”

## Portable projects

Version 1 projects are complete, strict documents. Unknown keys, missing fields
(including disabled settings), unsupported versions, non-finite values, and
out-of-range parameters are errors. Projects contain output size, rational frame
rate/frame count, seed, one camera/depth motion, optional shake, one optional
warp, and a depth-inversion adjustment. They never contain asset paths/bytes or
encoder settings.

Built-in presets replace camera, shake, and warp while preserving output, timing,
seed, and depth adjustments. Importing a saved project restores the whole
project. Disabling shake or warp retains its parameters; changing the camera or
warp type installs that variant's defaults.

Frames are evaluated by index, independent of playback speed or seek order.
Encoded duration is `frames * fps.den / fps.num`. Duration conversion rounds to
the nearest frame, with at least one frame. FPS edits preserve clip duration;
explicit duration edits use the final FPS. Active periodic harmonics must stay
strictly below half the frame rate; validation identifies the frequency and
suggests lowering speed/cycles or increasing FPS.

### Coordinates and assets

- Camera X/Y are fractions of **available crop travel**, not absolute source UV
  centers. Roll is in degrees, positive clockwise. Warp amplitudes and wavelengths
  are measured in image heights.
- Source PNG/JPEG/WebP images are EXIF-oriented once, composited over black, and
  retained at source resolution. The renderer covers and center-crops to the
  project output aspect without a mandatory upscaled intermediate.
- Imported depth is opaque 8/16-bit grayscale PNG at any grid resolution, aligned
  across the full UV extent of the oriented source. White is near. Imports are
  not independently reoriented, cropped, range-normalized, or inference-filtered.
- `depth.invert` changes sampling and inspection only. Save raw depth alongside
  the project; the project carries inversion, so do not invert the PNG again.

The `burns` crate with `default-features = false` exposes settings, presets,
timeline, image decoding, and depth-map utilities without wgpu, filesystem,
network, FFmpeg, GUI, or an inference runtime.

## Rendering from Rust

The default `gpu` feature uses Metal, Vulkan, or DX12 natively and WebGPU in a
secure browser context (HTTPS or loopback). There is no WebGL or flat-image
fallback. Unsupported adapters and oversized source/output textures are explicit
errors; the caller must resize rather than silently lose resolution.

```rust
use burns::{
    DepthMap, GpuContext, MotionProject, Renderer, RenderQuality, RenderView, SourceImage,
};

async fn capture(
    source: &SourceImage,
    depth: Option<&DepthMap>,
    project: &MotionProject,
    frame: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let context = GpuContext::headless().await?;
    let mut renderer = Renderer::new(context, project.output, RenderQuality::Final)?;
    renderer.set_source(source)?;
    renderer.set_depth(depth)?;
    renderer.render(project, frame, RenderView::Motion)?;
    let stride = project.output.width as usize * 4;
    let mut rgba = vec![0; project.output.rgba_len()?];
    renderer.read_rgba8(&mut rgba, stride).await?;
    Ok(rgba)
}
```

Keep the renderer and uploaded assets alive when rendering a clip. `Preview` caps
the target's long edge at 960; `Final` renders at 2× each axis and resolves with
an equal-weight 2×2 box. Both use the project output aspect and the same
camera/depth → shake → warp visual stack, evaluated as one inverse sampling map
and one source-color sample per rendered supersample.

`output_view()` is a reusable GPU texture view for egui/canvas presentation.
Capture/export only: `with_rgba8` lends mapped bytes plus the GPU row pitch to a
synchronous consumer, then unmaps. Stream only `width * 4` useful bytes per row
to an encoder; do not write GPU row padding. `read_rgba8` is the owned-buffer
convenience and preserves caller padding.

`Source` and effective `Depth` are transient inspection views. They do not change
the project; motion exports always request `RenderView::Motion`. Depth motion
without a map returns instructions to load/generate depth, while source
inspection remains available.

## Native media and export jobs

`burns-media` provides still/depth loading, safe output writing, final-quality
motion PNG capture, and the shared `RenderJob` used for native MP4 export.
`RenderRequest` snapshots the project and retains image/map assets through `Arc`;
preview playback speed and dropped preview frames never affect the offline
`0..N` frame loop.

MP4 export requires FFmpeg with libx264, even dimensions, and CRF 0–51 (default
20). It writes RGBA at the exact rational frame rate to H.264/yuv420p, with no
audio, `medium` preset, faststart, and explicit TV-range BT.709 matrix/primaries/
transfer tags. FFmpeg 9 also needs frame-level `setparams` tags so filters do not
replace output primaries/transfer with “unknown.”

Mapped GPU rows go directly to the encoder pipe; useful bytes are streamed
without GPU padding, an extra full-frame pixel buffer, or an unbounded queue.
Stderr is drained concurrently with bounded retained diagnostics. PNG capture
uses the same final renderer and accepts odd output dimensions, without FFmpeg
or an inference runtime.

Outputs are encoded into a unique owned temporary in the destination directory
and atomically published only on success. Existing files and symlinks (including
dangling ones) are refused unless overwrite is explicit; default publication
also refuses a destination that appears during encoding. Cancellation kills and
waits for only the owned encoder, removes its temporary, and preserves a previous
destination. Once atomic publication has begun, completion wins against late
cancellation. `Drop` joins the worker; failure is never reported as `Done`.

## Native workbench

```sh
just gui "still image.png"
just gui
```

The second form restores the last valid project and recent source path, stored
separately in platform application data. A missing recent image returns to the
open-image state. No image path is written into portable project JSON.

Open/drop one still; choose a preset or edit the descriptor-driven camera,
shake, warp, and seed controls. The GPU preview fits the output aspect inside
the black matte and caps its long edge at 960 pixels. Source and effective-depth
inspection do not change the project. Scrubbing pauses; paused edits redraw the
same frame. Space toggles playback except in text fields; unclaimed arrow,
Home, and End keys seek frames. FPS is an integer numerator/denominator, and
retiming preserves duration to the nearest representable frame.

Load a grayscale PNG or explicitly Generate depth. Changing the source clears
the previous map, keeps the project, and checks only the new image's map cache.
Missing depth shows the unchanged source with a remedy; 2D presets remain usable.
Save raw map preserves uninverted 16-bit samples; inversion stays in the project.
Open/save project and Reset operate on settings, not image/map contents.

The right inspector selects output dimensions, CRF, movie/frame paths, and
replacement permission. Movie and PNG jobs snapshot their inputs, run off the
UI thread, and always capture final-quality motion—even during source/depth
inspection. Cancel removes the owned incomplete output. Closing cancels and
joins owned jobs; an already-running ONNX call must finish before shutdown.

## Browser workbench

[Open the hosted workbench](https://kylegrantlucas.github.io/burns-rs/) in a
WebGPU-capable browser.

On a fresh checkout, run `pnpm --dir site install --frozen-lockfile` inside
`nix develop` before starting the browser workbench.

```sh
just site-serve 8472
# Open http://127.0.0.1:8472
just site
```

`just site` regenerates matching WASM bindings, installs with the frozen pnpm
lockfile, and builds the static `site/dist` output. Development and build scripts
copy the pinned ONNX CPU runtime assets to the same-origin `public/ort` path.
There is no backend, account, upload endpoint, or service worker.
Serve production files over HTTPS or localhost for WebGPU.

`.github/workflows/pages.yml` builds pull requests and deploys `main` to GitHub
Pages. It runs `just site /burns-rs/` so generated asset and worker URLs include
the repository path. The default `just site` build still serves from `/`.

The generated `site/src/wasm/burns_bg.wasm` is intentionally untracked.
`just wasm` recreates it; `just site` and `just site-serve` run that step for you.
Run `just wasm` before invoking the site's Vite scripts directly.

The workbench supports the same project/preset/stack, source/depth inspection,
rational timing, transport, and PNG/depth/project workflows. Rust owns image
decoding, project validation, defaults/descriptors, preprocessing, motion, and
rendering; TypeScript manages the DOM, playback, worker, and encoder lifecycle.
The public WASM interface is in `site/src/wasm/burns.d.ts`. `new BurnsEngine()`
creates a usable pure project before fallible `initCanvas`; getters remain
available during capture, while conflicting mutations report busy.

The slate editing bay keeps the image on a true-black matte. Use the inspector's
Motion, Depth, and Output tabs to move between parameter editing, map generation/
import, and dimensions/timing. Camera, shake, warp, and project groups fold
independently; Show help reveals their parameter guidance without changing any
settings. Arrow keys, Home, and End navigate the inspector tabs. The Project
menu opens/saves portable JSON, while frame PNG and MP4 actions stay beside the
transport, including on narrow screens. The amber ruler labels actual frame
indices from the Rust timeline; it is not a realtime recording indicator.

Generate is the only action that permits downloading the pinned public model.
A dedicated CPU worker verifies its SHA-256 and uses the shared Rust preparation
and postprocessing. Verified model bytes use CacheStorage; generated raw maps
use IndexedDB with the shared source-content key. A quota/persistence failure
is reported without discarding the usable in-memory result. Images and imported
maps remain on the device; only app/runtime assets and the requested model are
fetched. Save project plus raw depth and reopen them with the same image in
`burns frame` to reproduce a selected frame, including depth inversion.

MP4 export uses Mediabunny/WebCodecs AVC at browser quality **high**, one video
track and no audio. It renders every deterministic frame offline, awaits encoder
backpressure, and finalizes before download; it does not record realtime canvas
playback. Export freezes conflicting controls and restores preview/frame/view
after completion, cancellation, or error. A cancellation arriving during muxer
finalization waits for cleanup and discards that result.

AVC support is checked for the requested dimensions and rational rate. Unsupported
AVC disables only MP4; PNG/depth/project and the copyable CLI alternative remain.
Missing WebGPU disables preview and browser capture without displaying a fake
renderer; pure settings/project/CLI controls still work. Browser quality and
codec limits are distinct from native x264 CRF.

## Checks

```sh
just check
cargo test -p burns --no-default-features
cargo test -p burns --test render -- --ignored
cargo test -p burns-cli --test cli -- --ignored
cargo test -p burns-media -- --ignored
cargo check -p burns-wasm --target wasm32-unknown-unknown
pnpm --dir site typecheck
just site
```

Ignored rendering/media checks intentionally require a real GPU and FFmpeg;
they do not substitute a Noop adapter or silently pass without the prerequisite.

## Local depth

`burns-depth` is a separate pure preprocessing/postprocessing crate. Its optional
native feature loads CPU ONNX Runtime dynamically; missing ORT affects inference
only, never 2D or supplied-map rendering. The project shell sets `ORT_DYLIB_PATH`;
packaged applications may instead supply a sibling runtime.

Generation uses the Apache-2.0 **Depth Anything V2 Small** single-file fp32 graph:
[`onnx-community/depth-anything-v2-small`](https://huggingface.co/onnx-community/depth-anything-v2-small),
revision `4472b7362082ad9968fee890ca0f1e5aca36b93d`, 99,060,839 bytes,
SHA-256 `afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c`.
No larger or differently quantized model is substituted.

The shorter input edge is scaled to 518 and both dimensions rounded to multiples
of 14, preserving aspect (16:9 becomes 924×518). Generated predictions are
min/max-normalized, reflected-Gaussian-smoothed, maximum-filtered, then quantized
to 16-bit levels **before first use**, so a cached PNG has identical geometry.
A constant prediction remains a zero map, not a made-up gradient.

Only explicit generation permits model acquisition. Verified weights are cached
under the platform cache directory at `burns/models/<sha256>.onnx`; inferred raw
maps use `burns/depth/<source-content-key>.png`. Downloads publish atomically only
after length/hash verification. Cancellation interrupts a stalled download and
removes its temporary file; an already-running CPU inference call completes but
its cancelled result is discarded. Cache persistence failure is reported while
keeping the successful map usable in memory. Imported maps never enter this
inference postprocessing/cache path.

Depth is a height field, not recovered scene layers. Smooth/gradient predictions,
mirrored edges, and disocclusions can remain visible; there is no generated fill.

## License

Original code is available under MIT OR Apache-2.0; see `LICENSE-MIT` and
`LICENSE-APACHE`. Private verification artwork is not bundled with this project.

The static site retains B612's OFL, ONNX Runtime Web's MIT, and Mediabunny's
MPL-2.0 notices under `site/public/notices`, with visible runtime/font/source
links in the workbench. The separately downloaded model uses Apache-2.0; its
pinned source/license links are shown before generation. Keep those notices and
upstream source links when redistributing the corresponding third-party assets.

[Showing lines 1-324 of 325. Use :325 to continue]