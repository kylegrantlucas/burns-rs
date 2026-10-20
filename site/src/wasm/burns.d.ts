/* tslint:disable */
/* eslint-disable */

/** Model input tensor from `prepareDepthInput`: NCHW, normalized. */
export interface DepthInput {
    values: Float32Array;
    shape: [1, 3, number, number];
}

/** Typed setting value; choices are option indices. */
export type SettingValue =
| { type: "float"; value: number }
| { type: "int"; value: number }
| { type: "bool"; value: boolean }
| { type: "choice"; value: number };



/**
 * One editing and preview session.
 */
export class BurnsEngine {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Installs a preset's camera, shake, and warp, keeping output, timing,
     * seed, and depth adjustments.
     */
    applyPreset(name: string): string;
    /**
     * The raw depth map as a 16-bit grayscale PNG (never inverted).
     */
    depthPng(): Uint8Array;
    /**
     * `{"width":W,"height":H}` of the depth map.
     */
    depthSizeJson(): string;
    /**
     * Resolves to a PNG of frame `index`: always the Motion view at the exact
     * project output size with 2x supersampling, regardless of the preview
     * view or target, which are restored afterwards.
     */
    framePng(index: number): Promise<Uint8Array>;
    /**
     * Canonical project JSON.
     */
    getProjectJson(): string;
    /**
     * Current typed value of an active setting.
     */
    getSetting(id: string): SettingValue;
    hasDepth(): boolean;
    hasSource(): boolean;
    /**
     * Acquires a WebGPU device for `canvas` (secure contexts only), builds the
     * renderer for the current target, and uploads the loaded assets. Rejects
     * with an actionable error otherwise; the project and assets are kept.
     */
    initCanvas(canvas: HTMLCanvasElement): Promise<void>;
    isGpuReady(): boolean;
    /**
     * Decodes a raw 8/16-bit grayscale depth PNG exactly as stored (no
     * postprocessing, inversion, or reorientation). Returns the canonical project JSON.
     */
    loadDepth(bytes: Uint8Array): string;
    /**
     * Decodes a PNG/JPEG/WebP still (EXIF-oriented, alpha over black) as the
     * source and clears the depth map; uploads it once when a GPU exists. On
     * error nothing changes. Returns the canonical project JSON.
     */
    loadImage(bytes: Uint8Array): string;
    /**
     * Default project (`kenburns`), no assets, no GPU. Never touches WebGPU.
     */
    constructor();
    /**
     * Renders `view` (`"motion"`, `"source"`, or `"depth"`) of frame `index`
     * and presents it to the canvas before returning, so a capture stream
     * sampled right after this call sees exactly this frame.
     */
    renderFrame(index: number, view: "motion" | "source" | "depth"): void;
    /**
     * Sets the preview target (`finalQuality`: exact size with 2x
     * supersampling; otherwise the long edge is capped at 960). With a GPU
     * the canvas drawing buffer is resized to the output size and cleared.
     */
    resizeTarget(width: number, height: number, finalQuality: boolean): void;
    /**
     * Sets the project output size. The preview target is separate: call
     * `resizeTarget` too for an aspect-correct preview.
     */
    setOutputSize(width: number, height: number): string;
    /**
     * Strictly parses and validates a whole project, replacing the current one.
     */
    setProjectJson(json: string): string;
    /**
     * Sets one active setting from text (number, `true`/`false`, or a
     * choice's stable value or index) atomically; returns the canonical project.
     */
    setSetting(id: string, raw: string): string;
    /**
     * Replaces timing with `Timeline::from_duration(fpsNum/fpsDen, durationSeconds)`.
     */
    setTiming(fpsNum: number, fpsDen: number, durationSeconds: number): string;
    /**
     * `{"width":W,"height":H}` of the oriented source image.
     */
    sourceSizeJson(): string;
    /**
     * `{"width","height","final","outputWidth","outputHeight"}`: the
     * requested target and the canvas/output texture size it produces.
     */
    targetJson(): string;
}

/**
 * Content-addressed depth cache key of a still, shared with native.
 */
export function depthCacheKey(imageBytes: Uint8Array): string;

/**
 * The pinned depth model: URL, size, SHA-256, tensor names, preprocessing version.
 */
export function modelManifestJson(): string;

/**
 * Postprocesses raw `width * height` model output (larger is nearer) and
 * encodes the canonical 16-bit grayscale depth PNG.
 */
export function postprocessDepth(width: number, height: number, values: Float32Array): Uint8Array;

/**
 * Decodes a still exactly like `loadImage` and returns the model input tensor.
 */
export function prepareDepthInput(imageBytes: Uint8Array): DepthInput;

/**
 * The default project with the named preset applied, as canonical JSON.
 */
export function presetJson(name: string): string;

/**
 * Built-in preset metadata (`name`, `label`, `requires_depth`, `periodic`) in display order.
 */
export function presetsJson(): string;

/**
 * The settings registry: defaults, groups, ranges, choices, and visibility.
 */
export function schemaJson(): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_burnsengine_free: (a: number, b: number) => void;
    readonly burnsengine_applyPreset: (a: number, b: number, c: number) => [number, number, number, number];
    readonly burnsengine_depthPng: (a: number) => [number, number, number, number];
    readonly burnsengine_depthSizeJson: (a: number) => [number, number, number, number];
    readonly burnsengine_framePng: (a: number, b: number) => any;
    readonly burnsengine_getProjectJson: (a: number) => [number, number, number, number];
    readonly burnsengine_getSetting: (a: number, b: number, c: number) => [number, number, number];
    readonly burnsengine_hasDepth: (a: number) => number;
    readonly burnsengine_hasSource: (a: number) => number;
    readonly burnsengine_initCanvas: (a: number, b: any) => any;
    readonly burnsengine_isGpuReady: (a: number) => number;
    readonly burnsengine_loadDepth: (a: number, b: number, c: number) => [number, number, number, number];
    readonly burnsengine_loadImage: (a: number, b: number, c: number) => [number, number, number, number];
    readonly burnsengine_new: () => number;
    readonly burnsengine_renderFrame: (a: number, b: number, c: number, d: number) => [number, number];
    readonly burnsengine_resizeTarget: (a: number, b: number, c: number, d: number) => [number, number];
    readonly burnsengine_setOutputSize: (a: number, b: number, c: number) => [number, number, number, number];
    readonly burnsengine_setProjectJson: (a: number, b: number, c: number) => [number, number, number, number];
    readonly burnsengine_setSetting: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly burnsengine_setTiming: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly burnsengine_sourceSizeJson: (a: number) => [number, number, number, number];
    readonly burnsengine_targetJson: (a: number) => [number, number];
    readonly depthCacheKey: (a: number, b: number) => [number, number, number, number];
    readonly modelManifestJson: () => [number, number];
    readonly postprocessDepth: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly prepareDepthInput: (a: number, b: number) => [number, number, number];
    readonly presetJson: (a: number, b: number) => [number, number, number, number];
    readonly presetsJson: () => [number, number];
    readonly schemaJson: () => [number, number];
    readonly wasm_bindgen_b95919525d99f795___convert__closures_____invoke___js_sys_ebd39486fc75720d___Function_fn_wasm_bindgen_b95919525d99f795___JsValue_____wasm_bindgen_b95919525d99f795___sys__Undefined___js_sys_ebd39486fc75720d___Function_fn_wasm_bindgen_b95919525d99f795___JsValue_____wasm_bindgen_b95919525d99f795___sys__Undefined_______true_: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen_b95919525d99f795___convert__closures_____invoke___wasm_bindgen_b95919525d99f795___JsValue__core_7d5f0a2ba6a62c33___result__Result_____wasm_bindgen_b95919525d99f795___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_b95919525d99f795___convert__closures_____invoke___wasm_bindgen_b95919525d99f795___sys__JsNullable_wgpu_d996bd1c9f751be6___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_7d5f0a2ba6a62c33___result__Result_____wasm_bindgen_b95919525d99f795___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_b95919525d99f795___convert__closures_____invoke___wasm_bindgen_b95919525d99f795___sys__JsNullable_wgpu_d996bd1c9f751be6___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_7d5f0a2ba6a62c33___result__Result_____wasm_bindgen_b95919525d99f795___JsError___true__32: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_b95919525d99f795___convert__closures_____invoke___wasm_bindgen_b95919525d99f795___sys__JsNullable_wgpu_d996bd1c9f751be6___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_7d5f0a2ba6a62c33___result__Result_____wasm_bindgen_b95919525d99f795___JsError___true__33: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_b95919525d99f795___convert__closures_____invoke___wgpu_d996bd1c9f751be6___backend__webgpu__webgpu_sys__gen_GpuDeviceLostInfo__GpuDeviceLostInfo______true_: (a: number, b: number, c: any) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
