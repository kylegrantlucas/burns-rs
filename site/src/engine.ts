/**
 * Typed facade over the generated wasm-bindgen `BurnsEngine`.
 *
 * Rust owns the canonical project, defaults, ranges, presets and validation.
 * This module only parses the JSON the bindings return so the UI can read
 * widget values; every edit goes through a Rust setter and the returned
 * canonical JSON replaces the cached copy.
 */
import init, { BurnsEngine, modelManifestJson, presetsJson, schemaJson, type SettingValue } from './wasm/burns.js';

export { depthCacheKey, postprocessDepth, prepareDepthInput } from './wasm/burns.js';
export type { SettingValue };

export interface Size {
  width: number;
  height: number;
}

export interface FrameRate {
  num: number;
  den: number;
}

/** The fields of the canonical project the workbench reads directly. */
export interface Project {
  version: number;
  output: Size;
  timeline: { fps: FrameRate; frames: number };
  seed: number;
  camera: { type: string } & Record<string, unknown>;
  depth: { invert: boolean };
  shake: { enabled: boolean } & Record<string, unknown>;
  warp: { type: string; enabled: boolean } & Record<string, unknown>;
}

export type SettingKind =
  | { type: 'float'; min: number; max: number; step: number }
  | { type: 'int'; min: number; max: number }
  | { type: 'bool' }
  | { type: 'choice'; options: { value: string; label: string }[] };

export interface Visibility {
  id: string;
  value: string;
}

export interface SettingDescriptor {
  id: string;
  label: string;
  unit: string;
  help: string;
  kind: SettingKind;
  default: SettingValue;
  visible_when: Visibility[];
}

export interface SettingGroup {
  id: string;
  label: string;
  help: string;
  settings: SettingDescriptor[];
}

export interface SettingsSchema {
  version: number;
  defaults: Project;
  groups: SettingGroup[];
}

export interface PresetInfo {
  name: string;
  label: string;
  requires_depth: boolean;
  periodic: boolean;
}

export interface ModelManifest {
  id: string;
  name: string;
  revision: string;
  url: string;
  bytes: number;
  sha256: string;
  license: string;
  input_name: string;
  output_name: string;
  preprocessing_version: number;
}

export type RenderView = 'motion' | 'source' | 'depth';

export interface TargetInfo {
  width: number;
  height: number;
  final: boolean;
  outputWidth: number;
  outputHeight: number;
}

let wasmReady: Promise<void> | null = null;

/** Instantiates the generated module once (main thread or worker). */
export function loadWasm(): Promise<void> {
  wasmReady ??= init().then(() => undefined);
  return wasmReady;
}

/** Text of a thrown wasm-bindgen error or any other failure, ending as a sentence. */
export function errorMessage(error: unknown): string {
  const detail = error instanceof Error ? error.message.trim() || error.name : String(error);
  const text = detail.trim() || 'Unknown error';
  return /[.!?]$/.test(text) ? text : `${text}.`;
}

export function durationSeconds(project: Project): number {
  const { fps, frames } = project.timeline;
  return (frames * fps.den) / fps.num;
}

export function frameSeconds(project: Project, frame: number): number {
  const { fps } = project.timeline;
  return (frame * fps.den) / fps.num;
}

/** Whether the project's camera samples the depth map (reads the project itself). */
export function cameraUsesDepth(project: Project): boolean {
  return project.camera.type === 'depth';
}

export function readModelManifest(): ModelManifest {
  return JSON.parse(modelManifestJson()) as ModelManifest;
}

/**
 * One editing session. Wraps the generated engine and keeps the last
 * canonical project JSON returned by Rust.
 */
export class Engine {
  readonly raw = new BurnsEngine();
  readonly schema: SettingsSchema = JSON.parse(schemaJson()) as SettingsSchema;
  readonly presets: PresetInfo[] = JSON.parse(presetsJson()) as PresetInfo[];
  #json: string;
  #project: Project;

  constructor() {
    this.#json = this.raw.getProjectJson();
    this.#project = JSON.parse(this.#json) as Project;
  }

  get project(): Project {
    return this.#project;
  }

  get projectJson(): string {
    return this.#json;
  }

  #accept(json: string): Project {
    this.#json = json;
    this.#project = JSON.parse(json) as Project;
    return this.#project;
  }

  setSetting(id: string, raw: string): Project {
    return this.#accept(this.raw.setSetting(id, raw));
  }

  getSetting(id: string): SettingValue {
    return this.raw.getSetting(id);
  }

  setTiming(fpsNum: number, fpsDen: number, seconds: number): Project {
    return this.#accept(this.raw.setTiming(fpsNum, fpsDen, seconds));
  }

  setOutputSize(width: number, height: number): Project {
    return this.#accept(this.raw.setOutputSize(width, height));
  }

  applyPreset(name: string): Project {
    return this.#accept(this.raw.applyPreset(name));
  }

  setProjectJson(json: string): Project {
    return this.#accept(this.raw.setProjectJson(json));
  }

  loadImage(bytes: Uint8Array): Project {
    return this.#accept(this.raw.loadImage(bytes));
  }

  loadDepth(bytes: Uint8Array): Project {
    return this.#accept(this.raw.loadDepth(bytes));
  }

  sourceSize(): Size {
    return JSON.parse(this.raw.sourceSizeJson()) as Size;
  }

  depthSize(): Size {
    return JSON.parse(this.raw.depthSizeJson()) as Size;
  }

  target(): TargetInfo {
    return JSON.parse(this.raw.targetJson()) as TargetInfo;
  }

  /** The current value of a choice setting as its stable JSON string. */
  choiceValue(id: string): string | undefined {
    const value = this.getSetting(id);
    if (value.type !== 'choice') return undefined;
    for (const group of this.schema.groups) {
      for (const descriptor of group.settings) {
        if (descriptor.id === id && descriptor.kind.type === 'choice') {
          return descriptor.kind.options[value.value]?.value;
        }
      }
    }
    return undefined;
  }

  /** Every `visible_when` condition must hold (AND). */
  isVisible(descriptor: SettingDescriptor): boolean {
    return descriptor.visible_when.every((condition) => this.choiceValue(condition.id) === condition.value);
  }
}
