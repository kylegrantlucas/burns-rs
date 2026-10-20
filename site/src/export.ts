/**
 * Offline H.264 MP4 export with Mediabunny: every frame index is rendered
 * once, captured from the WebGPU canvas in the same task, and encoded with
 * backpressure. The browser's encoder runs at `new Quality('high')`; there is
 * no CRF (that is an x264 setting the browser does not expose).
 */
import { BufferTarget, CanvasSource, Mp4OutputFormat, Output, Quality, canEncodeVideo } from 'mediabunny';
import { errorMessage, type FrameRate } from './engine';

export interface Mp4Settings {
  width: number;
  height: number;
  fps: FrameRate;
  frames: number;
}

/** `null` when the browser can encode these exact settings, otherwise why not. */
export async function avcUnsupportedReason(settings: Mp4Settings): Promise<string | null> {
  if (typeof VideoEncoder === 'undefined') {
    return 'This browser has no WebCodecs video encoder.';
  }
  const supported = await canEncodeVideo('avc', {
    width: settings.width,
    height: settings.height,
    frameRate: settings.fps.num / settings.fps.den,
    quality: new Quality('high'),
  });
  return supported
    ? null
    : `This browser cannot encode H.264 at ${settings.width}×${settings.height}, ${settings.fps.num}/${settings.fps.den} fps.`;
}

export class Mp4Export {
  readonly #canvas: HTMLCanvasElement;
  readonly #settings: Mp4Settings;
  readonly #renderFrame: (index: number) => void;
  readonly #progress: (done: number, total: number) => void;
  #output: Output | null = null;
  #cancelled = false;
  #cleanup: Promise<void> | null = null;

  /**
   * `renderFrame(n)` must synchronously render and present frame `n` (Motion
   * view) to `canvas` at the output size.
   */
  constructor(
    canvas: HTMLCanvasElement,
    settings: Mp4Settings,
    renderFrame: (index: number) => void,
    progress: (done: number, total: number) => void,
  ) {
    this.#canvas = canvas;
    this.#settings = settings;
    this.#renderFrame = renderFrame;
    this.#progress = progress;
  }

  /** Resolves to the finalized MP4, or `null` when cancelled. Throws on any failure. */
  async run(): Promise<Blob | null> {
    if (this.#cancelled) return null;
    const { fps, frames } = this.#settings;
    const target = new BufferTarget();
    const output = new Output({ format: new Mp4OutputFormat(), target });
    this.#output = output;
    const source = new CanvasSource(this.#canvas, { codec: 'avc', quality: new Quality('high') });
    output.addVideoTrack(source, { frameRate: fps.num / fps.den });
    const frameDuration = fps.den / fps.num;
    try {
      await output.start();
      for (let index = 0; index < frames && !this.#cancelled; index += 1) {
        // Render and capture in the same task: nothing may run between these two calls.
        this.#renderFrame(index);
        await source.add((index * fps.den) / fps.num, frameDuration);
        this.#progress(index + 1, frames);
      }
      if (this.#cancelled) {
        await this.#release();
        return null;
      }
      await output.finalize();
      if (this.#cancelled) return null;
      if (!target.buffer) throw new Error('The MP4 muxer finished without producing a file.');
      return new Blob([target.buffer], { type: 'video/mp4' });
    } catch (error) {
      try {
        await this.#release();
      } catch (cleanup) {
        if (cleanup === error) throw cleanup;
        throw new Error(`${errorMessage(error)} MP4 cleanup also failed: ${errorMessage(cleanup)}`);
      }
      if (this.#cancelled) return null;
      throw error;
    }
  }

  /** Requests cancellation; an already-finalizing muxer finishes and is discarded. */
  async cancel(): Promise<void> {
    this.#cancelled = true;
    await this.#release();
  }

  #release(): Promise<void> {
    const output = this.#output;
    if (!this.#cleanup && output && output.state !== 'finalized') {
      this.#cleanup = output.cancel();
    }
    return this.#cleanup ?? Promise.resolve();
  }
}
