/**
 * Main-thread side of depth generation. The module worker is created lazily
 * on the first explicit Generate and terminated on Cancel, which also aborts
 * an in-flight model download.
 */

export type DepthStage = 'model-cache' | 'download' | 'verify' | 'runtime' | 'prepare' | 'inference' | 'postprocess';

export interface DepthProgress {
  stage: DepthStage;
  loaded?: number;
  total?: number;
}

export interface WorkerRequest {
  type: 'generate';
  id: number;
  image: Uint8Array;
}

export type WorkerReply =
  | { type: 'progress'; id: number; progress: DepthProgress }
  | { type: 'result'; id: number; png: Uint8Array; width: number; height: number; modelNote: string }
  | { type: 'error'; id: number; message: string };

export interface GenerateHandlers {
  progress(progress: DepthProgress): void;
  result(png: Uint8Array, modelNote: string): void;
  error(message: string): void;
}

export class DepthGenerator {
  #worker: Worker | null = null;
  #job = 0;
  #active: { id: number; handlers: GenerateHandlers } | null = null;

  get running(): boolean {
    return this.#active !== null;
  }

  /**
   * Starts one generation. The worker receives its own copy of the image
   * bytes, so the caller's buffer is never detached by the transfer.
   */
  start(image: Uint8Array, handlers: GenerateHandlers): void {
    this.cancel();
    this.#job += 1;
    const id = this.#job;
    this.#active = { id, handlers };
    const worker = (this.#worker ??= this.#spawn());
    const copy = image.slice();
    const request: WorkerRequest = { type: 'generate', id, image: copy };
    worker.postMessage(request, [copy.buffer]);
  }

  /** Terminates the worker (aborting any download) and forgets the job. */
  cancel(): void {
    if (!this.#active) return;
    this.#active = null;
    this.#worker?.terminate();
    this.#worker = null;
  }

  #spawn(): Worker {
    const worker = new Worker(new URL('./depth-worker.ts', import.meta.url), { type: 'module', name: 'burns-depth' });
    worker.onmessage = (event: MessageEvent<WorkerReply>) => {
      const reply = event.data;
      const active = this.#active;
      if (!active || reply.id !== active.id) return;
      if (reply.type === 'progress') {
        active.handlers.progress(reply.progress);
      } else if (reply.type === 'result') {
        this.#active = null;
        active.handlers.result(reply.png, reply.modelNote);
      } else {
        this.#active = null;
        active.handlers.error(reply.message);
      }
    };
    worker.onerror = (event) => {
      event.preventDefault();
      const active = this.#active;
      this.#active = null;
      worker.terminate();
      if (this.#worker === worker) this.#worker = null;
      active?.handlers.error(`The depth worker failed to start: ${event.message || 'script error'}.`);
    };
    return worker;
  }
}
