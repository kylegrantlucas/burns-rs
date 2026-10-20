/**
 * Depth generation off the UI thread.
 *
 * The pinned model comes from the Rust manifest only. Its bytes are verified
 * (length and SHA-256 via WebCrypto) after every download and every cache hit.
 * Preprocessing and postprocessing run in Rust/WASM; ONNX Runtime Web runs the
 * graph on the CPU WASM backend from same-origin runtime files.
 */
import type * as Ort from 'onnxruntime-web/wasm';
import type { InferenceSession, Tensor } from 'onnxruntime-web/wasm';
import {
  errorMessage,
  loadWasm,
  postprocessDepth,
  prepareDepthInput,
  readModelManifest,
  type ModelManifest,
} from './engine';
import type { DepthProgress, WorkerReply, WorkerRequest } from './depth';

interface WorkerScope {
  postMessage(message: WorkerReply, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
}

const scope = self as unknown as WorkerScope;

/** Versioned CacheStorage bucket; entries are keyed by the pinned revision URL. */
const MODEL_CACHE = 'burns-models-v1';

let session: InferenceSession | null = null;
let ortModule: typeof Ort | null = null;

function hex(digest: ArrayBuffer): string {
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

async function verify(bytes: Uint8Array<ArrayBuffer>, manifest: ModelManifest): Promise<string | null> {
  if (bytes.byteLength !== manifest.bytes) {
    return `expected ${manifest.bytes} bytes, got ${bytes.byteLength}`;
  }
  const digest = hex(await crypto.subtle.digest('SHA-256', bytes));
  return digest === manifest.sha256 ? null : `SHA-256 ${digest} does not match ${manifest.sha256}`;
}

async function download(manifest: ModelManifest, report: (progress: DepthProgress) => void): Promise<Uint8Array<ArrayBuffer>> {
  const response = await fetch(manifest.url, { credentials: 'omit', referrerPolicy: 'no-referrer' });
  if (!response.ok || !response.body) {
    throw new Error(`Model download failed: HTTP ${response.status} from ${new URL(manifest.url).host}`);
  }
  const bytes = new Uint8Array(manifest.bytes);
  const reader = response.body.getReader();
  let loaded = 0;
  let lastReport = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.byteLength > manifest.bytes) {
      await reader.cancel();
      throw new Error(`Model download is larger than the pinned ${manifest.bytes} bytes; refusing it`);
    }
    bytes.set(value, loaded);
    loaded += value.byteLength;
    if (loaded - lastReport >= 1 << 20 || loaded === manifest.bytes) {
      lastReport = loaded;
      report({ stage: 'download', loaded, total: manifest.bytes });
    }
  }
  if (loaded !== manifest.bytes) {
    throw new Error(`Model download ended after ${loaded} of ${manifest.bytes} bytes`);
  }
  return bytes;
}

/** Returns verified model bytes and a note on where they came from. */
async function modelBytes(
  manifest: ModelManifest,
  report: (progress: DepthProgress) => void,
): Promise<{ bytes: Uint8Array<ArrayBuffer>; note: string }> {
  report({ stage: 'model-cache' });
  let cache: Cache | null = null;
  let cacheNote = '';
  try {
    cache = await caches.open(MODEL_CACHE);
  } catch (error) {
    cacheNote = ` The browser cache is unavailable, so the model will download again next time: ${errorMessage(error)}`;
  }
  if (cache) {
    const hit = await cache.match(manifest.url);
    if (hit) {
      report({ stage: 'verify' });
      const bytes = new Uint8Array(await hit.arrayBuffer());
      const problem = await verify(bytes, manifest);
      if (!problem) return { bytes, note: 'Model loaded from the browser cache and verified.' };
      await cache.delete(manifest.url);
      cacheNote = ` A cached copy failed verification (${problem}) and was replaced.`;
    }
  }
  report({ stage: 'download', loaded: 0, total: manifest.bytes });
  const bytes = await download(manifest, report);
  report({ stage: 'verify' });
  const problem = await verify(bytes, manifest);
  if (problem) throw new Error(`Downloaded model failed verification: ${problem}`);
  if (cache) {
    try {
      await cache.put(
        manifest.url,
        new Response(bytes, { headers: { 'content-type': 'application/octet-stream', 'content-length': String(bytes.byteLength) } }),
      );
      return { bytes, note: `Model downloaded, verified and cached.${cacheNote}` };
    } catch (error) {
      return { bytes, note: `Model downloaded and verified, but it could not be cached: ${errorMessage(error)}${cacheNote}` };
    }
  }
  return { bytes, note: `Model downloaded and verified.${cacheNote}` };
}

function requireTensor(
  metadata: readonly InferenceSession.ValueMetadata[],
  name: string,
  rank: number,
  role: string,
): void {
  const value = metadata.find((entry) => entry.name === name);
  if (!value) throw new Error(`The model has no ${role} named ${name}`);
  if (!value.isTensor || value.type !== 'float32' || value.shape.length !== rank) {
    throw new Error(`The model ${role} ${name} must be a float32 rank-${rank} tensor`);
  }
  if (typeof value.shape[0] === 'number' && value.shape[0] !== 1) {
    throw new Error(`The model ${role} ${name} must have batch size 1`);
  }
}

async function loadSession(
  manifest: ModelManifest,
  report: (progress: DepthProgress) => void,
): Promise<{ ort: typeof Ort; session: InferenceSession; note: string }> {
  if (session && ortModule) return { ort: ortModule, session, note: 'Model already loaded in this session.' };
  const { bytes, note } = await modelBytes(manifest, report);
  report({ stage: 'runtime' });
  // Dynamic on purpose: the 33 MB runtime package loads only after an explicit Generate.
  const ort = await import('onnxruntime-web/wasm');
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = new URL(`${import.meta.env.BASE_URL}ort/`, self.location.href).href;
  const created = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
  if (created.inputMetadata.length !== 1 || created.outputMetadata.length !== 1) {
    throw new Error('The pinned model must have exactly one input and one output');
  }
  requireTensor(created.inputMetadata, manifest.input_name, 4, 'input');
  requireTensor(created.outputMetadata, manifest.output_name, 3, 'output');
  const input = created.inputMetadata[0];
  if (!input.isTensor || input.shape[1] !== 3) {
    throw new Error(`Model input ${manifest.input_name} must have exactly three RGB channels`);
  }
  ortModule = ort;
  session = created;
  return { ort, session: created, note };
}

async function generate(request: WorkerRequest): Promise<void> {
  const report = (progress: DepthProgress) => scope.postMessage({ type: 'progress', id: request.id, progress });
  await loadWasm();
  const manifest = readModelManifest();
  const { ort, session: model, note } = await loadSession(manifest, report);

  report({ stage: 'prepare' });
  const input = prepareDepthInput(request.image);
  const [batch, channels, height, width] = input.shape;
  if (batch !== 1 || channels !== 3 || input.values.length !== channels * height * width) {
    throw new Error(`Prepared input has unexpected shape [${input.shape.join(', ')}]`);
  }

  report({ stage: 'inference' });
  const feeds: Record<string, Tensor> = {
    [manifest.input_name]: new ort.Tensor('float32', input.values, input.shape),
  };
  const outputs = await model.run(feeds);
  const output = outputs[manifest.output_name];
  if (!output || output.type !== 'float32' || output.dims.length !== 3 || output.dims[0] !== 1) {
    throw new Error(`Model output ${manifest.output_name} must be float32 [1, H, W]`);
  }
  const outHeight = output.dims[1];
  const outWidth = output.dims[2];
  if (outHeight !== height || outWidth !== width) {
    throw new Error(`Model output ${manifest.output_name} is ${outWidth}×${outHeight}; expected ${width}×${height}`);
  }
  const values = output.data as Float32Array;

  report({ stage: 'postprocess' });
  const png = postprocessDepth(outWidth, outHeight, values);
  output.dispose();
  scope.postMessage(
    { type: 'result', id: request.id, png, width: outWidth, height: outHeight, modelNote: note },
    [png.buffer],
  );
}

scope.onmessage = (event) => {
  const request = event.data;
  generate(request).catch((error: unknown) =>
    scope.postMessage({ type: 'error', id: request.id, message: errorMessage(error) }),
  );
};
