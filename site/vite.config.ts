import { defaultClientConditions, defineConfig } from 'vite';

export default defineConfig({
  resolve: {
    // `onnxruntime-web/wasm` then resolves to the build that loads its .mjs/.wasm
    // from `env.wasm.wasmPaths` (public/ort) instead of embedding/emitting copies.
    conditions: [...defaultClientConditions, 'onnxruntime-web-use-extern-wasm'],
  },
  // The depth worker code-splits ONNX Runtime behind a dynamic import.
  worker: { format: 'es' },
  // Keep ONNX Runtime's module URL unbundled in dev, as in the build.
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
