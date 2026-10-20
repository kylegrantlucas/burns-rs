// Copies same-origin runtime files and license notices from the installed
// packages into public/ before `vite` and `vite build`:
//   public/ort/      ONNX Runtime Web CPU (WASM) loader and binary
//   public/notices/  B612 OFL, ONNX Runtime MIT, Mediabunny MPL-2.0 texts
// Nothing is fetched from a CDN at runtime.
import { copyFile, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const site = dirname(dirname(fileURLToPath(import.meta.url)));
// Direct dependencies only; some packages do not export ./package.json.
const packageDir = (name) => join(site, 'node_modules', name);

const pins = JSON.parse(await readFile(join(site, 'package.json'), 'utf8')).dependencies;

async function installed(name) {
  const dir = packageDir(name);
  const { version } = JSON.parse(await readFile(join(dir, 'package.json'), 'utf8'));
  if (version !== pins[name]) {
    throw new Error(`${name} ${version} is installed but package.json pins ${pins[name]}; run pnpm install`);
  }
  return { dir, version };
}

const ort = await installed('onnxruntime-web');
const mediabunny = await installed('mediabunny');
const b612 = await installed('@fontsource/b612');

const ortOut = join(site, 'public', 'ort');
await rm(ortOut, { recursive: true, force: true });
await mkdir(ortOut, { recursive: true });
// Single-threaded CPU build: no SharedArrayBuffer or cross-origin isolation needed.
for (const file of ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
  await copyFile(join(ort.dir, 'dist', file), join(ortOut, file));
}

const noticesOut = join(site, 'public', 'notices');
await rm(noticesOut, { recursive: true, force: true });
await mkdir(noticesOut, { recursive: true });
await copyFile(join(b612.dir, 'LICENSE'), join(noticesOut, 'B612-OFL.txt'));
await copyFile(join(mediabunny.dir, 'LICENSE'), join(noticesOut, 'mediabunny-MPL-2.0.txt'));

// The npm package carries its license in the bundle banner rather than a file.
const banner = (await readFile(join(ort.dir, 'dist', 'ort.wasm.bundle.min.mjs'), 'utf8')).match(/^\/\*![\s\S]*?\*\//)?.[0];
if (!banner?.includes('Licensed under the MIT License')) {
  throw new Error('onnxruntime-web bundle no longer carries its MIT license banner; update the notice step');
}
await writeFile(
  join(noticesOut, 'onnxruntime-web-MIT.txt'),
  `ONNX Runtime Web ${ort.version} (onnxruntime-web on npm)
Source: https://github.com/microsoft/onnxruntime/tree/v${ort.version}
Third-party notices for the runtime binary: https://github.com/microsoft/onnxruntime/blob/v${ort.version}/ThirdPartyNotices.txt

${banner}

MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
`,
);

console.log(`copied onnxruntime-web ${ort.version} runtime to public/ort and notices to public/notices`);
