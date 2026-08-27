import { copyFile, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const web = resolve(here, '..');
const sourceMjs = resolve(web, 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs');
const sourceWasm = resolve(web, 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm');
const targetMjs = resolve(web, 'src/runtime/ort-mobile-threaded.mjs');
const targetWasm = resolve(web, 'src/runtime/ort-mobile-threaded.wasm');
let code = await readFile(sourceMjs, 'utf8');

function replaceOnce(before, after, label) {
  const first = code.indexOf(before);
  if (first < 0) throw new Error(`Cannot patch ${label}: source pattern was not found.`);
  if (code.indexOf(before, first + before.length) >= 0) throw new Error(`Cannot patch ${label}: source pattern is not unique.`);
  code = code.slice(0, first) + after + code.slice(first + before.length);
}

// The validated mobile workload stays below 448 MiB (measured worst-case floor: 416 MiB). Give the pthread build
// the same fixed shared heap:
// workers share it, so no per-worker model or KV allocation is introduced.
replaceOnce(
  'n||(x=new WebAssembly.Memory({initial:256,maximum:65536,shared:!0}),qa());',
  'n||(x=new WebAssembly.Memory({initial:7168,maximum:7168,shared:!0}),qa());',
  'fixed 448 MiB shared heap',
);

await writeFile(targetMjs, code);
await copyFile(sourceWasm, targetWasm);
console.log(`wrote ${targetMjs}`);
console.log(`wrote ${targetWasm}`);
