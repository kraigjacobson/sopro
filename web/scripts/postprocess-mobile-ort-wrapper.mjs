import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const web = resolve(here, '..');
const source = resolve(web, 'node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs');
const target = resolve(web, 'src/runtime/ort-mobile-loader.mjs');
let code = await readFile(source, 'utf8');

function replaceOnce(before, after, label) {
  const first = code.indexOf(before);
  if (first < 0) throw new Error(`Cannot patch ${label}: source pattern was not found.`);
  if (code.indexOf(before, first + before.length) >= 0) throw new Error(`Cannot patch ${label}: source pattern is not unique.`);
  code = code.slice(0, first) + after + code.slice(first + before.length);
}

// Let a CPU-pinned Tensor own the native ORT output handle. The upstream
// constructor deliberately has no disposer for cpu-pinned buffers, so normal
// Tensor.dispose() cannot release a zero-copy WASM output.
replaceOnce(
  'this.cpuData=t.data;break}case"texture"',
  'this.cpuData=t.data,this.disposer=t.dispose;break}case"texture"',
  'cpu-pinned tensor disposer',
);

// Preserve cpu-pinned location when a zero-copy output becomes the next KV or
// acoustic-state input. The data is already inside the active WASM heap.
replaceOnce(
  'case"cpu":return[n.type,n.dims,n.data,"cpu"];case"gpu-buffer"',
  'case"cpu":return[n.type,n.dims,n.data,"cpu"];case"cpu-pinned":{let t=n.data;if(t.buffer!==z().HEAPU8.buffer&&n.__soproRefreshWasmView)t=n.__soproRefreshWasmView();return[n.type,n.dims,t,"cpu-pinned"]}case"gpu-buffer"',
  'cpu-pinned input encoding',
);
replaceOnce(
  'case"cpu":return new de(n[0],n[2],n[1]);case"gpu-buffer"',
  'case"cpu":return new de(n[0],n[2],n[1]);case"cpu-pinned":{let{data:t,dispose:a}=n[2],u=t.byteOffset,o=t.length,d=t.constructor,c=new de({location:"cpu-pinned",type:n[0],dims:n[1],data:t,dispose:a});return c.__soproRefreshWasmView=()=>{let l=new d(z().HEAPU8.buffer,u,o);return c.cpuData=l,l},c}case"gpu-buffer"',
  'cpu-pinned output decoding',
);

replaceOnce(
  'if(b==="gpu-buffer"){let v=n[2].gpuBuffer;',
  'if(b==="cpu-pinned"){let v=n[2];if(v.buffer!==l.HEAPU8.buffer)throw new Error("cpu-pinned tensor is not backed by the active WASM heap");I=v.byteLength,T=v.byteOffset}else if(b==="gpu-buffer"){let v=n[2].gpuBuffer;',
  'zero-copy cpu-pinned input',
);

// Upstream allocates a fresh JS TypedArray and copies every CPU output out of
// WASM. Full KV outputs make that several GiB of short-lived ArrayBuffers per
// generation on this model. Keep a typed view over the native output instead;
// Tensor.dispose() releases its OrtValue deterministically after last use.
replaceOnce(
  'else{let L=io(j),V=new L(ue);new Uint8Array(V.buffer,V.byteOffset,V.byteLength).set(c.HEAPU8.subarray(q,q+V.byteLength)),re.push([j,te,V,"cpu"])}',
  'else{let L=io(j),V=new L(c.HEAPU8.buffer,Number(q),ue);he=!0,re.push([j,te,{data:V,dispose:()=>{c._OrtReleaseTensor(K)!==0&&G("Can\'t release tensor.")}},"cpu-pinned"])}',
  'zero-copy WASM output',
);

code = code.split('ort-wasm-simd-threaded.wasm').join('ort-mobile.wasm');
code = code.split('ort.wasm.bundle.min.mjs').join('ort-mobile-loader.mjs');

await writeFile(target, code);
console.log(`wrote ${target}`);
