import { readFile, writeFile } from 'node:fs/promises';

const path = new URL('../src/runtime/ort-mobile.mjs', import.meta.url);
const wasmPath = new URL('../src/runtime/ort-mobile.wasm', import.meta.url);
const glue = await readFile(path, 'utf8');
const factory = 'async function ortWasm(moduleArg={}){var moduleRtn;';
const views = 'function da(){var a=J.buffer;e.HEAP8=B=new Int8Array(a);D=new Int16Array(a);e.HEAPU8=C=new Uint8Array(a);new Uint16Array(a);e.HEAP32=E=new Int32Array(a);e.HEAPU32=F=new Uint32Array(a);G=new Float32Array(a);H=new Float64Array(a);I=new BigInt64Array(a);new BigUint64Array(a)}';
const grow = 'try{J.grow(d);da();var f=1;break a}catch(h){}';
if (!glue.includes(factory) || !glue.includes(views) || !glue.includes(grow)) throw new Error('ORT glue did not match the expected mobile build');

const initialPages = 7168;
const wasm = new Uint8Array(await readFile(wasmPath));
const cursor = { offset: 8 };
const readVarUint = () => {
  let value = 0, shift = 0, byte;
  do { byte = wasm[cursor.offset++]; value |= (byte & 0x7f) << shift; shift += 7; } while (byte & 0x80);
  return value >>> 0;
};
const encodeVarUint = (value) => {
  const bytes = [];
  do { let byte = value & 0x7f; value >>>= 7; if (value) byte |= 0x80; bytes.push(byte); } while (value);
  return bytes;
};
let patchedMemory = false;
while (cursor.offset < wasm.length) {
  const section = wasm[cursor.offset++], size = readVarUint(), end = cursor.offset + size;
  if (section === 5) {
    const count = readVarUint();
    if (count !== 1) throw new Error(`expected one WASM memory, found ${count}`);
    const flags = readVarUint();
    if (flags & 0x02) throw new Error('the mobile WASM memory unexpectedly became shared');
    const minimumStart = cursor.offset, oldMinimum = readVarUint(), minimumEnd = cursor.offset;
    const minimum = encodeVarUint(initialPages);
    if (minimum.length !== minimumEnd - minimumStart) throw new Error('new WASM minimum changes the memory section width');
    wasm.set(minimum, minimumStart);
    patchedMemory = oldMinimum !== initialPages;
    break;
  }
  cursor.offset = end;
}
if (patchedMemory) {
  await writeFile(wasmPath, wasm);
  console.log('patched mobile WASM initial memory to 448 MiB');
} else {
  console.log('mobile WASM initial memory is already 448 MiB');
}
