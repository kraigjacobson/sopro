# Sopro ONNX Web

Browser runtime and exporter for Sopro. It uses WebGPU on supported desktop
browsers and WASM/SIMD on mobile or when WebGPU is unavailable. Synthesis runs
locally in the browser.

## Export

From the repository root, install the export dependencies:

```bash
python -m pip install -r web/requirements-export.txt
```

Then export the model:

```bash
python web/export.py \
  --artifacts /path/to/sopro-artifacts \
  --output /path/to/web-model
```

The default export contains the desktop WebGPU and mobile/fallback WASM model
profiles. Validate the result with:

```bash
python web/validate.py \
  --artifacts /path/to/sopro-artifacts \
  --web /path/to/web-model
```

## Use

```bash
npm install @soprotts/onnx-web
```

```js
import { SoproTTS } from '@soprotts/onnx-web';

const tts = await SoproTTS.create();
const reference = await tts.prepareReference(file);
await tts.prepareStreaming(reference);

// Stream audio as it is generated.
for await (const audio of tts.stream('Hello.', reference)) {
  // Float32Array, 24 kHz mono
}

// Generate the complete audio offline.
const audio = await tts.synthesize('Hello.', reference);
// Float32Array, 24 kHz mono

await tts.dispose();
```

Language is detected by the model when omitted or set to `"auto"`. Pass `en`,
`pt`, `fr`, or `de` only when you want to force a language tag.

Model assets are loaded lazily and cached by export revision. Mobile relies on
the browser HTTP cache instead of keeping a second copy in CacheStorage.

Serve these headers to enable multithreaded WASM:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without cross-origin isolation, the WASM runtime still works with SIMD on one
thread.
