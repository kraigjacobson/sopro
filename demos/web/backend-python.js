export const runtimeUi = {
  title: 'Sopro',
  description: 'Local Python voice synthesis',
  loaderTitle: 'Loading Sopro',
  loaderStatus: 'Downloading weights and loading the Python runtime…',
  modelUrl: 'https://huggingface.co/samuel-vitorino/sopro-v2-turbo',
  alwaysShowLoader: true,
  selector: {
    label: 'Precision',
    storageKey: 'sopro:python-precision-v1',
    options: [['full', 'FP32'], ['int8', 'INT8']],
  },
};

async function errorFrom(response) {
  try { return new Error((await response.json()).error); }
  catch { return new Error(`${response.status} ${response.statusText}`); }
}

async function checked(response) {
  if (!response.ok) throw await errorFrom(response);
  return response;
}

function generationBody(text, reference, options) {
  return JSON.stringify({
    text,
    reference: reference.id,
    language: options.language ?? null,
    temperature: options.temperature,
    top_p: options.topP,
    top_k: options.topK,
    seed: options.seed,
  });
}

async function* floatStream(response) {
  const reader = response.body?.getReader();
  if (!reader) throw new Error('Streaming responses are not supported by this browser.');
  let carry = new Uint8Array(0);
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    const bytes = new Uint8Array(carry.length + value.length);
    bytes.set(carry);
    bytes.set(value, carry.length);
    const byteLength = bytes.length - bytes.length % 4;
    if (byteLength) {
      const samples = new Float32Array(byteLength / 4);
      new Uint8Array(samples.buffer).set(bytes.subarray(0, byteLength));
      yield samples;
    }
    carry = bytes.slice(byteLength);
  }
  if (carry.length) throw new Error('The audio stream ended unexpectedly.');
}

class PythonRuntime {
  constructor(info) { this.sampleRate = info.sample_rate; this.variant = info.precision; this.cacheKey = null; }

  async prepareReference(file) {
    const Offline = globalThis.OfflineAudioContext || globalThis.webkitOfflineAudioContext;
    const Context = globalThis.AudioContext || globalThis.webkitAudioContext;
    if (!Offline && !Context) throw new Error('Web Audio is unavailable in this browser.');
    const context = Offline ? new Offline(1, 1, 48000) : new Context();
    let samples, sampleRate;
    try {
      const decoded = await context.decodeAudioData(await file.arrayBuffer());
      sampleRate = decoded.sampleRate;
      const length = Math.min(decoded.length, sampleRate * 60);
      samples = new Float32Array(length);
      for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
        const source = decoded.getChannelData(channel);
        for (let index = 0; index < length; index++) samples[index] += source[index] / decoded.numberOfChannels;
      }
    } finally { if (context.close) await context.close(); }
    const response = await checked(await fetch('/api/reference', {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream', 'X-Sopro-Sample-Rate': String(sampleRate) },
      body: samples.buffer,
    }));
    return response.json();
  }

  async prepareStreaming() {}

  async *stream(text, reference, options) {
    const response = await checked(await fetch('/api/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: generationBody(text, reference, options),
    }));
    yield* floatStream(response);
  }

  async synthesize(text, reference, options) {
    const response = await checked(await fetch('/api/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: generationBody(text, reference, options),
    }));
    const bytes = new Uint8Array(await response.arrayBuffer());
    const samples = new Float32Array(bytes.byteLength / 4);
    new Uint8Array(samples.buffer).set(bytes);
    return samples;
  }
}

export async function createRuntime(options = {}) {
  const query = options.variant ? `?precision=${encodeURIComponent(options.variant)}` : '';
  const response = await checked(await fetch(`/api/info${query}`));
  return new PythonRuntime(await response.json());
}
