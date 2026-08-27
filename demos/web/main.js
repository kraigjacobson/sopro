import { createRuntime, runtimeUi } from './backend.js';

const $ = (selector) => document.querySelector(selector);
const drop = $('#drop'), fileInput = $('#reference'), speak = $('#speak'), status = $('#status'), progress = $('#progress');
const choose = $('#choose'), record = $('#record'), recordLabel = $('#record-label');
const textInput = $('#text'), language = $('#language'), backend = $('#backend'), backendControl = $('#backend-control');
const loader = $('#model-loader'), modelProgress = $('#model-progress'), modelStatus = $('#model-status'), app = $('#app');
const audio = $('#audio'), player = $('#player'), play = $('#play'), waveTrack = $('#wave-track'), waveform = $('#waveform'), audioTime = $('#audio-time'), download = $('#download');
const streamAudio = new Audio(); streamAudio.playsInline = true;
const MODEL_READY_KEY = 'sopro:model-ready-v1';
let reference = null, tts = null, mode = 'stream', objectUrl = null, running = false;
let waveformPeaks = null, waveformDuration = 0, animationFrame = null, livePlayer = null;
let liveResultReady = false, livePlaybackEnded = false, liveResultUrl = null;
let playbackContext = null, playbackReady = null, playbackDestination = null, playbackNode = null;
let recorder = null, recordingStream = null, recordingChunks = [], recordingError = null;
let defaultTextActive = true;
const DEFAULT_TEXT = {
  '': 'The smallest voice can still fill a room.',
  en: 'The smallest voice can still fill a room.',
  pt: 'A brisa do mar atravessa devagar as ruas antigas de Lisboa.',
  fr: 'Une lumière douce traverse les fenêtres au bord de la mer.',
  de: 'Ein leiser Wind zieht durch die stillen Straßen am Meer.',
};

function storedModel() { try { return localStorage.getItem(MODEL_READY_KEY); } catch { return null; } }
function rememberModel(value) { try { localStorage.setItem(MODEL_READY_KEY, value); } catch {} }
function storedVariant() {
  if (!runtimeUi.selector) return undefined;
  try {
    const value = localStorage.getItem(runtimeUi.selector.storageKey);
    return runtimeUi.selector.options.some(([option]) => option === value) ? value : undefined;
  } catch { return undefined; }
}
function setStatus(message, busy = false) {
  status.textContent = message; progress.classList.toggle('busy', busy);
  if (busy) progress.style.removeProperty('width');
  else progress.style.width = message.includes('ready') || message.includes('restored') ? '100%' : '0';
}
function parameters() {
  const selectedLanguage = language.value;
  return { ...(selectedLanguage ? { language: selectedLanguage } : {}), temperature: Number($('#temperature').value), topP: Number($('#top-p').value), topK: Number($('#top-k').value), seed: Number($('#seed').value) };
}
function formatTime(seconds) { const value = Math.max(0, Math.floor(Number(seconds) || 0)); return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, '0')}`; }

function encodeWav(samples, sampleRate) {
  const bytes = new ArrayBuffer(44 + samples.length * 2), view = new DataView(bytes);
  const text = (offset, value) => { for (let index = 0; index < value.length; index++) view.setUint8(offset + index, value.charCodeAt(index)); };
  text(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); text(8, 'WAVE'); text(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, samples.length * 2, true);
  for (let index = 0; index < samples.length; index++) view.setInt16(44 + index * 2, Math.max(-32768, Math.min(32767, Math.round(samples[index] * 32767))), true);
  return new Blob([bytes], { type: 'audio/wav' });
}

function preparePlayback() {
  if (playbackContext && playbackContext.state !== 'closed') return playbackReady;
  const Context = globalThis.AudioContext || globalThis.webkitAudioContext;
  if (!Context) throw new Error('Web Audio is unavailable in this browser.');
  playbackContext = new Context();
  if (!playbackContext.audioWorklet || !globalThis.AudioWorkletNode) throw new Error('Streaming audio is unavailable in this browser.');
  playbackDestination = playbackContext.createMediaStreamDestination(); streamAudio.srcObject = playbackDestination.stream;
  playbackReady = playbackContext.audioWorklet.addModule(new URL('./output-worklet.js', import.meta.url)).then(() => preparePlaybackNode());
  return playbackReady;
}

function preparePlaybackNode() {
  if (!playbackNode && playbackContext?.state !== 'closed') {
    playbackNode = new AudioWorkletNode(playbackContext, 'sopro-output', { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1] });
    playbackNode.connect(playbackDestination);
  }
}

function activatePlayback() {
  const ready = preparePlayback(), playing = streamAudio.play(), resumed = playbackContext.resume();
  return Promise.all([ready, playing, resumed]);
}

async function closePlayback() {
  const context = playbackContext; playbackContext = null; playbackReady = null;
  playbackNode?.disconnect(); playbackNode = null;
  streamAudio.pause(); streamAudio.srcObject = null; playbackDestination = null;
  if (context && context.state !== 'closed') await context.close();
}

async function stopPlaybackForCapture() {
  const closing = livePlayer?.abort(); livePlayer = null; liveResultReady = false; livePlaybackEnded = false; liveResultUrl = null;
  audio.pause(); audio.loop = false; audio.removeAttribute('src'); audio.load(); cancelAnimationFrame(animationFrame); player.hidden = true;
  play.classList.remove('playing'); play.setAttribute('aria-label', 'Play audio');
  await closing; await closePlayback(); await new Promise((resolve) => setTimeout(resolve, 50));
}

function makePeaks(samples, count = 240) {
  const peaks = new Float32Array(count), stride = Math.max(1, Math.floor(samples.length / count));
  for (let bar = 0; bar < count; bar++) {
    const start = bar * stride, end = Math.min(samples.length, start + stride); let peak = 0;
    for (let index = start; index < end; index++) peak = Math.max(peak, Math.abs(samples[index]));
    peaks[bar] = peak;
  }
  const max = Math.max(...peaks, 1e-6);
  for (let index = 0; index < peaks.length; index++) peaks[index] = Math.sqrt(peaks[index] / max);
  return peaks;
}

function drawWaveform() {
  if (!waveformPeaks?.length || player.hidden) return;
  const width = Math.max(1, Math.floor(waveTrack.getBoundingClientRect().width)), height = 42, ratio = devicePixelRatio || 1;
  if (waveform.width !== width * ratio || waveform.height !== height * ratio) { waveform.width = width * ratio; waveform.height = height * ratio; }
  const context = waveform.getContext('2d'); context.setTransform(ratio, 0, 0, ratio, 0, 0); context.clearRect(0, 0, width, height);
  const bars = Math.max(24, Math.min(waveformPeaks.length, Math.floor(width / 3))), gap = 2, barWidth = Math.max(1, (width - gap * (bars - 1)) / bars);
  const played = waveformDuration ? (livePlayer ? livePlayer.currentTime : audio.currentTime) / waveformDuration : 0;
  for (let index = 0; index < bars; index++) {
    const start = Math.floor(index * waveformPeaks.length / bars), end = Math.max(start + 1, Math.floor((index + 1) * waveformPeaks.length / bars)); let peak = 0;
    for (let sample = start; sample < end; sample++) peak = Math.max(peak, waveformPeaks[sample] ?? 0);
    const barHeight = Math.max(3, peak * (height - 7)), x = index * (barWidth + gap), y = (height - barHeight) / 2;
    context.fillStyle = index / bars <= played ? '#171717' : '#cececb';
    context.fillRect(x, y, barWidth, barHeight);
  }
}

function updatePlayer() {
  audioTime.textContent = `${formatTime(livePlayer ? livePlayer.currentTime : audio.currentTime)} / ${formatTime(waveformDuration)}`;
  drawWaveform();
  if (livePlayer || !audio.paused) animationFrame = requestAnimationFrame(updatePlayer);
}

function startLivePlayer(controller) {
  livePlayer?.abort(); livePlayer = controller; liveResultReady = false; livePlaybackEnded = false; liveResultUrl = null;
  waveformPeaks = null; waveformDuration = 0; download.removeAttribute('href');
  play.disabled = true; waveTrack.disabled = true; play.classList.remove('playing'); play.setAttribute('aria-label', 'Streaming audio');
  audioTime.textContent = '0:00 / 0:00'; player.hidden = true; cancelAnimationFrame(animationFrame);
}

function appendLiveDuration(samples, sampleRate) {
  waveformDuration += samples.length / sampleRate;
}

function finishLivePlayer() {
  livePlaybackEnded = true;
  if (!liveResultReady) return;
  audio.pause(); audio.loop = false; audio.src = liveResultUrl;
  livePlayer = null;
  play.disabled = false; waveTrack.disabled = false; play.classList.remove('playing'); play.setAttribute('aria-label', 'Play audio');
  audio.currentTime = 0; cancelAnimationFrame(animationFrame); updatePlayer();
}

function setLiveResult(samples, url, sampleRate) {
  waveformPeaks = makePeaks(samples); waveformDuration = samples.length / sampleRate;
  liveResultUrl = url; download.href = url; liveResultReady = true;
  if (livePlaybackEnded) finishLivePlayer();
  else { player.hidden = false; cancelAnimationFrame(animationFrame); updatePlayer(); }
}

play.addEventListener('click', () => { if (audio.paused) audio.play(); else audio.pause(); });
waveTrack.addEventListener('click', (event) => { if (!waveformDuration) return; const rect = waveTrack.getBoundingClientRect(); audio.currentTime = Math.max(0, Math.min(waveformDuration, (event.clientX - rect.left) / rect.width * waveformDuration)); updatePlayer(); });
audio.addEventListener('play', () => { if (livePlayer) return; play.classList.add('playing'); play.setAttribute('aria-label', 'Pause audio'); cancelAnimationFrame(animationFrame); updatePlayer(); });
audio.addEventListener('pause', () => { if (livePlayer) return; play.classList.remove('playing'); play.setAttribute('aria-label', 'Play audio'); cancelAnimationFrame(animationFrame); updatePlayer(); });
audio.addEventListener('ended', updatePlayer);
new ResizeObserver(drawWaveform).observe(waveTrack);

async function boot() {
  document.title = runtimeUi.title; $('#runtime-description').textContent = runtimeUi.description; $('#model-loader-title').textContent = runtimeUi.loaderTitle; $('#model-link').href = runtimeUi.modelUrl;
  backendControl.hidden = !runtimeUi.selector;
  if (runtimeUi.selector) {
    backendControl.title = runtimeUi.selector.label; backend.setAttribute('aria-label', runtimeUi.selector.label);
    backend.replaceChildren(...runtimeUi.selector.options.map(([value, label]) => new Option(label, value)));
  }
  modelStatus.textContent = runtimeUi.loaderStatus;
  if (runtimeUi.alwaysShowLoader || !storedModel()) { loader.hidden = false; modelProgress.classList.add('busy'); }
  try {
    tts = await createRuntime({
      variant: storedVariant(),
      onProgress: ({ loaded, total }) => { if (!app.hidden && total) { progress.classList.remove('busy'); progress.style.width = `${Math.round(loaded / total * 100)}%`; } },
    });
    if (runtimeUi.selector) backend.value = tts.variant;
    const modelKey = tts.cacheKey;
    if (modelKey && storedModel() !== modelKey && tts.preload) {
      loader.hidden = false; modelProgress.classList.remove('busy'); modelProgress.style.width = '0'; modelStatus.textContent = '0%';
      await tts.preload(({ loaded, total }) => {
        const percent = total ? Math.min(100, Math.round(loaded / total * 100)) : 0;
        modelProgress.style.width = `${percent}%`; modelStatus.textContent = `${percent}%`;
      });
      rememberModel(modelKey);
    }
    modelProgress.classList.remove('busy'); loader.hidden = true; app.hidden = false;
  } catch (error) {
    console.error(error); loader.hidden = false; modelStatus.textContent = error.message; modelProgress.classList.remove('busy'); modelProgress.style.width = '0';
  }
}

async function setReference(next) {
  reference = null; speak.disabled = true; choose.disabled = true; record.disabled = true; $('#reference-name').textContent = next.name;
  try {
    setStatus('Checking local reference cache…', true);
    reference = await tts.prepareReference(next);
    setStatus(reference.fromCache ? 'Restoring cached voice…' : 'Preparing streaming voice…', true);
    await tts.prepareStreaming(reference, parameters());
    await preparePlayback();
    setStatus(reference.fromCache ? 'Reference restored from this device' : 'Reference ready');
    speak.disabled = false;
  } catch (error) { console.error(error); setStatus(error.message); }
  finally { choose.disabled = false; record.disabled = false; }
}

function finishRecordingUi() {
  record.classList.remove('recording'); drop.classList.remove('recording'); recordLabel.textContent = 'Record';
  record.setAttribute('aria-label', 'Record a voice reference'); record.disabled = false; choose.disabled = false; speak.disabled = !reference;
}

function recordingExtension(type) {
  if (type.includes('mp4') || type.includes('aac')) return 'm4a';
  if (type.includes('ogg')) return 'ogg';
  if (type.includes('wav')) return 'wav';
  return 'webm';
}

async function toggleRecording(event) {
  event.stopPropagation();
  if (recorder?.state === 'recording') {
    recorder.stop(); recordingStream?.getTracks().forEach((track) => track.stop()); recordingStream = null;
    preparePlayback().catch(() => {}); return;
  }
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
    setStatus('Microphone recording is not supported by this browser.'); return;
  }
  record.disabled = true; choose.disabled = true; await stopPlaybackForCapture();
  try {
    recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const candidates = ['audio/mp4', 'audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus'];
    const mimeType = candidates.find((type) => MediaRecorder.isTypeSupported?.(type));
    recorder = mimeType ? new MediaRecorder(recordingStream, { mimeType }) : new MediaRecorder(recordingStream);
    const activeRecorder = recorder;
    recordingChunks = []; recordingError = null;
    activeRecorder.addEventListener('dataavailable', ({ data }) => { if (data.size) recordingChunks.push(data); });
    activeRecorder.addEventListener('error', ({ error }) => { recordingError = error ?? new Error('Microphone recording failed.'); });
    activeRecorder.addEventListener('stop', async () => {
      recordingStream?.getTracks().forEach((track) => track.stop()); recordingStream = null; recorder = null; finishRecordingUi();
      if (recordingError) { console.error(recordingError); setStatus(recordingError.message); return; }
      if (!recordingChunks.some((chunk) => chunk.size)) { setStatus('No microphone audio was captured.'); return; }
      const type = activeRecorder.mimeType || recordingChunks[0].type || 'audio/webm';
      const blob = new Blob(recordingChunks, { type }); recordingChunks = [];
      record.disabled = true; choose.disabled = true; speak.disabled = true; setStatus('Preparing recording…', true);
      await new Promise((resolve) => setTimeout(resolve, 200));
      await setReference(new File([blob], `recorded-reference.${recordingExtension(type)}`, { type }));
    }, { once: true });
    recorder.start(250); record.disabled = false; speak.disabled = true; drop.classList.add('recording'); record.classList.add('recording');
    recordLabel.textContent = 'Stop'; record.setAttribute('aria-label', 'Stop recording'); setStatus('Recording reference, tap Stop when finished', true);
  } catch (error) {
    recordingStream?.getTracks().forEach((track) => track.stop()); recordingStream = null; recorder = null; finishRecordingUi();
    console.error(error); setStatus(error.name === 'NotAllowedError' ? 'Microphone permission was denied.' : error.message);
  }
}

drop.addEventListener('click', () => { preparePlayback().catch(() => {}); fileInput.click(); });
choose.addEventListener('click', (event) => { event.stopPropagation(); preparePlayback().catch(() => {}); fileInput.click(); });
record.addEventListener('click', toggleRecording);
fileInput.addEventListener('change', () => fileInput.files[0] && setReference(fileInput.files[0]));
for (const event of ['dragenter','dragover']) drop.addEventListener(event, (value) => { value.preventDefault(); drop.classList.add('drag'); });
for (const event of ['dragleave','drop']) drop.addEventListener(event, (value) => { value.preventDefault(); drop.classList.remove('drag'); });
drop.addEventListener('drop', (event) => {
  if (recorder?.state !== 'recording' && event.dataTransfer.files[0]) { preparePlayback().catch(() => {}); setReference(event.dataTransfer.files[0]); }
});
document.querySelectorAll('[data-mode]').forEach((button) => button.addEventListener('click', () => { mode = button.dataset.mode; document.querySelectorAll('[data-mode]').forEach((item) => item.classList.toggle('active', item === button)); }));
for (const [input, output] of [['#temperature','#temperature-value'],['#top-p','#top-p-value']]) $(input).addEventListener('input', () => $(output).textContent = Number($(input).value).toFixed(2));
textInput.addEventListener('input', () => { defaultTextActive = false; });
language.addEventListener('change', () => { if (defaultTextActive) textInput.value = DEFAULT_TEXT[language.value] ?? DEFAULT_TEXT['']; });
backend.addEventListener('change', () => { if (!runtimeUi.selector) return; try { localStorage.setItem(runtimeUi.selector.storageKey, backend.value); } catch {} location.reload(); });

class StreamResampler {
  constructor(sourceRate, targetRate) { this.ratio = sourceRate / targetRate; this.phase = 0; this.previous = null; }

  push(input) {
    if (!input.length) return new Float32Array(0);
    const source = new Float32Array(input.length + (this.previous === null ? 0 : 1));
    if (this.previous !== null) { source[0] = this.previous; source.set(input, 1); } else source.set(input);
    const last = source.length - 1;
    if (!last) { this.previous = source[0]; return new Float32Array(0); }
    const length = this.phase <= last ? Math.floor((last - this.phase) / this.ratio) + 1 : 0, output = new Float32Array(length);
    let position = this.phase;
    for (let index = 0; index < length; index++, position += this.ratio) {
      const left = Math.floor(position), mix = position - left;
      output[index] = source[left] + (source[Math.min(last, left + 1)] - source[left]) * mix;
    }
    this.phase = position - last; this.previous = source[last];
    return output;
  }
}

class StreamPlayer {
  constructor(sampleRate, onStart, onBuffer, onEnd) {
    if (!playbackContext || !playbackReady || !playbackNode) throw new Error('Audio playback is not ready.');
    this.context = playbackContext; this.node = playbackNode; playbackNode = null;
    this.sampleRate = sampleRate; this.resampler = new StreamResampler(sampleRate, this.context.sampleRate); this.startTime = null;
    this.previousChunkAt = null; this.productionRtf = null; this.generatedSeconds = 0;
    this.started = false; this.buffering = false; this.suspendedForBuffer = false; this.ended = false; this.aborted = false;
    this.onStart = onStart; this.onBuffer = onBuffer; this.onEnd = onEnd;
    const activated = activatePlayback();
    this.ready = Promise.all([activated, playbackReady]).then(() => {
      if (this.aborted) return;
      this.node.port.onmessage = ({ data }) => {
        if (data.type === 'start' && this.startTime === null) { this.startTime = data.at; this.onStart?.(); }
        else if (data.type === 'end') this._finish();
      };
    });
  }

  get currentTime() {
    if (this.startTime === null) return 0;
    return Math.max(0, Math.min(this.generatedSeconds, this.context.currentTime - this.startTime));
  }

  _beginBuffering() {
    if (this.buffering) return;
    this.buffering = true; this.onBuffer?.();
  }

  async push(samples) {
    const now = performance.now(), audioSeconds = samples.length / this.sampleRate;
    this.generatedSeconds += audioSeconds;
    if (this.previousChunkAt !== null) {
      const observedRtf = ((now - this.previousChunkAt) / 1000) / Math.max(audioSeconds, 1e-6);
      this.productionRtf = this.productionRtf === null ? observedRtf : this.productionRtf * .65 + observedRtf * .35;
    }
    this.previousChunkAt = now; await this.ready;
    if (this.aborted) return;
    if (this.started && this.productionRtf > 1) this._beginBuffering();
    if (this.buffering && !this.suspendedForBuffer) { await this.context.suspend(); this.suspendedForBuffer = true; }
    const output = this.resampler.push(samples);
    if (output.length) this.node.port.postMessage({ type: 'chunk', samples: output }, [output.buffer]);
    this.started = true;
    if (this.buffering && this.productionRtf < .95 && this.generatedSeconds - this.currentTime >= .5) {
      await this.context.resume(); this.buffering = false; this.suspendedForBuffer = false; this.onStart?.();
    }
  }

  async finish() {
    await this.ready;
    if (this.aborted) return;
    this.node.port.postMessage({ type: 'end' });
    if (this.context.state !== 'running') { await this.context.resume(); this.buffering = false; this.suspendedForBuffer = false; this.onStart?.(); }
  }

  _finish() {
    if (this.ended || this.aborted) return;
    this.ended = true; this.node.port.onmessage = null; this.node.disconnect(); preparePlaybackNode(); this.onEnd?.();
  }

  abort() {
    this.aborted = true;
    if (this.node) { this.node.port.onmessage = null; this.node.disconnect(); preparePlaybackNode(); }
    return Promise.resolve();
  }
}

async function run() {
  if (running || !reference) return; running = true; speak.disabled = true; choose.disabled = true; record.disabled = true; livePlayer?.abort(); livePlayer = null; player.hidden = true; audio.pause();
  const started = performance.now(), parts = [], runMode = mode; let firstAudioAt = null, streamPlayer = null, generationFinished = false;
  try {
    streamPlayer = new StreamPlayer(tts.sampleRate, () => { play.classList.add('playing'); if (!generationFinished) setStatus(runMode === 'stream' ? 'Generating…' : 'Playing audio…', true); }, () => { play.classList.remove('playing'); setStatus('Buffering to prevent playback gaps…', true); }, finishLivePlayer);
    setStatus(runMode === 'stream' ? 'Generating…' : 'Rendering locally…', true);
    if (runMode === 'stream') startLivePlayer(streamPlayer);
    if (runMode === 'stream') {
      for await (const chunk of tts.stream(textInput.value, reference, parameters())) { firstAudioAt ??= performance.now(); parts.push(chunk); appendLiveDuration(chunk, tts.sampleRate); await streamPlayer.push(chunk); }
      await streamPlayer.finish();
    } else {
      const samples = await tts.synthesize(textInput.value, reference, parameters()); firstAudioAt = performance.now(); parts.push(samples); startLivePlayer(streamPlayer); appendLiveDuration(samples, tts.sampleRate); await streamPlayer.push(samples); await streamPlayer.finish();
    }
    generationFinished = true;
    const samples = new Float32Array(parts.reduce((sum, part) => sum + part.length, 0)); let offset = 0;
    for (const part of parts) { samples.set(part, offset); offset += part.length; }
    const blob = encodeWav(samples, tts.sampleRate); if (objectUrl) URL.revokeObjectURL(objectUrl); objectUrl = URL.createObjectURL(blob); setLiveResult(samples, objectUrl, tts.sampleRate);
    const elapsed = (performance.now() - started) / 1000, audioSeconds = samples.length / tts.sampleRate, ttfa = ((firstAudioAt ?? performance.now()) - started) / 1000;
    $('#timing').textContent = `TTFA ${ttfa.toFixed(2)}s RTF ${(elapsed / Math.max(audioSeconds, 1e-6)).toFixed(2)}`; setStatus('Done');
  } catch (error) { streamPlayer?.abort(); if (livePlayer === streamPlayer) livePlayer = null; console.error(error); setStatus(error.message); }
  finally { running = false; speak.disabled = !reference; choose.disabled = false; record.disabled = false; }
}

speak.addEventListener('click', run);
textInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); run(); }
});
window.addEventListener('beforeunload', () => recordingStream?.getTracks().forEach((track) => track.stop()));
boot();
