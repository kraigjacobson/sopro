function detachBuffer(buffer) {
  if (!(buffer instanceof ArrayBuffer) || !buffer.byteLength) return;
  try {
    if (buffer.__soproShrinkable && typeof buffer.resize === 'function') buffer.resize(0);
    else if (typeof buffer.transfer === 'function') buffer.transfer(0);
    else if (typeof structuredClone === 'function') structuredClone(buffer, { transfer: [buffer] });
  } catch {}
}

function byteArray(length) {
  try {
    const buffer = new ArrayBuffer(length, { maxByteLength: length });
    if (buffer.resizable && typeof buffer.resize === 'function') {
      Object.defineProperty(buffer, '__soproShrinkable', { value: true });
      return new Uint8Array(buffer);
    }
  } catch {}
  return new Uint8Array(length);
}

export class AssetCache {
  constructor(name = 'sopro-onnx-v1', enabled = true, headers = null) { this.name = name; this.enabled = enabled && 'caches' in globalThis; this.headers = headers; this.memory = new Map(); }
  release(url) { const key = String(url); this.memory.delete(key); this.memory.delete(`bytes:${key}`); }
  clearMemory() { this.memory.clear(); }
  async blob(url, onProgress = null) {
    const key = String(url);
    if (this.memory.has(key)) return this.memory.get(key);
    const promise = this.#load(key, onProgress); this.memory.set(key, promise);
    try { return await promise; } catch (error) { this.memory.delete(key); throw error; }
  }
  async bytes(url, onProgress = null) {
    const urlKey = String(url), key = `bytes:${urlKey}`;
    if (this.memory.has(key)) return this.memory.get(key);
    const promise = this.#loadBytes(urlKey, onProgress); this.memory.set(key, promise);
    try { return await promise; } catch (error) { this.memory.delete(key); throw error; }
  }
  async prefetch(url, onProgress = null) {
    const key = String(url);
    const cache = this.enabled ? await caches.open(this.name) : null;
    const cached = cache ? await cache.match(key) : null;
    if (cached) {
      const total = Number(cached.headers.get('content-length') || 0);
      if (onProgress) onProgress({ url: key, loaded: total, total });
      return total;
    }
    const response = await fetch(key, { cache: 'force-cache', ...(this.headers ? { headers: this.headers } : {}) });
    if (!response.ok) throw new Error(`Failed to fetch ${key}: ${response.status}`);
    const cacheWrite = cache ? cache.put(key, response.clone()).catch(() => {}) : null;
    const total = Number(response.headers.get('content-length') || 0);
    let loaded = 0;
    if (response.body) {
      const reader = response.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        loaded += value.byteLength;
        if (onProgress) onProgress({ url: key, loaded, total });
      }
    } else {
      loaded = (await response.blob()).size;
      if (onProgress) onProgress({ url: key, loaded, total: total || loaded });
    }
    if (cacheWrite) await cacheWrite;
    return loaded;
  }
  async #load(url, onProgress) {
    return new Blob([await this.#loadBytes(url, onProgress)]);
  }
  async #loadBytes(url, onProgress) {
    const cache = this.enabled ? await caches.open(this.name) : null;
    const cached = cache ? await cache.match(url) : null;
    if (cached) return new Uint8Array(await cached.arrayBuffer());
    const fetched = await fetch(url, { cache: 'force-cache', ...(this.headers ? { headers: this.headers } : {}) });
    if (!fetched.ok) throw new Error(`Failed to fetch ${url}: ${fetched.status}`);
    const cacheWrite = cache ? cache.put(url, fetched.clone()).catch(() => {}) : null;
    const total = Number(fetched.headers.get('content-length') || 0);
    let bytes, loaded = 0;
    if (fetched.body) {
      bytes = byteArray(total);
      const reader = fetched.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (loaded + value.byteLength > bytes.byteLength) {
          const grown = byteArray(Math.max(loaded + value.byteLength, bytes.byteLength * 2, 1 << 20));
          grown.set(bytes.subarray(0, loaded));
          detachBuffer(bytes.buffer);
          bytes = grown;
        }
        bytes.set(value, loaded); loaded += value.byteLength;
        if (onProgress) onProgress({ url, loaded, total });
      }
      if (loaded !== bytes.byteLength) {
        if (bytes.buffer.__soproShrinkable) { bytes.buffer.resize(loaded); bytes = new Uint8Array(bytes.buffer); }
        else bytes = new Uint8Array(bytes.buffer, 0, loaded);
      }
    } else {
      bytes = new Uint8Array(await fetched.arrayBuffer());
      loaded = bytes.byteLength;
      if (onProgress) onProgress({ url, loaded, total: total || loaded });
    }
    if (cacheWrite) await cacheWrite;
    return bytes;
  }
  async json(url, { revalidate = false } = {}) {
    const key = String(url);
    if (revalidate) {
      this.memory.delete(key);
      const response = await fetch(key, { cache: 'no-cache', ...(this.headers ? { headers: this.headers } : {}) });
      if (!response.ok) throw new Error(`Failed to fetch ${key}: ${response.status}`);
      const value = await response.clone().json();
      if (this.enabled) await (await caches.open(this.name)).put(key, response);
      return value;
    }
    try { return JSON.parse(await (await this.blob(key)).text()); }
    catch (firstError) {
      this.memory.delete(key);
      const cache = this.enabled ? await caches.open(this.name) : null;
      if (cache) await cache.delete(key);
      const response = await fetch(key, { cache: 'reload', ...(this.headers ? { headers: this.headers } : {}) });
      if (!response.ok) throw new Error(`Failed to fetch ${key}: ${response.status}`);
      try {
        const value = await response.clone().json();
        if (cache) await cache.put(key, response);
        return value;
      } catch { throw firstError; }
    }
  }
}

function openDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('sopro-reference-v1', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('references');
    request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
  });
}
export async function getCachedReference(key) {
  if (!('indexedDB' in globalThis)) return null;
  const db = await openDb();
  return new Promise((resolve, reject) => { const request = db.transaction('references').objectStore('references').get(key); request.onsuccess = () => resolve(request.result ?? null); request.onerror = () => reject(request.error); });
}
export async function putCachedReference(key, value) {
  if (!('indexedDB' in globalThis)) return;
  const db = await openDb();
  await new Promise((resolve, reject) => { const request = db.transaction('references', 'readwrite').objectStore('references').put(value, key); request.onsuccess = resolve; request.onerror = () => reject(request.error); });
}
