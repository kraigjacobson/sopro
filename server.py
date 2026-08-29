#!/usr/bin/env python3
"""HTTP server for sopro-v2-turbo, wire-compatible with the cosy2 container.

aitts (and anything else already speaking to cosy2) can point at this server
without changing a line: same endpoint, same multipart request, same raw PCM
reply. Sopro's own demo server (`soprotts serve`) speaks a browser-oriented
float32 protocol with a separate reference-upload step; this one is the
one-shot shape a shell client wants.

  POST /inference_zero_shot
        multipart: tts_text (str)            text to speak
                   prompt_wav (file)         reference clip (wav/flac/ogg; 5-20 s)
                   lang       (str, opt)     en | pt | fr | de  — helps ambiguous text
                   steps      (int, opt)     acoustic solver steps (default from model, 2)
                   temperature/top_p/top_k   (opt) sampling overrides
                   seed       (int, opt)     deterministic output
                   prompt_text, stream       accepted and ignored (cosy2 compat)
        -> raw int16 mono PCM @ 24000 Hz  (audio/pcm), exactly like cosy2

  POST /synthesize
        JSON: {"text": ..., "speaker_audio_base64": ..., "lang"?, "steps"?, ...}
        -> {"audio_base64": <wav>, "sample_rate": 24000, "format": "wav"}
        (the RunPod job shape, exposed over HTTP so both transports are testable)

  GET  /health      -> {"status": "ok", "model", "device", "sample_rate", "steps"}
  GET  /docs        FastAPI's OpenAPI UI (aitts pings this for cosy2; works here too)

Reference clips are prepared once and cached in-process by sha256 of the bytes
(SOPRO_REF_CACHE entries, default 16) — the speaker/semantic encoders are the
expensive part of a request, and a voice is reused for every chunk of a reply.

Environment:
  SOPRO_MODEL        HF repo id or local artifact dir (default samuel-vitorino/sopro-v2-turbo)
  SOPRO_DEVICE       auto (default) | cuda | cpu
  SOPRO_INT8         1 -> int8 AR weights (CPU only)
  SOPRO_STEPS        default acoustic steps when a request doesn't pass one
  SOPRO_LANG         default language hint (empty = let the model decide)
  SOPRO_REF_CACHE    prepared-reference LRU size (default 16)
  SOPRO_LOG_PROMPTS  1 -> log the text being spoken (default 0: privacy)
  PORT               listen port for `python server.py` (default 50010)
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import logging
import os
import sys
import threading
import time
import wave
from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from sopro import audio as audio_ops
from sopro.model import Reference, SoproTTS

log = logging.getLogger("sopro.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MODEL_ID = os.environ.get("SOPRO_MODEL", "samuel-vitorino/sopro-v2-turbo")
DEVICE_PREF = os.environ.get("SOPRO_DEVICE", "auto")
INT8 = os.environ.get("SOPRO_INT8", "0") in ("1", "true", "True")
DEFAULT_STEPS = int(os.environ["SOPRO_STEPS"]) if os.environ.get("SOPRO_STEPS") else None
DEFAULT_LANG = os.environ.get("SOPRO_LANG") or None


def _env_num(name: str, cast):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else None


# Server-wide defaults for the other generation levers (a request can still override
# each one). Empty = the model's own config (temperature 0.8, top_p 0.9, top_k 25,
# ref_seconds 10, prompt_tokens 120, style_tokens 160). Set these in compose / the
# RunPod template once a sweep (tools/sweep.py) has found the values you like.
DEFAULT_TEMPERATURE = _env_num("SOPRO_TEMPERATURE", float)
DEFAULT_TOP_P = _env_num("SOPRO_TOP_P", float)
DEFAULT_TOP_K = _env_num("SOPRO_TOP_K", int)
DEFAULT_REF_SECONDS = _env_num("SOPRO_REF_SECONDS", float)
DEFAULT_PROMPT_TOKENS = _env_num("SOPRO_PROMPT_TOKENS", int)
DEFAULT_STYLE_TOKENS = _env_num("SOPRO_STYLE_TOKENS", int)
REF_CACHE_SIZE = int(os.environ.get("SOPRO_REF_CACHE", "16"))
LOG_PROMPTS = os.environ.get("SOPRO_LOG_PROMPTS", "0") in ("1", "true", "True")


def pick_device(pref: str) -> str:
    if pref != "auto":
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# model + reference cache
# ---------------------------------------------------------------------------
tts: Optional[SoproTTS] = None
DEVICE = pick_device(DEVICE_PREF)
if INT8:
    DEVICE = "cpu"
_gen_lock = threading.Lock()           # one synthesis at a time on the accelerator
_ref_lock = threading.Lock()
_refs: "OrderedDict[str, Reference]" = OrderedDict()


def load_model() -> SoproTTS:
    global tts
    if tts is None:
        t0 = time.perf_counter()
        log.info("loading %s on %s%s", MODEL_ID, DEVICE, " (int8)" if INT8 else "")
        tts = SoproTTS.from_pretrained(MODEL_ID, device=DEVICE, quantization="int8" if INT8 else None)
        log.info("model ready in %.1fs (sample_rate=%d)", time.perf_counter() - t0, tts.sample_rate)
    return tts


def decode_audio(data: bytes) -> "tuple[torch.Tensor, int]":
    """Any soundfile-readable container (wav/flac/ogg) -> (channels-first float tensor, sr)."""
    try:
        arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not decode prompt_wav: {e}") from e
    if arr.shape[0] < int(sr * 0.5):
        raise HTTPException(400, "prompt_wav is too short (need at least 0.5 s)")
    return torch.from_numpy(np.ascontiguousarray(arr.T)), int(sr)


def prepare_reference(data: bytes, ref_seconds: Optional[float] = None) -> Reference:
    """Prepared-reference LRU keyed by the clip's bytes (+ the crop length).

    ref_seconds: how much of the clip to keep (sopro crops on a pause; model default
    10 s). More seconds = more of the speaker's style in the prompt, slower prefill."""
    if ref_seconds is None:
        ref_seconds = DEFAULT_REF_SECONDS
    key = hashlib.sha256(data + f"|{ref_seconds}".encode()).hexdigest()
    with _ref_lock:
        ref = _refs.get(key)
        if ref is not None:
            _refs.move_to_end(key)
            return ref
    wav, sr = decode_audio(data)
    model = load_model()
    with _gen_lock:
        ref = model.prepare_reference(ref_audio=wav, sample_rate=sr, seconds=ref_seconds)
    with _ref_lock:
        _refs[key] = ref
        while len(_refs) > REF_CACHE_SIZE:
            _refs.popitem(last=False)
    return ref


def synth_pcm(
    text: str,
    ref: Reference,
    lang: Optional[str] = None,
    steps: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    prompt_tokens: Optional[int] = None,
    style_tokens: Optional[int] = None,
) -> bytes:
    """Synthesize -> raw int16 mono PCM at the model's sample rate (24 kHz).

    prompt_tokens / style_tokens override the model's generation config for THIS call
    (how many of the reference's semantic tokens seed the continuation / the style
    prefix; defaults 120 / 160). They live on the shared config, so the override is
    applied and restored under the generation lock."""
    text = " ".join(str(text).split())
    if not text:
        raise HTTPException(400, "tts_text is required")
    model = load_model()
    if temperature is None:
        temperature = DEFAULT_TEMPERATURE
    if top_p is None:
        top_p = DEFAULT_TOP_P
    if top_k is None:
        top_k = DEFAULT_TOP_K
    if prompt_tokens is None:
        prompt_tokens = DEFAULT_PROMPT_TOKENS
    if style_tokens is None:
        style_tokens = DEFAULT_STYLE_TOKENS
    t0 = time.perf_counter()
    with _gen_lock:
        if seed is not None:
            torch.manual_seed(int(seed))
        gen = model.generation
        saved = (gen.prompt_tokens, gen.style_tokens)
        try:
            if prompt_tokens is not None:
                gen.prompt_tokens = int(prompt_tokens)
            if style_tokens is not None:
                gen.style_tokens = int(style_tokens)
            wav = model.synthesize(
                text,
                ref=ref,
                lang=lang or DEFAULT_LANG,
                steps=steps if steps is not None else DEFAULT_STEPS,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        finally:
            gen.prompt_tokens, gen.style_tokens = saved
    wav = wav.detach().float().cpu().reshape(-1).clamp(-1.0, 1.0)
    pcm = (wav.numpy() * 32767.0).astype("<i2").tobytes()
    dur = len(pcm) / 2 / model.sample_rate
    el = time.perf_counter() - t0
    log.info(
        "synth %d chars -> %.2fs audio in %.2fs (rtf %.2f)%s",
        len(text), dur, el, el / max(dur, 1e-6), f"  text={text!r}" if LOG_PROMPTS else "",
    )
    return pcm


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def warmup() -> None:
    """One tiny synthesis at boot so the first real request doesn't pay for CUDA init."""
    model = load_model()
    sr = model.sample_rate
    t = torch.linspace(0, 2.0, int(sr * 2.0))
    tone = (0.2 * torch.sin(2 * torch.pi * 220 * t) * (1 + 0.3 * torch.sin(2 * torch.pi * 3 * t)))
    try:
        ref = model.prepare_reference(ref_audio=tone, sample_rate=sr)
        with _gen_lock:
            model.synthesize("Warm up.", ref=ref, steps=DEFAULT_STEPS)
        log.info("warm-up done")
    except Exception as e:  # noqa: BLE001
        log.warning("warm-up skipped: %s", e)


# ---------------------------------------------------------------------------
# HTTP app
# ---------------------------------------------------------------------------
app = FastAPI(title="sopro", version="2.0", description=__doc__)


@app.get("/health")
def health() -> Dict[str, Any]:
    model = load_model()
    gen = model.generation
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "sample_rate": model.sample_rate,
        "refs_cached": len(_refs),
        # effective server defaults (env override, else the model's config)
        "steps": DEFAULT_STEPS if DEFAULT_STEPS is not None else int(gen.steps),
        "temperature": DEFAULT_TEMPERATURE if DEFAULT_TEMPERATURE is not None else float(gen.temperature),
        "top_p": DEFAULT_TOP_P if DEFAULT_TOP_P is not None else float(gen.top_p),
        "top_k": DEFAULT_TOP_K if DEFAULT_TOP_K is not None else int(gen.top_k),
        "ref_seconds": DEFAULT_REF_SECONDS if DEFAULT_REF_SECONDS is not None else float(gen.ref_seconds),
        "prompt_tokens": DEFAULT_PROMPT_TOKENS if DEFAULT_PROMPT_TOKENS is not None else int(gen.prompt_tokens),
        "style_tokens": DEFAULT_STYLE_TOKENS if DEFAULT_STYLE_TOKENS is not None else int(gen.style_tokens),
        "lang": DEFAULT_LANG,
    }


def _opt_float(v: Optional[str]) -> Optional[float]:
    return float(v) if v not in (None, "") else None


def _opt_int(v: Optional[str]) -> Optional[int]:
    return int(v) if v not in (None, "") else None


@app.post("/inference_zero_shot")
async def inference_zero_shot(
    tts_text: str = Form(...),
    prompt_wav: UploadFile = File(...),
    lang: Optional[str] = Form(None),
    steps: Optional[str] = Form(None),
    temperature: Optional[str] = Form(None),
    top_p: Optional[str] = Form(None),
    top_k: Optional[str] = Form(None),
    seed: Optional[str] = Form(None),
    ref_seconds: Optional[str] = Form(None),
    prompt_tokens: Optional[str] = Form(None),
    style_tokens: Optional[str] = Form(None),
    prompt_text: Optional[str] = Form(None),   # cosy2 compat; sopro needs no transcript
    stream: Optional[str] = Form(None),        # cosy2 compat; always one-shot here
):
    data = await prompt_wav.read()
    if not data:
        raise HTTPException(400, "prompt_wav is empty")

    def work() -> bytes:
        ref = prepare_reference(data, _opt_float(ref_seconds))
        return synth_pcm(
            tts_text, ref, lang=lang or None, steps=_opt_int(steps),
            temperature=_opt_float(temperature), top_p=_opt_float(top_p),
            top_k=_opt_int(top_k), seed=_opt_int(seed),
            prompt_tokens=_opt_int(prompt_tokens), style_tokens=_opt_int(style_tokens),
        )

    pcm = await run_in_threadpool(work)
    model = load_model()
    return Response(
        content=pcm,
        media_type="audio/pcm",
        headers={"X-Sample-Rate": str(model.sample_rate), "X-Channels": "1", "X-Format": "s16le"},
    )


class SynthRequest(BaseModel):
    text: str
    speaker_audio_base64: str
    lang: Optional[str] = None
    steps: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None
    ref_seconds: Optional[float] = None
    prompt_tokens: Optional[int] = None
    style_tokens: Optional[int] = None


def synthesize_job(body: Dict[str, Any]) -> Dict[str, Any]:
    """The RunPod job body -> RunPod job output. Shared by /synthesize and rp_handler."""
    text = (body.get("text") or "").strip()
    b64 = body.get("speaker_audio_base64")
    if not text:
        return {"error": "text is required"}
    if not b64:
        return {"error": "speaker_audio_base64 is required"}
    try:
        data = base64.b64decode(b64)
    except Exception as e:  # noqa: BLE001
        return {"error": f"speaker_audio_base64 is not valid base64: {e}"}
    try:
        ref = prepare_reference(data, body.get("ref_seconds"))
        pcm = synth_pcm(
            text, ref, lang=body.get("lang") or None, steps=body.get("steps"),
            temperature=body.get("temperature"), top_p=body.get("top_p"),
            top_k=body.get("top_k"), seed=body.get("seed"),
            prompt_tokens=body.get("prompt_tokens"), style_tokens=body.get("style_tokens"),
        )
    except HTTPException as e:
        return {"error": str(e.detail)}
    except Exception as e:  # noqa: BLE001
        log.exception("synthesis failed")
        return {"error": str(e)}
    sr = load_model().sample_rate
    return {
        "audio_base64": base64.b64encode(pcm_to_wav(pcm, sr)).decode("ascii"),
        "sample_rate": sr,
        "format": "wav",
    }


@app.post("/synthesize")
async def synthesize(req: SynthRequest):
    out = await run_in_threadpool(synthesize_job, req.model_dump())
    if "error" in out:
        return JSONResponse(out, status_code=400)
    return out


def main(argv: Optional[list] = None) -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="sopro HTTP server (cosy2-compatible)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "50010")))
    p.add_argument("--no-warmup", action="store_true")
    args = p.parse_args(argv)
    load_model()
    if not args.no_warmup:
        warmup()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main(sys.argv[1:])
