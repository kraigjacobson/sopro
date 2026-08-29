#!/usr/bin/env python3
"""RunPod serverless entrypoint for sopro — same job shape as the cosy2 worker.

Job input (job["input"]):
  text                  (str, required)  text to synthesize
  speaker_audio_base64  (str, required)  base64 of the reference voice clip (wav/flac/ogg)
  lang                  (str, optional)  en | pt | fr | de
  steps                 (int, optional)  acoustic solver steps (default 2)
  temperature / top_p / top_k / seed     (optional) sampling overrides
  prompt_text, stream, speaker_filename  accepted and ignored (cosy2 compat)

Returns:
  {"audio_base64": <base64 wav, int16 mono 24 kHz>, "sample_rate": 24000, "format": "wav"}
  or {"error": "..."} on failure.

Run modes (auto-detected, like cosy2's rp_handler):
  - RunPod serverless (RUNPOD_* env present): runpod.serverless.start(...)
  - otherwise: serves the FastAPI app (uvicorn) on $PORT, so `python rp_handler.py`
    behaves exactly like `python server.py`.
"""
from __future__ import annotations

import asyncio
import os

import server

print(f"[rp_handler] loading {server.MODEL_ID} on {server.DEVICE}", flush=True)
server.load_model()
server.warmup()
print("[rp_handler] model loaded", flush=True)


def handler_sync(job):
    return server.synthesize_job(job.get("input") or {})


async def handler(job):
    # Off-load the blocking synthesis so RunPod's event loop keeps reporting status.
    return await asyncio.to_thread(handler_sync, job)


def adjust_concurrency(current_concurrency):
    """Jobs one worker accepts at once. Synthesis is serialized on a lock inside
    server.py, so >1 just lets the next job's reference decode/upload overlap the
    current job's GPU time. Default 4; tune with RUNPOD_CONCURRENCY."""
    try:
        return int(os.environ.get("RUNPOD_CONCURRENCY", "4"))
    except (ValueError, TypeError):
        return 4


def is_runpod_environment():
    return any(
        os.environ.get(v)
        for v in ("RUNPOD_POD_ID", "RUNPOD_API_KEY", "RUNPOD_WEBHOOK_GET_URL", "RUNPOD_WEBHOOK_POST_URL")
    )


if __name__ == "__main__":
    if is_runpod_environment():
        import runpod

        print("[rp_handler] starting RunPod serverless handler", flush=True)
        runpod.serverless.start({"handler": handler, "concurrency_modifier": adjust_concurrency})
    else:
        import uvicorn

        port = int(os.environ.get("PORT", "50010"))
        print(f"[rp_handler] no RunPod env detected; serving FastAPI on :{port}", flush=True)
        uvicorn.run(server.app, host="0.0.0.0", port=port)
