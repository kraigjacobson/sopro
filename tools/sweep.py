#!/usr/bin/env python3
"""Parameter sweep for the sopro server — one line of text, one reference voice,
a grid of generation settings -> a folder of WAVs + an HTML gallery to A/B by ear,
with an objective intelligibility score (whisper WER) next to each clip.

Stdlib only (runs with the system python3; the model stays in the container).

    tools/sweep.py --ref /path/reference.wav [--text "..."] [--out sweeps/NAME]
                   [--url http://127.0.0.1:50010] [--whisper http://127.0.0.1:8007/transcribe_upload]
                   [--seed 7] [--only steps,temp]      # run a subset of the grid groups
                   [--grid grid.json]                  # your own list of {"name":..., params...}

Levers the server exposes (all optional form fields on /inference_zero_shot):
    steps         acoustic flow-matching solver steps (model default 2; 8-32 = cleaner, slower)
    temperature   AR sampling temperature (0.8) — lower = flatter/safer, higher = livelier/riskier
    top_p, top_k  nucleus / top-k on the AR sampler (0.9 / 25)
    ref_seconds   how much of the reference clip to keep, cropped on a pause (10.0)
    prompt_tokens semantic tokens of the reference that seed the continuation (120)
    style_tokens  semantic tokens used as the style prefix (160)
    lang          en | pt | fr | de hint
    seed          fixed here so every variant differs ONLY by its parameters
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import io
import json
import mimetypes
import os
import re
import sys
import time
import urllib.request
import uuid
import wave
from pathlib import Path

DEFAULT_TEXT = ("Okay so the good news is I found it. The bad news is it has been like that "
                "since March and nobody noticed, including me, just now, twice.")

# name -> params. Groups (the prefix before the dash) are what --only filters on.
GRID = [
    {"name": "steps-2 (model default)", "steps": 2},
    {"name": "steps-4", "steps": 4},
    {"name": "steps-8", "steps": 8},
    {"name": "steps-16", "steps": 16},
    {"name": "steps-32", "steps": 32},
    {"name": "temp-0.6 @8", "steps": 8, "temperature": 0.6},
    {"name": "temp-1.0 @8", "steps": 8, "temperature": 1.0},
    {"name": "topk-50 topp-0.95 @8", "steps": 8, "top_k": 50, "top_p": 0.95},
    {"name": "ref-6s @8", "steps": 8, "ref_seconds": 6},
    {"name": "ref-15s @8", "steps": 8, "ref_seconds": 15},
    {"name": "ref-20s @8", "steps": 8, "ref_seconds": 20},
    {"name": "prompt-0 @8 (no continuation prompt)", "steps": 8, "prompt_tokens": 0},
    {"name": "prompt-240 @8", "steps": 8, "prompt_tokens": 240},
    {"name": "style-320 @8", "steps": 8, "style_tokens": 320},
]
PARAM_KEYS = ("steps", "temperature", "top_p", "top_k", "ref_seconds", "prompt_tokens", "style_tokens", "lang")


def multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    boundary = "----sopro" + uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        if v is None:
            continue
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    for k, (fname, data) in files.items():
        ctype = mimetypes.guess_type(fname)[0] or "application/octet-stream"
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fname}\"\r\n"
                  f"Content-Type: {ctype}\r\n\r\n".encode())
        out.write(data)
        out.write(b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def post(url: str, fields: dict, files: dict, timeout: float = 300) -> tuple[int, bytes, dict]:
    body, ctype = multipart(fields, files)
    req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), {}


def norm_words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()


def wer(ref: str, hyp: str) -> float:
    r, h = norm_words(ref), norm_words(hyp)
    if not r:
        return 0.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def transcribe(whisper_url: str, wav_path: Path) -> str:
    if not whisper_url:
        return ""
    code, body, _ = post(whisper_url, {}, {"audio": (wav_path.name, wav_path.read_bytes())}, timeout=120)
    if code != 200:
        return ""
    try:
        return json.loads(body).get("text", "")
    except Exception:  # noqa: BLE001
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="reference voice clip")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--out", default=None, help="output dir (default sweeps/<timestamp>)")
    ap.add_argument("--url", default="http://127.0.0.1:50010")
    ap.add_argument("--whisper", default="http://127.0.0.1:8007/transcribe_upload", help="'' to skip scoring")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", default="", help="comma list of grid group prefixes, e.g. steps,ref")
    ap.add_argument("--grid", default=None, help="JSON file: list of {name, <params>} to run instead")
    ap.add_argument("--refs", nargs="*", default=None,
                    help="REFERENCE BAKEOFF: one clip per file (glob ok), all at the same params (--params)")
    ap.add_argument("--params", default="{}", help='JSON of fixed params for --refs, e.g. \'{"steps":8,"temperature":0.6}\'')
    args = ap.parse_args()

    ref_path = Path(args.ref)
    ref_bytes = ref_path.read_bytes()
    if args.refs is not None:
        # Reference bakeoff: the grid is the list of clips; every row uses --params.
        fixed = json.loads(args.params)
        import glob as _glob
        files = sorted({Path(p) for pat in args.refs for p in (_glob.glob(pat) if any(c in pat for c in "*?[") else [pat])},
                       key=lambda p: (len(p.name), p.name))   # reference_2 before reference_10
        grid = [{"name": f"ref {Path(p).name}", "_ref": str(p), **fixed} for p in files]
    else:
        grid = json.loads(Path(args.grid).read_text()) if args.grid else GRID
    if args.only:
        wanted = [w.strip() for w in args.only.split(",") if w.strip()]
        grid = [g for g in grid if any(g["name"].startswith(w) for w in wanted)]
    out = Path(args.out or f"sweeps/{dt.datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)

    # sample rate from /health (24000 for sopro-v2-turbo)
    try:
        with urllib.request.urlopen(f"{args.url}/health", timeout=10) as r:
            sr = int(json.load(r)["sample_rate"])
    except Exception as e:  # noqa: BLE001
        sys.exit(f"server not reachable at {args.url}: {e}")

    rows = []
    print(f"sweep -> {out.resolve()}\nref={ref_path}  seed={args.seed}\ntext={args.text!r}\n")
    for i, g in enumerate(grid, 1):
        params = {k: g[k] for k in PARAM_KEYS if k in g}
        slug = re.sub(r"[^a-z0-9]+", "-", g["name"].lower()).strip("-")
        wav_path = out / f"{i:02d}-{slug}.wav"
        fields = {"tts_text": args.text, "seed": args.seed, **params}
        rp, rb = (Path(g["_ref"]), Path(g["_ref"]).read_bytes()) if g.get("_ref") else (ref_path, ref_bytes)
        t0 = time.perf_counter()
        code, body, hdr = post(f"{args.url}/inference_zero_shot", fields, {"prompt_wav": (rp.name, rb)})
        el = time.perf_counter() - t0
        if code != 200 or not body:
            print(f"{i:02d} {g['name']:<42} FAILED {code}: {body[:120]!r}")
            rows.append({"n": i, "name": g["name"], "params": params, "error": f"{code} {body[:200]!r}"})
            continue
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(body)
        dur = len(body) / 2 / sr
        hyp = transcribe(args.whisper, wav_path)
        score = wer(args.text, hyp) if hyp else None
        rows.append({"n": i, "name": g["name"], "params": params, "file": wav_path.name, "audio_s": round(dur, 2),
                     "synth_s": round(el, 2), "rtf": round(el / max(dur, 1e-6), 3),
                     "wer": None if score is None else round(score, 3), "transcript": hyp})
        print(f"{i:02d} {g['name']:<42} {dur:5.2f}s audio  {el:5.2f}s synth  rtf {el/max(dur,1e-6):.2f}"
              f"  WER {'-' if score is None else f'{score*100:4.0f}%'}  {wav_path.resolve()}")

    (out / "results.json").write_text(json.dumps({"text": args.text, "ref": str(ref_path), "seed": args.seed, "rows": rows}, indent=1))

    # Gallery: one row per variant with an inline player, so A/B is a click each.
    def cell(v):
        return html.escape("" if v is None else str(v))
    trs = []
    for r in rows:
        if "error" in r:
            trs.append(f"<tr><td>{r['n']:02d}</td><td>{cell(r['name'])}</td><td colspan=5 class=err>{cell(r['error'])}</td></tr>")
            continue
        p = ", ".join(f"{k}={v}" for k, v in r["params"].items()) or "defaults"
        w = "" if r["wer"] is None else f"{r['wer']*100:.0f}%"
        trs.append(f"<tr><td>{r['n']:02d}</td><td><b>{cell(r['name'])}</b><br><small>{cell(p)}</small></td>"
                   f"<td><audio controls preload=none src=\"{cell(r['file'])}\"></audio></td>"
                   f"<td>{r['audio_s']}s</td><td>{r['synth_s']}s<br><small>rtf {r['rtf']}</small></td>"
                   f"<td>{w}</td><td><small>{cell(r['transcript'])}</small></td></tr>")
    page = f"""<!doctype html><meta charset=utf-8><title>sopro sweep {html.escape(out.name)}</title>
<style>body{{font:14px system-ui;margin:2rem;background:#111;color:#eee}}table{{border-collapse:collapse}}
td,th{{border:1px solid #333;padding:.4rem .6rem;vertical-align:top}}th{{background:#222}}.err{{color:#f66}}
small{{color:#aaa}}audio{{width:280px}}</style>
<h2>sopro sweep — {html.escape(out.name)}</h2>
<p>ref: <code>{html.escape(str(ref_path))}</code> &nbsp; seed {args.seed}<br>text: <i>{html.escape(args.text)}</i></p>
<table><tr><th>#</th><th>variant</th><th>listen</th><th>audio</th><th>synth</th><th>WER</th><th>whisper heard</th></tr>
{''.join(trs)}</table>"""
    (out / "index.html").write_text(page)
    print(f"\ngallery: file://{(out / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
