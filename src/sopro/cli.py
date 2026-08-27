from __future__ import annotations

import argparse
import sys
import time

import torch

from sopro.model import SoproTTS


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        from sopro.server import serve

        serve(sys.argv[2:])
        return
    p = argparse.ArgumentParser(prog="soprotts", description="Sopro text-to-speech", epilog="Run the local demo with: soprotts serve")
    p.add_argument("text")
    p.add_argument("--ref", required=True, help="reference audio (5-20 s)")
    p.add_argument("--out", default="out.wav")
    p.add_argument("--model", default="samuel-vitorino/sopro-v2-turbo", help="local artifact dir or Hugging Face repo id")
    p.add_argument("--lang", default=None, help="en, pt, fr, de")
    p.add_argument("--device", default="cpu")
    p.add_argument("--int8", action="store_true", help="int8 AR weights (cpu)")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=None, help="cap per generated segment; long text is split into segments, so total length is unbounded")
    p.add_argument("--stream", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(int(args.seed))
    tts = SoproTTS.from_pretrained(args.model, device=args.device, quantization="int8" if args.int8 else None)
    ref = tts.prepare_reference(ref_audio_path=args.ref, stream=args.stream)
    kwargs = dict(lang=args.lang, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, steps=args.steps, max_seconds=args.max_seconds)
    live = sys.stderr.isatty()

    def show(line: str) -> None:
        if live:
            print(f"\r\x1b[K  {line}", end="", file=sys.stderr, flush=True)

    def clear() -> None:
        if live:
            print("\r\x1b[K", end="", file=sys.stderr, flush=True)

    t0 = time.perf_counter()
    if args.stream:
        chunks = []
        first = None
        emitted = 0
        for chunk in tts.stream(args.text, ref=ref, **kwargs):
            now = time.perf_counter() - t0
            if first is None:
                first = now
            chunks.append(chunk.cpu())
            emitted += int(chunk.shape[-1])
            audio_s = emitted / tts.sample_rate
            show(f"{audio_s:.1f}s audio · {audio_s / max(now, 1e-6):.1f}x realtime")
        clear()
        wav = torch.cat(chunks, dim=-1)
        total = time.perf_counter() - t0
        audio_s = wav.shape[-1] / tts.sample_rate
        print(f"ttfa {first:.3f}s rtf {total / max(audio_s, 1e-6):.2f} audio {audio_s:.2f}s total {total:.3f}s")
    else:
        seconds_per_token = tts.semantic_encoder.token_samples / tts.sample_rate

        def on_tokens(n: int) -> None:
            elapsed = time.perf_counter() - t0
            show(f"{n} tokens · {n / max(elapsed, 1e-6):.1f} tok/s · {n * seconds_per_token:.1f}s audio")

        wav = tts.synthesize(args.text, ref=ref, on_tokens=on_tokens if live else None, **kwargs).cpu()
        clear()
        total = time.perf_counter() - t0
        audio_s = wav.shape[-1] / tts.sample_rate
        print(f"rtf {total / max(audio_s, 1e-6):.2f} audio {audio_s:.2f}s total {total:.3f}s")
    tts.save_wav(args.out, wav)


if __name__ == "__main__":
    main()
