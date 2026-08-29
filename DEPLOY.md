# Deploying sopro for aitts (local container + RunPod serverless)

This fork adds a **cosy2-compatible server** on top of upstream sopro so it drops
straight into the aitts voice stack next to CosyVoice2:

| file | what |
|---|---|
| `server.py` | FastAPI server: `POST /inference_zero_shot` (multipart → raw int16 PCM @ 24 kHz) and `POST /synthesize` (JSON → base64 WAV), `GET /health` |
| `rp_handler.py` | RunPod serverless handler — same job shape as the cosy2 worker |
| `Dockerfile` | torch 2.7.0 / CUDA 12.8 image with the model weights baked in (offline at runtime) |
| `docker-compose.yml` | the local `sopro` container on host port **50010** (GPU via CDI) |
| `.github/workflows/docker-build-push.yml` | builds + pushes the image on a `v*` tag or by hand |

Upstream is kept as the `upstream` remote; everything above sits beside it, so
`git fetch upstream && git merge upstream/main` stays painless.

## Local

```sh
podman compose up -d            # builds ghcr.io/kraigjacobson/sopro:v0.1.0 the first time
curl -s http://127.0.0.1:50010/health
curl -X POST http://127.0.0.1:50010/inference_zero_shot \
     -F 'tts_text=hello from sopro' -F 'prompt_wav=@reference.wav' -o out.pcm
aplay -f S16_LE -c 1 -r 24000 out.pcm
```

Measured on the RTX 3090: model load 2.2 s, RTF ≈ 0.07 (8 s of speech in 0.6 s),
~1 GB VRAM. `SOPRO_DEVICE=cpu` works too (slower, still real-time on a desktop CPU).

Knobs (compose env): `SOPRO_STEPS` (2 default; 8–32 for harder references),
`SOPRO_LANG` (en|pt|fr|de hint), `SOPRO_LOG_PROMPTS` (1 = log spoken text).

## Image → registry

Tag and push; the workflow builds on a GitHub runner and pushes to
`ghcr.io/kraigjacobson/sopro:<tag>` (and to `krjacobson1/sopro` on Docker Hub when
the `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` secrets are set — same pair as cosy2):

```sh
git tag v0.1.1 && git push origin v0.1.1
# or: Actions tab → "Build and push sopro image" → Run workflow → version
```

Local build of the same image: `podman build -t ghcr.io/kraigjacobson/sopro:v0.1.1 .`

## RunPod serverless

Two endpoints exist (created 2026-08-29), both **0 min workers / 1 max** — the
account has a 10-worker quota shared across every endpoint, which is what capped
them. Each template runs `python -u /app/rp_handler.py` with
`RUNPOD_CONCURRENCY=4`, `SOPRO_STEPS=2`, 5 GB container disk, flashboot on,
GPU pool = the cheap 16–24 GB cards (RTX 2000 Ada … 4090).

| name | endpoint id | template id |
|---|---|---|
| `sopro` (prod) | `jj3uk9hb8x4ly1` | `vm91emqhdn` |
| `sopro stage` | `ivzzx8q0vxv3tb` | `8wsical8lc` |

To roll a new image: build/push a tag, then point the template at it (REST):

```sh
KEY=$(grep ^RUNPOD_API_KEY= /var/mnt/ssd/repos/tts/.env | cut -d= -f2-)
curl -X PATCH https://rest.runpod.io/v1/templates/8wsical8lc \
     -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
     -d '{"imageName":"ghcr.io/kraigjacobson/sopro:v0.1.1"}'
```

Job shape (identical to cosy2's worker):

```json
{"input": {"text": "...", "speaker_audio_base64": "<wav>", "lang": "en", "steps": 2}}
→ {"output": {"audio_base64": "<wav, int16 mono 24 kHz>", "sample_rate": 24000, "format": "wav"}}
```

## aitts

```sh
aitts use sopro local             # local container (auto-started if it's stopped)
aitts use sopro runpod stage      # RunPod stage endpoint
aitts use sopro runpod prod
aitts use cosy2 runpod prod       # back to CosyVoice2
aitts status                      # shows both models' backend/endpoint + containers
```

`aitts backend` / `aitts endpoint` take an optional trailing model name to set the
other model without switching to it (`aitts endpoint stage sopro`).
