# sopro — sopro-v2-turbo zero-shot TTS as a self-contained image, wire-compatible
# with the cosy2 container (POST /inference_zero_shot -> raw int16 PCM @ 24 kHz) and
# the cosy2 RunPod worker (JSON job -> base64 WAV). The model weights (~760 MB) are
# baked in so the container never touches the network at runtime.
#
# Same base as cosy2 (torch 2.7.0 / CUDA 12.8) so it runs on the same RunPod GPU
# pool and the same local podman CDI setup. The model is 120M params: it also runs
# fine on CPU (SOPRO_DEVICE=cpu), which is what the base image falls back to.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# soundfile wheels bundle libsndfile, but keep the system one as a fallback.
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin torch/torchaudio to the base image's exact builds so `pip install .` can't
# swap them for differently-built wheels (same trick as the cosy2 image).
RUN python -c "import torch, torchaudio; \
    open('/constraints.txt','w').write(f'torch=={torch.__version__}\ntorchaudio=={torchaudio.__version__}\n')" \
    && cat /constraints.txt

# The sopro package (pyproject force-includes demos/web into the wheel, so it must
# be in the build context). Serving deps: FastAPI + multipart for the HTTP path,
# runpod for serverless.
COPY pyproject.toml README.md LICENSE.txt ./
COPY src ./src
COPY demos/web ./demos/web
RUN pip install --constraint /constraints.txt . \
    && pip install --constraint /constraints.txt \
        'fastapi>=0.110' 'uvicorn[standard]>=0.29' python-multipart runpod

# Bake the model into the HF cache so startup is fully offline.
ARG SOPRO_MODEL=samuel-vitorino/sopro-v2-turbo
ENV SOPRO_MODEL=${SOPRO_MODEL}
RUN python -c "from sopro.hub import resolve_artifacts; print(resolve_artifacts('${SOPRO_MODEL}'))"
ENV HF_HUB_OFFLINE=1

COPY server.py rp_handler.py ./

EXPOSE 50010

# Local / Pod: the FastAPI server. RunPod serverless: set the container start
# command to `python -u /app/rp_handler.py`.
CMD ["python", "-u", "/app/server.py", "--port", "50010"]
