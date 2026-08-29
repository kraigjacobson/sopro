#!/usr/bin/env bash
# DNSMOS-score a sweep folder using ursula's clip-builder image (has speechmos).
#   tools/dnsmos_score.sh sweeps/<name>
set -euo pipefail
dir="$(realpath "${1:?usage: $0 <sweep dir>}")"
here="$(cd "$(dirname "$0")" && pwd)"
exec podman run --rm --security-opt label=disable \
  -v "$dir":/sweep -v "$here":/tools:ro -v "$HOME/.cache":/root/.cache \
  --entrypoint python "${CLIP_BUILDER_IMAGE:-localhost/clip-builder:latest}" /tools/dnsmos_score.py /sweep
