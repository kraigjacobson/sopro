from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * rms * self.weight.float()).to(dtype=x.dtype)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float = 0.01) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((int(dim),), float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, mult: float) -> None:
        super().__init__()
        hidden = max(1, int(round(float(mult) * int(dim))))
        self.up_proj = nn.Linear(int(dim), hidden, bias=False)
        self.gate_proj = nn.Linear(int(dim), hidden, bias=False)
        self.down_proj = nn.Linear(hidden, int(dim), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def rotary_cos_sin(positions: torch.Tensor, dim: int, dtype: torch.dtype, base: float = 10_000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (float(base) ** (torch.arange(0, int(dim), 2, device=positions.device, dtype=torch.float32) / float(dim)))
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos[None, None, :, :]) + (rotate_half(x) * sin[None, None, :, :])


def sinusoidal_time_embedding(t: torch.Tensor, dim: int, scale: float = 1000.0) -> torch.Tensor:
    half = max(1, int(dim) // 2)
    denom = max(1, half - 1)
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * (-(math.log(10000.0) / float(denom))))
    args = float(scale) * t.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)
