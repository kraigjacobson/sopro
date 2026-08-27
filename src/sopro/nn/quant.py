from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GEMV_ROWS = 4


class Int8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, dtype: torch.dtype) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.register_buffer("weight", torch.zeros((self.out_features, self.in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.ones(self.out_features, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(self.out_features, dtype=dtype)) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Int8Linear":
        w = linear.weight.detach().float()
        scales = w.abs().amax(dim=1).clamp_min(1e-12) / 127.0
        q = torch.round(w / scales[:, None]).clamp(-127.0, 127.0).to(torch.int8)
        mod = cls(linear.in_features, linear.out_features, linear.bias is not None, linear.weight.dtype).to(w.device)
        mod.weight.copy_(q)
        mod.scales.copy_(scales.to(linear.weight.dtype))
        if linear.bias is not None:
            mod.bias.data.copy_(linear.bias.detach())
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, self.in_features)
        if int(x2.shape[0]) > GEMV_ROWS:
            y = F.linear(x2, self.weight.to(dtype=x.dtype) * self.scales[:, None])
        else:
            y = torch._weight_int8pack_mm(x2.contiguous(), self.weight, self.scales)
        y = y.view(*x.shape[:-1], self.out_features)
        return y if self.bias is None else y + self.bias


def quantize_ar_int8(model) -> None:
    lm = model.ar_prior
    model.sem_in_proj = Int8Linear.from_linear(model.sem_in_proj)
    lm.token_head = Int8Linear.from_linear(lm.token_head)
    for layer in lm.temporal.layers:
        attn, ffn = layer.attn, layer.ffn
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(attn, name, Int8Linear.from_linear(getattr(attn, name)))
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(ffn, name, Int8Linear.from_linear(getattr(ffn, name)))
