from __future__ import annotations

import torch


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float, top_k: int, bos_id: int, eos_id: int, allow_eos: bool) -> torch.Tensor:
    x = logits.clone()
    x[:, int(bos_id)] = -1e9
    if not allow_eos:
        x[:, int(eos_id)] = -1e9
    if float(temperature) <= 0.0:
        return torch.argmax(x, dim=-1)
    x = x / max(1e-5, float(temperature))
    probs = torch.softmax(x, dim=-1)
    if int(top_k) > 0 and int(top_k) < int(probs.shape[-1]):
        kth = torch.topk(probs, k=int(top_k), dim=-1).values[:, -1:]
        probs = torch.where(probs < kth, torch.zeros_like(probs), probs)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if float(top_p) < 1.0:
        p = float(max(0.0, min(1.0, top_p)))
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        remove = cdf > p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, :1] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        nucleus = torch.zeros_like(probs)
        nucleus.scatter_(1, sorted_idx, sorted_probs)
        probs = nucleus / nucleus.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.multinomial(probs, num_samples=1).squeeze(1)
