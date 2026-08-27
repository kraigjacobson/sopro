from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sopro.config import ModelConfig
from sopro.nn.layers import LayerScale, RMSNorm, SwiGLUFeedForward, apply_rotary, rotary_cos_sin


class KVCache:
    def __init__(self, num_layers: int, batch_size: int, kv_heads: int, max_len: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        shape = (int(batch_size), int(kv_heads), int(max_len), int(head_dim))
        self.k: List[torch.Tensor] = [torch.empty(shape, device=device, dtype=dtype) for _ in range(int(num_layers))]
        self.v: List[torch.Tensor] = [torch.empty(shape, device=device, dtype=dtype) for _ in range(int(num_layers))]
        self.cos, self.sin = rotary_cos_sin(torch.arange(int(max_len), device=device), int(head_dim), dtype)
        self.max_len = int(max_len)
        self.length = 0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.model_dim = int(cfg.ar_model_dim)
        self.num_heads = int(cfg.ar_heads)
        self.num_kv_heads = int(cfg.ar_kv_heads)
        self.head_dim = self.model_dim // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.qk_rms_norm = bool(cfg.ar_qk_rms_norm)
        if self.qk_rms_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.q_proj = nn.Linear(self.model_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.model_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.model_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.model_dim, self.model_dim, bias=False)

    def forward(self, x: torch.Tensor, cache: KVCache, layer: int, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        start = cache.length
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.qk_rms_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        cache.k[layer][:, :, start : start + t] = k
        cache.v[layer][:, :, start : start + t] = v
        keys = cache.k[layer][:, :, : start + t]
        values = cache.v[layer][:, :, : start + t]
        if self.num_kv_groups > 1:
            keys = keys.repeat_interleave(self.num_kv_groups, dim=1)
            values = values.repeat_interleave(self.num_kv_groups, dim=1)
        if start == 0:
            y = F.scaled_dot_product_attention(q, keys, values, is_causal=t > 1)
        elif t == 1:
            y = F.scaled_dot_product_attention(q, keys, values)
        else:
            qpos = torch.arange(start, start + t, device=x.device).unsqueeze(1)
            kpos = torch.arange(0, start + t, device=x.device).unsqueeze(0)
            y = F.scaled_dot_product_attention(q, keys, values, attn_mask=(kpos <= qpos))
        y = y.transpose(1, 2).contiguous().view(b, t, self.model_dim)
        return self.out_proj(y)

    def forward_static(self, x: torch.Tensor, cache: KVCache, layer: int, cos: torch.Tensor, sin: torch.Tensor, pos: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.qk_rms_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        cache.k[layer].index_copy_(2, pos, k)
        cache.v[layer].index_copy_(2, pos, v)
        length = int(bias.shape[-1])
        keys = cache.k[layer][:, :, :length]
        values = cache.v[layer][:, :, :length]
        if self.num_kv_groups > 1:
            keys = keys.repeat_interleave(self.num_kv_groups, dim=1)
            values = values.repeat_interleave(self.num_kv_groups, dim=1)
        y = F.scaled_dot_product_attention(q, keys, values, attn_mask=bias)
        y = y.transpose(1, 2).contiguous().view(b, t, self.model_dim)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        dim = int(cfg.ar_model_dim)
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(cfg)
        self.attn_scale = LayerScale(dim)
        self.ffn = SwiGLUFeedForward(dim, float(cfg.ar_ffn_mult))
        self.ffn_scale = LayerScale(dim)

    def forward(self, x: torch.Tensor, cache: KVCache, layer: int, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_scale(self.attn(self.attn_norm(x), cache, layer, cos, sin))
        return x + self.ffn_scale(self.ffn(self.ffn_norm(x)))

    def forward_static(self, x: torch.Tensor, cache: KVCache, layer: int, cos: torch.Tensor, sin: torch.Tensor, pos: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_scale(self.attn.forward_static(self.attn_norm(x), cache, layer, cos, sin, pos, bias))
        return x + self.ffn_scale(self.ffn(self.ffn_norm(x)))


class TemporalTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(int(cfg.ar_blocks))])


class SemanticLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.temporal = TemporalTransformer(cfg)
        self.out_norm = RMSNorm(int(cfg.ar_model_dim))
        self.token_head = nn.Linear(int(cfg.ar_model_dim), int(cfg.semantic_vocab_size) + 2)
        self.head_dim = int(cfg.ar_model_dim) // int(cfg.ar_heads)

    def new_cache(self, batch_size: int, max_len: int, device: torch.device, dtype: torch.dtype) -> KVCache:
        return KVCache(len(self.temporal.layers), batch_size, int(self.cfg.ar_kv_heads), max_len, self.head_dim, device, dtype)

    def forward(self, emb: torch.Tensor, cache: KVCache) -> torch.Tensor:
        t = int(emb.shape[1])
        cos = cache.cos[cache.length : cache.length + t]
        sin = cache.sin[cache.length : cache.length + t]
        h = emb
        for i, layer in enumerate(self.temporal.layers):
            h = layer(h, cache, i, cos, sin)
        cache.length += t
        return self.token_head(self.out_norm(h[:, -1:, :]))[:, 0]

    def forward_static(self, emb: torch.Tensor, cache: KVCache, pos: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        cos = cache.cos.index_select(0, pos)
        sin = cache.sin.index_select(0, pos)
        h = emb
        for i, layer in enumerate(self.temporal.layers):
            h = layer.forward_static(h, cache, i, cos, sin, pos, bias)
        return self.token_head(self.out_norm(h[:, -1:, :]))[:, 0]


class StylePrefixEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        dim = int(cfg.ar_model_dim)
        self.num_tokens = int(cfg.style_prefix_tokens)
        self.heads = int(cfg.ar_heads)
        self.head_dim = dim // self.heads
        self.queries = nn.Parameter(torch.zeros(self.num_tokens, dim))
        self.kv_norm = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.out_norm = RMSNorm(dim)

    def forward(self, ref_emb: torch.Tensor) -> torch.Tensor:
        bsz, steps, dim = ref_emb.shape
        queries = self.queries.unsqueeze(0).to(dtype=ref_emb.dtype)
        if steps == 0:
            return self.out_norm(queries).expand(bsz, -1, -1)
        kv = self.kv_norm(ref_emb)
        q = self.q_proj(queries.expand(bsz, -1, -1)).view(bsz, self.num_tokens, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(bsz, steps, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(bsz, steps, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        attn = torch.softmax(scores, dim=-1)
        y = torch.matmul(attn, v.float()).to(dtype=ref_emb.dtype)
        y = y.transpose(1, 2).contiguous().view(bsz, self.num_tokens, dim)
        return self.out_norm(queries + self.out_proj(y))
