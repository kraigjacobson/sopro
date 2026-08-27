from __future__ import annotations

from typing import Iterator, Optional

import torch
import torch.nn as nn

from sopro.config import ModelConfig
from sopro.nn.acoustic import AcousticHead
from sopro.nn.ar import SemanticLM, StylePrefixEncoder
from sopro.nn.decode import CudaGraphDecoder, EagerDecoder
from sopro.sampling import sample_next_token

GRAPH_CAPACITY = 2048


class SoproModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cond_proj = nn.Sequential(
            nn.Linear(int(cfg.cond_in_dim), int(cfg.cond_hidden_dim)),
            nn.SiLU(),
            nn.Identity(),
            nn.Linear(int(cfg.cond_hidden_dim), int(cfg.cond_hidden_dim)),
        )
        self.text_tok_emb = nn.Embedding(int(cfg.text_vocab_size), int(cfg.ar_model_dim))
        self.semantic_tok_emb = nn.Embedding(int(cfg.semantic_vocab_size) + 2, int(cfg.latent_dim))
        self.sem_in_proj = nn.Linear(int(cfg.latent_dim), int(cfg.ar_model_dim))
        self.style_prefix = StylePrefixEncoder(cfg)
        self.ar_prior = SemanticLM(cfg)
        self.acoustic_head = AcousticHead(cfg)
        self._graph_decoder = None

    @property
    def device(self) -> torch.device:
        return self.text_tok_emb.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.text_tok_emb.weight.dtype

    def build_condition(self, id_emb: torch.Tensor, style_emb: torch.Tensor, style_ctrl: torch.Tensor) -> torch.Tensor:
        return self.cond_proj(torch.cat([id_emb, style_emb, style_ctrl], dim=-1))

    def embed_semantic(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.sem_in_proj(self.semantic_tok_emb(tokens.to(dtype=torch.long)))

    def decoder(self, capacity: int):
        if self.device.type != "cuda":
            return EagerDecoder(self, capacity)
        if self._graph_decoder is None or self._graph_decoder.capacity < int(capacity):
            self._graph_decoder = CudaGraphDecoder(self, max(int(capacity), GRAPH_CAPACITY))
        return self._graph_decoder

    def build_prefix(self, text_ids: torch.Tensor, style_tokens: torch.Tensor, prompt_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        text_ids = text_ids[:, : int(self.cfg.max_text_len)]
        bos = torch.full((1, 1), int(self.cfg.semantic_bos_id), device=text_ids.device, dtype=torch.long)
        bos_emb = self.embed_semantic(bos)
        parts = [self.style_prefix(self.embed_semantic(style_tokens)).to(dtype=bos_emb.dtype), self.text_tok_emb(text_ids).to(dtype=bos_emb.dtype)]
        if prompt_tokens is not None and int(prompt_tokens.shape[-1]) > 0:
            parts.append(self.embed_semantic(prompt_tokens).to(dtype=bos_emb.dtype))
        parts.append(bos_emb)
        return torch.cat(parts, dim=1)

    @torch.no_grad()
    def stream_semantic_tokens(
        self,
        text_ids: torch.Tensor,
        style_tokens: torch.Tensor,
        prompt_tokens: Optional[torch.Tensor],
        *,
        chunk_size: int,
        max_steps: int,
        min_steps: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Iterator[torch.Tensor]:
        prefix = self.build_prefix(text_ids, style_tokens, prompt_tokens)
        decoder = self.decoder(int(prefix.shape[1]) + int(max_steps))
        logits = decoder.prefill(prefix)
        bos_id, eos_id = int(self.cfg.semantic_bos_id), int(self.cfg.semantic_eos_id)
        pending = []
        for step in range(int(max_steps)):
            allow_eos = (step + 1) >= max(1, int(min_steps))
            tok = sample_next_token(logits, float(temperature), float(top_p), int(top_k), bos_id, eos_id, allow_eos)
            if allow_eos and int(tok.item()) == eos_id:
                break
            tok = tok.clamp_min(0).clamp_max(int(self.cfg.semantic_vocab_size) - 1)
            pending.append(tok.view(1, 1))
            if len(pending) >= int(chunk_size):
                yield torch.cat(pending, dim=1)
                pending = []
            if step + 1 < int(max_steps):
                logits = decoder.step(tok)
        if pending:
            yield torch.cat(pending, dim=1)
