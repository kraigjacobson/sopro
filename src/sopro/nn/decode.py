from __future__ import annotations

from typing import Dict, Tuple

import torch

from sopro.nn.ar import KVCache

GRAPH_BUCKET = 256


class EagerDecoder:
    def __init__(self, model, capacity: int) -> None:
        self.model = model
        self.capacity = int(capacity)
        self.cache: KVCache = model.ar_prior.new_cache(1, self.capacity, model.device, model.dtype)

    def prefill(self, prefix: torch.Tensor) -> torch.Tensor:
        self.cache.length = 0
        return self.model.ar_prior(prefix, self.cache)

    def step(self, tok: torch.Tensor) -> torch.Tensor:
        return self.model.ar_prior(self.model.embed_semantic(tok.view(1, 1)), self.cache)


class CudaGraphDecoder:
    def __init__(self, model, capacity: int) -> None:
        self.model = model
        self.capacity = int(capacity)
        device, dtype = model.device, model.dtype
        self.cache: KVCache = model.ar_prior.new_cache(1, self.capacity, device, dtype)
        for t in self.cache.k + self.cache.v:
            t.zero_()
        self.tok = torch.zeros((1, 1), device=device, dtype=torch.long)
        self.pos = torch.zeros((1,), device=device, dtype=torch.long)
        self.bias = torch.full((1, 1, 1, self.capacity), float("-inf"), device=device, dtype=dtype)
        self.graphs: Dict[int, Tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}

    def _static_step(self, length: int) -> torch.Tensor:
        self.bias.index_fill_(3, self.pos, 0.0)
        logits = self.model.ar_prior.forward_static(self.model.embed_semantic(self.tok), self.cache, self.pos, self.bias[:, :, :, :length])
        self.pos.add_(1)
        return logits

    def _capture(self, length: int) -> Tuple[torch.cuda.CUDAGraph, torch.Tensor]:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._static_step(length)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = self._static_step(length)
        torch.cuda.synchronize()
        self._reset()
        self.graphs[length] = (graph, logits)
        return graph, logits

    def _reset(self) -> None:
        length = int(self.cache.length)
        self.pos.fill_(length)
        self.bias.fill_(float("-inf"))
        self.bias[:, :, :, :length] = 0.0

    def prefill(self, prefix: torch.Tensor) -> torch.Tensor:
        self.cache.length = 0
        logits = self.model.ar_prior(prefix, self.cache)
        self._reset()
        return logits

    def step(self, tok: torch.Tensor) -> torch.Tensor:
        length = min(self.capacity, ((self.cache.length + GRAPH_BUCKET) // GRAPH_BUCKET) * GRAPH_BUCKET)
        graph, logits = self.graphs.get(length) or self._capture(length)
        self.tok.copy_(tok.view(1, 1))
        graph.replay()
        self.cache.length += 1
        return logits
