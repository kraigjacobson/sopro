from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from sopro.config import SemanticEncoderConfig

CONV_RIGHT_CONTEXT_FRAMES = 2


class WhisperMelFrontend(nn.Module):
    def __init__(self, cfg: SemanticEncoderConfig) -> None:
        super().__init__()
        self.hop = int(cfg.hop_length)
        self.n_fft = int(cfg.n_fft)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(cfg.sample_rate),
            n_fft=self.n_fft,
            hop_length=self.hop,
            n_mels=int(cfg.n_mels),
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
            center=True,
        )

    def forward(self, wav16: torch.Tensor) -> tuple[torch.Tensor, int]:
        n = int(wav16.shape[-1])
        frames = (n + self.hop - 1) // self.hop
        audio = F.pad(wav16.float(), (0, self.n_fft))
        spec = self.mel(audio)[..., : frames + CONV_RIGHT_CONTEXT_FRAMES]
        log_spec = torch.log10(spec.clamp_min(1e-10))
        log_spec = torch.maximum(log_spec, log_spec.amax(dim=(1, 2), keepdim=True) - 8.0)
        return (log_spec + 4.0) / 4.0, frames


class EncoderAttention(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(d_model) // int(heads)
        self.scaling = self.head_dim ** -0.5
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = (self.q_proj(x) * self.scaling).view(b, t, self.heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2).contiguous()
        y = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        return self.out_proj(y.transpose(1, 2).reshape(b, t, -1))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.self_attn = EncoderAttention(d_model, heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class SemanticEncoder(nn.Module):
    def __init__(self, cfg: SemanticEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = int(cfg.d_model)
        self.frontend = WhisperMelFrontend(cfg)
        self.conv1 = nn.Conv1d(int(cfg.n_mels), d, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1)
        self.register_buffer("pos_emb", torch.zeros(int(cfg.max_positions), d), persistent=True)
        self.layers = nn.ModuleList([EncoderLayer(d, int(cfg.heads), int(cfg.ffn_dim)) for _ in range(int(cfg.layers))])
        self.final_norm = nn.LayerNorm(d)
        self.levels = [int(v) for v in cfg.fsq_levels]
        bases = [1] * len(self.levels)
        for i in range(1, len(self.levels)):
            bases[i] = bases[i - 1] * self.levels[i - 1]
        self.register_buffer("_bases", torch.tensor(bases, dtype=torch.long), persistent=False)
        self.pre_head_norm = nn.LayerNorm(d)
        self.digit_head = nn.Linear(d, sum(self.levels))

    @property
    def token_samples(self) -> int:
        return int(self.cfg.token_samples_24k)

    @torch.no_grad()
    def encode(self, wav24: torch.Tensor) -> torch.Tensor:
        n24 = int(wav24.shape[-1])
        n_tokens = (n24 + self.token_samples - 1) // self.token_samples
        wav16 = torchaudio.functional.resample(wav24, 24000, int(self.cfg.sample_rate))
        n16 = (n24 * 2 + 2) // 3
        wav16 = wav16[..., :n16] if int(wav16.shape[-1]) >= n16 else F.pad(wav16, (0, n16 - int(wav16.shape[-1])))
        mel, mel_frames = self.frontend(wav16)
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        n50 = (mel_frames + 1) // 2
        x = (x + self.pos_emb[: int(x.shape[1])].to(x.dtype).unsqueeze(0))[:, :n50]
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        x = self._interpolate(x, n_tokens)
        logits = self.digit_head(self.pre_head_norm(x))
        digits = torch.stack([piece.argmax(dim=-1) for piece in torch.split(logits, self.levels, dim=-1)], dim=-1)
        return (digits.to(dtype=torch.long) * self._bases.view(1, 1, -1)).sum(dim=-1)

    @staticmethod
    def _interpolate(x: torch.Tensor, n_out: int) -> torch.Tensor:
        n_in = int(x.shape[1])
        out_pos = torch.arange(int(n_out), device=x.device, dtype=torch.float32)
        ratio = torch.tensor(float(n_in), device=x.device, dtype=torch.float32) / torch.tensor(float(n_out), device=x.device, dtype=torch.float32)
        src = (out_pos + 0.5) * ratio - 0.5
        src = src.clamp(min=0.0, max=float(n_in - 1))
        left = src.floor().long()
        right = torch.minimum(left + 1, torch.full_like(left, n_in - 1))
        weight = (src - left.float()).to(dtype=x.dtype).view(1, -1, 1)
        return x[:, left] * (1.0 - weight) + x[:, right] * weight
