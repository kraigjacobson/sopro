from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from sopro.config import SpeakerEncoderConfig


class LogMelFrontend(nn.Module):
    def __init__(self, cfg: SpeakerEncoderConfig) -> None:
        super().__init__()
        self.log_floor = float(cfg.mel_log_floor)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(cfg.sample_rate),
            n_fft=int(cfg.n_fft),
            win_length=int(cfg.win_length),
            hop_length=int(cfg.hop_length),
            n_mels=int(cfg.n_mels),
            f_min=float(cfg.f_min),
            f_max=float(cfg.f_max),
            center=True,
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        mel = torch.log(self.mel(wav).clamp_min(self.log_floor)).transpose(1, 2)
        return F.layer_norm(mel, (mel.shape[-1],)).transpose(1, 2)


class Conv1dPad(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1, groups: int = 1) -> None:
        super().__init__()
        total = int(dilation) * (int(kernel_size) - 1)
        self.left_pad = total // 2
        self.right_pad = total - self.left_pad
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=int(kernel_size), dilation=int(dilation), groups=int(groups))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_pad, self.right_pad)))


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden = max(8, int(channels) // int(reduction))
        self.net = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, hidden, kernel_size=1), nn.SiLU(), nn.Conv1d(hidden, channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int, se_reduction: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.pw_in = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.dw = Conv1dPad(channels, channels, kernel_size=kernel_size, dilation=dilation, groups=channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.se = SqueezeExcite1d(channels, se_reduction)
        self.pw_out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pw_in(self.norm1(x))
        a, b = h.chunk(2, dim=1)
        h = self.dw(a * torch.sigmoid(b))
        h = F.silu(self.norm2(h))
        h = self.pw_out(self.se(h))
        return x + h


class StageTransition(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=int(stride))
        self.norm = nn.GroupNorm(1, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.conv(F.pad(x, (1, 1)))))


class AttentiveStatsPool(nn.Module):
    def __init__(self, channels: int, attn_hidden: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(nn.Conv1d(channels, attn_hidden, kernel_size=1), nn.Tanh(), nn.Conv1d(attn_hidden, 1, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.attn(x).float(), dim=-1)
        x32 = x.float()
        mean = torch.sum(w * x32, dim=-1)
        std = torch.sqrt(torch.sum(w * (x32 - mean.unsqueeze(-1)) ** 2, dim=-1).clamp_min(1e-6))
        return torch.cat([mean, std], dim=1).to(dtype=x.dtype)


class MultiScaleStylePool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.local_pool = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        denom = float(x.shape[-1])
        for y in (x, self.local_pool(x)):
            mean = y.sum(dim=-1) / denom
            var = ((y - mean.unsqueeze(-1)) ** 2).sum(dim=-1) / denom
            feats.extend([mean, torch.sqrt(var.clamp_min(1e-6))])
        return torch.cat(feats, dim=1)


class SpeakerEncoder(nn.Module):
    def __init__(self, cfg: SpeakerEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.frontend = LogMelFrontend(cfg)
        self.stem = nn.Sequential(Conv1dPad(int(cfg.n_mels), int(cfg.stem_channels), kernel_size=5), nn.GroupNorm(1, int(cfg.stem_channels)), nn.SiLU())
        in_ch = int(cfg.stem_channels)
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for stage_idx, (out_ch, n_blocks) in enumerate(zip(cfg.stage_channels, cfg.blocks_per_stage)):
            self.transitions.append(StageTransition(in_ch, int(out_ch), stride=2 if stage_idx == 0 else 1))
            self.stages.append(
                nn.Sequential(
                    *[
                        ResBlock(int(out_ch), int(cfg.dilation_cycle[b % len(cfg.dilation_cycle)]), int(cfg.depthwise_kernel_size), int(cfg.se_reduction))
                        for b in range(int(n_blocks))
                    ]
                )
            )
            in_ch = int(out_ch)
        head_ch = int(cfg.stage_channels[-1])
        self.fuse = nn.Sequential(nn.Conv1d(sum(int(c) for c in cfg.stage_channels), head_ch, kernel_size=1), nn.GroupNorm(1, head_ch), nn.SiLU())
        self.id_pool = AttentiveStatsPool(head_ch, int(cfg.attn_hidden))
        self.id_head = nn.Sequential(nn.Linear(head_ch * 2, int(cfg.id_head_hidden)), nn.SiLU(), nn.Identity(), nn.Linear(int(cfg.id_head_hidden), int(cfg.id_emb_dim)))
        self.style_pool = MultiScaleStylePool()
        self.style_head = nn.Sequential(nn.Linear(head_ch * 4, int(cfg.style_head_hidden)), nn.SiLU(), nn.Identity(), nn.Linear(int(cfg.style_head_hidden), int(cfg.style_emb_dim)))
        self.style_ctrl_head = nn.Sequential(nn.Linear(head_ch * 4, int(cfg.style_head_hidden)), nn.SiLU(), nn.Identity(), nn.Linear(int(cfg.style_head_hidden), int(cfg.style_ctrl_dim)))

    @torch.no_grad()
    def forward(self, wav16: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(self.frontend(wav16))
        feats = []
        for transition, stage in zip(self.transitions, self.stages):
            x = stage(transition(x))
            feats.append(x)
        x = self.fuse(torch.cat(feats, dim=1))
        id_emb = F.normalize(self.id_head(self.id_pool(x)), p=2, dim=-1)
        pooled_style = self.style_pool(x)
        return {"id_emb": id_emb, "style_emb": self.style_head(pooled_style), "style_ctrl": self.style_ctrl_head(pooled_style)}
