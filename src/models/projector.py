"""Two-layer MLP projector with GELU — standard LLaVA scheme.

Maps CLIP vision-tower token features (in_dim=1024) into the LLM input
embedding space (out_dim=4096) via a hidden GELU layer. Acts on the full token
sequence: input shape (..., in_dim), output shape (..., out_dim).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ProjectorConfig:
    in_dim: int = 1024
    hidden_dim: int = 4096
    out_dim: int = 4096


class MLP2xGELU(nn.Module):
    def __init__(self, cfg: ProjectorConfig):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.in_dim, cfg.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.out_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in (self.fc1, self.fc2):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim) -> (..., out_dim). Operates token-wise; works on
        # (B, num_visual_tokens, in_dim) without reshaping.
        return self.fc2(self.act(self.fc1(x)))

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_projector(cfg: ProjectorConfig | dict | None = None) -> MLP2xGELU:
    if cfg is None:
        cfg = ProjectorConfig()
    elif isinstance(cfg, dict):
        cfg = ProjectorConfig(
            in_dim=int(cfg.get("in_dim", 1024)),
            hidden_dim=int(cfg.get("hidden_dim", 4096)),
            out_dim=int(cfg.get("out_dim", 4096)),
        )
    return MLP2xGELU(cfg)
