"""Shared trainer pieces: optimizer, scheduler, logger."""
from __future__ import annotations

import math
from typing import Iterable

import torch


def build_adamw(params: Iterable[torch.nn.Parameter], lr: float, wd: float, betas, eps) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """LambdaLR: linear warmup then cosine decay to 0."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def freeze_module(m: torch.nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False
    m.eval()


def unfreeze_module(m: torch.nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = True
    m.train()
