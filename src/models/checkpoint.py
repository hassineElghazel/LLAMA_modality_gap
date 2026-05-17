"""Checkpoint save/load helpers for projector and full VLM.

Conventions:
- Projector-only checkpoint stores ``state_dict`` + ``ProjectorConfig`` so it
  round-trips without external metadata.
- VLM checkpoints store projector + LLM trainable weights separately to keep
  load logic explicit.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from .projector import MLP2xGELU, ProjectorConfig, build_projector


def save_projector(projector: MLP2xGELU, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": projector.state_dict(), "config": asdict(projector.cfg)},
        path,
    )


def load_projector(path: str | Path, map_location: str = "cpu") -> MLP2xGELU:
    blob = torch.load(path, map_location=map_location)
    cfg = ProjectorConfig(**blob["config"])
    proj = build_projector(cfg)
    proj.load_state_dict(blob["state_dict"])
    return proj
