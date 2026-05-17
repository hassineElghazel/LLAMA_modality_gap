"""Minimal distributed-training helpers (used by Stage 2).

Stage 1 is single-GPU on the 16GB dev box. Stage 2 may run multi-GPU on A100,
but for one-epoch fine-tuning we only need rank/world-size/barrier helpers.
"""
from __future__ import annotations

import os


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return get_rank() == 0
