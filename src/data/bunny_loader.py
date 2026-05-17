"""Bunny-pretrain 1M loader for Stage 1 projector pretraining.

Source: BoyaWu10/Bunny-v1_0-data on HuggingFace. Filtered ~1M caption-only
samples used for modality-substitution pretraining (per ReVision / both papers).

NOTE: per the plan, Stage 1 pretraining follows the modality-substitution
recipe — text-only inputs that the projector learns to map into the LLM's
embedding space without ever seeing images. So this dataset yields
``{"text": str}`` items, not image-text pairs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class BunnyTextItem:
    text: str


class BunnyPretrainDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def __iter__(self) -> Iterator[BunnyTextItem]:
        # Bunny ships JSONL. Adjust filename to whatever the release contains
        # once downloaded; verified at first run.
        manifest = self.root / "bunny_pretrain.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(
                f"Bunny manifest not found at {manifest}. "
                "Run scripts/01_download_data.sh first."
            )
        with manifest.open() as f:
            for line in f:
                row = json.loads(line)
                # Real schema TBD on first download — placeholder field name.
                yield BunnyTextItem(text=row.get("text") or row.get("caption") or row.get("conversation"))
