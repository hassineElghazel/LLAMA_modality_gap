"""InternVL-Chat-V1.2-SFT loader for Stage 2 visual instruction tuning.

Roughly 1.2M image-instruction pairs in conversation format. Each item yields
an image path plus a multi-turn conversation (LLaVA-style messages list).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class InternVLConvItem:
    image_path: Path
    conversations: list[dict]   # [{"from": "human"|"gpt", "value": str}, ...]


class InternVLSFTDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def __iter__(self) -> Iterator[InternVLConvItem]:
        manifest = self.root / "internvl_sft.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"InternVL SFT manifest not found at {manifest}. "
                "Run scripts/01_download_data.sh first."
            )
        with manifest.open() as f:
            rows = json.load(f)
        for r in rows:
            yield InternVLConvItem(
                image_path=self.root / r["image"],
                conversations=r["conversations"],
            )
