"""LLaVA-Instruct-150K loader for Stage 2 autoregressive captioning.

Source: ``liuhaotian/LLaVA-Instruct-150K`` on HuggingFace. 150K image-grounded
instructions (image source: COCO train2017). Each item yields a conversation
(LLaVA format: list of {"from": "human"|"gpt", "value": str}).

The conversation is collapsed into a single prompt/target pair by Stage 2's
collator; for that, this loader returns the raw conversation untouched.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image


@dataclass
class LLaVAInstructItem:
    image_path: Path
    conversations: list[dict]   # [{"from": "human"|"gpt", "value": str}, ...]
    item_id: str | None = None


class LLaVAInstruct150KDataset:
    """Iterates the LLaVA-Instruct-150K conversations.

    Args:
        root: directory containing the LLaVA-Instruct JSON file.
        image_root: COCO train2017 directory.
        manifest_name: optional explicit filename (auto-detected otherwise).
    """

    def __init__(
        self,
        root: str | Path,
        image_root: str | Path,
        manifest_name: str | None = None,
    ):
        self.root = Path(root)
        self.image_root = Path(image_root)
        if manifest_name is not None:
            self.manifest = self.root / manifest_name
        else:
            cands = list(self.root.glob("*.json"))
            if not cands:
                raise FileNotFoundError(
                    f"No LLaVA-Instruct JSON under {self.root}. "
                    "Run scripts/01_download_data.sh first."
                )
            # Prefer the canonical filename if present.
            preferred = self.root / "llava_instruct_150k.json"
            self.manifest = preferred if preferred.exists() else cands[0]

    def __iter__(self) -> Iterator[LLaVAInstructItem]:
        with self.manifest.open() as f:
            rows = json.load(f)
        for r in rows:
            img = r.get("image")
            if not img:
                continue
            yield LLaVAInstructItem(
                image_path=self.image_root / img,
                conversations=r.get("conversations", []),
                item_id=str(r.get("id")) if "id" in r else None,
            )


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")
