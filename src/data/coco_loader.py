"""COCO Captions loader for the Karpathy split.

Two responsibilities:
1. Iterate the train/val/test split with image+caption pairs (for training and
   captioning eval).
2. Build the deterministic 10K paired sample used by the gap diagnostics.

The Karpathy ``dataset_coco.json`` is the canonical source of truth for the
113287 / 5000 / 5000 split.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from PIL import Image


SplitName = Literal["train", "val", "test", "restval"]


@dataclass
class CocoCaptionItem:
    image_id: int
    image_path: Path
    captions: list[str]
    caption_ids: list[int]


@dataclass
class CocoPair:
    """One image-caption pair used by gap diagnostics."""
    image_id: int
    image_path: Path
    caption_id: int
    caption: str


class CocoKarpathyDataset:
    def __init__(self, karpathy_json: str | Path, image_root: str | Path):
        self.karpathy_json = Path(karpathy_json)
        self.image_root = Path(image_root)
        with self.karpathy_json.open() as f:
            blob = json.load(f)
        self._raw = blob["images"]

    def items(self, splits: tuple[SplitName, ...]) -> Iterator[CocoCaptionItem]:
        wanted = set(splits)
        for entry in self._raw:
            if entry["split"] not in wanted:
                continue
            img_path = self.image_root / entry["filepath"] / entry["filename"]
            yield CocoCaptionItem(
                image_id=entry["cocoid"],
                image_path=img_path,
                captions=[s["raw"] for s in entry["sentences"]],
                caption_ids=[s["sentid"] for s in entry["sentences"]],
            )

    def count(self, split: SplitName) -> int:
        return sum(1 for e in self._raw if e["split"] == split)

    # ---------- diagnostic sample ----------

    def diagnostic_sample(
        self,
        n: int = 10000,
        pool: tuple[SplitName, ...] = ("val", "test"),
        seed: int = 42,
    ) -> list[CocoPair]:
        rng = random.Random(seed)
        all_items = list(self.items(pool))
        if n > len(all_items):
            raise ValueError(f"requested {n} pairs but pool only has {len(all_items)} images")
        chosen = rng.sample(all_items, n)
        out: list[CocoPair] = []
        for it in chosen:
            j = rng.randrange(len(it.captions))
            out.append(
                CocoPair(
                    image_id=it.image_id,
                    image_path=it.image_path,
                    caption_id=it.caption_ids[j],
                    caption=it.captions[j],
                )
            )
        return out


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_diagnostic_manifest(pairs: list[CocoPair], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "image_id": p.image_id,
            "image_path": str(p.image_path),
            "caption_id": p.caption_id,
            "caption": p.caption,
        }
        for p in pairs
    ]
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_diagnostic_manifest(in_path: str | Path) -> list[CocoPair]:
    with Path(in_path).open() as f:
        rows = json.load(f)
    return [
        CocoPair(
            image_id=r["image_id"],
            image_path=Path(r["image_path"]),
            caption_id=r["caption_id"],
            caption=r["caption"],
        )
        for r in rows
    ]
