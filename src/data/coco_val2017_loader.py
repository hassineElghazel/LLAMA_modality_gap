"""COCO val2017 loader.

Per Overleaf spec §6.4: evaluation set is COCO val2017 (5,000 images, 5
captions each). Same set is used for every gap measurement across C0/C1/C2/C3
AND for the captioning quality benchmarks. No overlap with Stage 1 (Bunny-v1.1)
or Stage 2 (LLaVA-Instruct-150K, which is image-grounded on COCO train2017).

The annotations JSON follows the official COCO format:
    {
      "images": [{"id": ..., "file_name": ...}, ...],
      "annotations": [{"id": ..., "image_id": ..., "caption": ...}, ...],
      ...
    }
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image


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


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


class CocoVal2017Dataset:
    """Loader for COCO val2017 captions.

    Args:
        annotations_json: path to ``captions_val2017.json``.
        image_root: directory containing the val2017 jpeg files.
    """

    def __init__(self, annotations_json: str | Path, image_root: str | Path):
        self.annotations_json = Path(annotations_json)
        self.image_root = Path(image_root)
        with self.annotations_json.open() as f:
            blob = json.load(f)
        # Build image_id -> file_name and image_id -> [(cap_id, caption), ...].
        self._file_names: dict[int, str] = {
            img["id"]: img["file_name"] for img in blob["images"]
        }
        caps: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for ann in blob["annotations"]:
            caps[ann["image_id"]].append((ann["id"], ann["caption"].strip()))
        # Stable order by caption id.
        self._caps: dict[int, list[tuple[int, str]]] = {
            iid: sorted(v, key=lambda t: t[0]) for iid, v in caps.items()
        }

    # ---------------- iteration ----------------

    def __len__(self) -> int:
        return len(self._file_names)

    def items(self) -> Iterator[CocoCaptionItem]:
        for image_id in sorted(self._file_names):
            entries = self._caps.get(image_id, [])
            yield CocoCaptionItem(
                image_id=image_id,
                image_path=self.image_root / self._file_names[image_id],
                captions=[c for _, c in entries],
                caption_ids=[cid for cid, _ in entries],
            )

    # ---------------- diagnostic sample ----------------

    def diagnostic_sample(
        self,
        n: int | None = None,
        caption_per_image: int = 1,
        seed: int = 42,
    ) -> list[CocoPair]:
        """Deterministic single-caption-per-image sample (n images).

        Defaults to the full 5K val2017. ``caption_per_image=1`` picks one
        caption per image with a fixed seed; >1 returns multiple paired
        rows per image (rarely needed).
        """
        rng = random.Random(seed)
        all_items = list(self.items())
        if n is None or n >= len(all_items):
            chosen = all_items
        else:
            chosen = rng.sample(all_items, n)
        out: list[CocoPair] = []
        for it in chosen:
            if not it.captions:
                continue
            k = min(caption_per_image, len(it.captions))
            picks = rng.sample(range(len(it.captions)), k)
            for j in picks:
                out.append(
                    CocoPair(
                        image_id=it.image_id,
                        image_path=it.image_path,
                        caption_id=it.caption_ids[j],
                        caption=it.captions[j],
                    )
                )
        return out

    def references(self) -> dict[int, list[str]]:
        """Return image_id -> list of all reference captions (for pycocoevalcap)."""
        return {iid: [c for _, c in self._caps.get(iid, [])] for iid in self._file_names}


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
