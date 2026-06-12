"""COCO train2017 caption lookup for the C4 contrastive target.

LLaVA-Instruct-150K images come from COCO train2017. The C4 contrastive branch
needs a holistic *caption* for each image (not the instruction answer), so this
module maps a LLaVA image filename to its COCO train2017 caption.

Why first-caption-per-image and not random: the contrastive target must be
deterministic so a run is reproducible and so the (image, caption) pairing is
identical across the lambda sweep. We take the first annotation encountered for
each ``image_id``.

Prerequisite (only ``captions_val2017.json`` ships extracted by
``scripts/01_download_data.sh``): extract the train split once with

    unzip -j data/coco/annotations_trainval2017.zip \\
        annotations/captions_train2017.json -d data/coco/annotations/
"""
from __future__ import annotations

import json
import re
from pathlib import Path


class CocoTrainCaptions:
    """image_id -> caption map for COCO train2017, keyed off image filenames."""

    def __init__(self, annotations_json: str | Path):
        path = Path(annotations_json)
        if not path.exists():
            raise FileNotFoundError(
                f"COCO train2017 captions not found at {path}. Extract them with:\n"
                "  unzip -j data/coco/annotations_trainval2017.zip "
                "annotations/captions_train2017.json -d data/coco/annotations/"
            )
        with path.open() as f:
            blob = json.load(f)
        by_id: dict[int, str] = {}
        for ann in blob.get("annotations", []):
            iid = int(ann["image_id"])
            if iid not in by_id:  # keep the first caption per image (deterministic)
                by_id[iid] = str(ann["caption"]).strip()
        if not by_id:
            raise ValueError(f"no annotations parsed from {path}")
        self._by_id = by_id

    @staticmethod
    def image_id_from_filename(name: str | Path) -> int | None:
        """COCO train2017 filenames are 12-digit zero-padded ids, e.g.
        ``000000033471.jpg`` -> 33471. Handles optional subdir prefixes."""
        stem = Path(name).stem
        digits = re.sub(r"\D", "", stem)
        return int(digits) if digits else None

    def caption_for_filename(self, name: str | Path) -> str | None:
        iid = self.image_id_from_filename(name)
        if iid is None:
            return None
        return self._by_id.get(iid)

    def __len__(self) -> int:
        return len(self._by_id)
