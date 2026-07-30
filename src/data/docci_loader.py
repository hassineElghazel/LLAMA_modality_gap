"""DOCCI loader (google/docci, ECCV 2024, CC BY 4.0).

DOCCI is UNSEEN by LLaVA-1.5 (2024 single-photographer private photos; disjoint
from COCO / Visual Genome / Open Images / LAION-558K), so a vanilla fine-tune of
LLaVA on it is a FAIR baseline (on data the model already trained on, the AR
gradient is ~0 and any geometry arm would win trivially). It also carries long
(~136-token) human DETAIL captions -> real headroom for detailed captioning.

This module reads the on-disk manifests written by ``scripts/24_prep_docci.py``.
Each manifest row is::

    {"image_id": int, "image_path": str, "example_id": str, "caption": str}

``image_id`` is the split-local enumeration index (0..N-1); images are saved by
24 as zero-padded ``{image_id:012d}.jpg`` so the existing COCO-style scorers
(15_clipscore, 15b_siglip) resolve them via ``image_root / f"{id:012d}.jpg"``
UNCHANGED. ``caption`` is the single gold DOCCI ``description``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocciItem:
    image_id: int
    image_path: Path
    example_id: str
    caption: str


def load_docci_manifest(path: str | Path) -> list[DocciItem]:
    with Path(path).open() as f:
        rows = json.load(f)
    return [
        DocciItem(
            image_id=int(r["image_id"]),
            image_path=Path(r["image_path"]),
            example_id=str(r["example_id"]),
            caption=str(r["caption"]),
        )
        for r in rows
    ]


def docci_references(path: str | Path) -> dict[int, list[str]]:
    """image_id -> [gold DOCCI caption] for pycocoevalcap (single reference).

    Single-reference METEOR/SPICE are usable (SPICE is scene-graph semantic, not
    n-gram); CIDEr with one ref is noisy so callers may drop it. Circular vs the
    zero-FT reference arm (arms fine-tuned on DOCCI captions match its style) but
    FAIR between the geometry arms, which share identical training data.
    """
    items = load_docci_manifest(path)
    return {it.image_id: [it.caption] for it in items}
