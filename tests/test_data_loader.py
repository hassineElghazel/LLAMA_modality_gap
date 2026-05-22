"""COCO val2017 loader tests — exercise the deterministic-sample logic
without touching real images."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.coco_val2017_loader import (
    CocoVal2017Dataset,
    load_diagnostic_manifest,
    save_diagnostic_manifest,
)


@pytest.fixture
def fake_val2017(tmp_path: Path) -> Path:
    images = [
        {"id": 1000 + i, "file_name": f"COCO_val2017_{i:012d}.jpg"}
        for i in range(100)
    ]
    annotations = []
    ann_id = 0
    for img in images:
        for s in range(5):   # 5 captions per image, per COCO convention
            annotations.append(
                {"id": ann_id, "image_id": img["id"], "caption": f"image {img['id']} caption {s}"}
            )
            ann_id += 1
    blob = {"images": images, "annotations": annotations}
    p = tmp_path / "captions_val2017.json"
    p.write_text(json.dumps(blob))
    return p


def test_loader_counts(fake_val2017):
    ds = CocoVal2017Dataset(fake_val2017, image_root="/dev/null")
    assert len(ds) == 100
    refs = ds.references()
    assert all(len(v) == 5 for v in refs.values())


def test_diagnostic_sample_deterministic(fake_val2017):
    ds = CocoVal2017Dataset(fake_val2017, image_root="/dev/null")
    a = ds.diagnostic_sample(n=20, seed=42)
    b = ds.diagnostic_sample(n=20, seed=42)
    assert [(p.image_id, p.caption_id) for p in a] == [(p.image_id, p.caption_id) for p in b]


def test_diagnostic_sample_different_seed_differs(fake_val2017):
    ds = CocoVal2017Dataset(fake_val2017, image_root="/dev/null")
    a = ds.diagnostic_sample(n=20, seed=42)
    b = ds.diagnostic_sample(n=20, seed=7)
    assert [(p.image_id, p.caption_id) for p in a] != [(p.image_id, p.caption_id) for p in b]


def test_manifest_roundtrip(fake_val2017, tmp_path):
    ds = CocoVal2017Dataset(fake_val2017, image_root="/dev/null")
    pairs = ds.diagnostic_sample(n=20, seed=42)
    manifest = tmp_path / "manifest.json"
    save_diagnostic_manifest(pairs, manifest)
    restored = load_diagnostic_manifest(manifest)
    assert [(p.image_id, p.caption_id, p.caption) for p in pairs] == \
           [(p.image_id, p.caption_id, p.caption) for p in restored]


def test_full_sample_when_n_exceeds_pool(fake_val2017):
    ds = CocoVal2017Dataset(fake_val2017, image_root="/dev/null")
    out = ds.diagnostic_sample(n=10_000, seed=42)
    # All 100 images returned; caption_per_image=1 -> 100 pairs.
    assert len(out) == 100
