"""COCO Karpathy loader tests — exercise the deterministic-sample logic
without touching real images."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.coco_loader import CocoKarpathyDataset, load_diagnostic_manifest, save_diagnostic_manifest


@pytest.fixture
def fake_karpathy(tmp_path: Path) -> Path:
    blob = {
        "images": [
            {
                "cocoid": 1000 + i,
                "split": "val" if i < 50 else "test",
                "filepath": "val2014",
                "filename": f"COCO_val2014_{i:012d}.jpg",
                "sentences": [
                    {"sentid": 10 * i + s, "raw": f"caption {i} variant {s}"}
                    for s in range(5)
                ],
            }
            for i in range(100)
        ]
    }
    p = tmp_path / "dataset_coco.json"
    p.write_text(json.dumps(blob))
    return p


def test_loader_counts(fake_karpathy):
    ds = CocoKarpathyDataset(fake_karpathy, image_root="/dev/null")
    assert ds.count("val") == 50
    assert ds.count("test") == 50


def test_diagnostic_sample_deterministic(fake_karpathy):
    ds = CocoKarpathyDataset(fake_karpathy, image_root="/dev/null")
    a = ds.diagnostic_sample(n=20, pool=("val", "test"), seed=42)
    b = ds.diagnostic_sample(n=20, pool=("val", "test"), seed=42)
    assert [(p.image_id, p.caption_id) for p in a] == [(p.image_id, p.caption_id) for p in b]


def test_diagnostic_sample_different_seed_differs(fake_karpathy):
    ds = CocoKarpathyDataset(fake_karpathy, image_root="/dev/null")
    a = ds.diagnostic_sample(n=20, pool=("val", "test"), seed=42)
    b = ds.diagnostic_sample(n=20, pool=("val", "test"), seed=7)
    assert [(p.image_id, p.caption_id) for p in a] != [(p.image_id, p.caption_id) for p in b]


def test_manifest_roundtrip(fake_karpathy, tmp_path):
    ds = CocoKarpathyDataset(fake_karpathy, image_root="/dev/null")
    pairs = ds.diagnostic_sample(n=20, pool=("val", "test"), seed=42)
    manifest = tmp_path / "manifest.json"
    save_diagnostic_manifest(pairs, manifest)
    restored = load_diagnostic_manifest(manifest)
    assert [(p.image_id, p.caption_id, p.caption) for p in pairs] == \
           [(p.image_id, p.caption_id, p.caption) for p in restored]


def test_diagnostic_sample_caps_at_pool_size(fake_karpathy):
    ds = CocoKarpathyDataset(fake_karpathy, image_root="/dev/null")
    with pytest.raises(ValueError):
        ds.diagnostic_sample(n=10_000, pool=("val", "test"), seed=42)
