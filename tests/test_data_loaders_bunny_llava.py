"""Loader tests for Bunny-v1.1 (Stage 1) and LLaVA-Instruct-150K (Stage 2).

Uses synthetic JSON / JSONL fixtures so we cover the schema-handling logic
without downloading multi-GB datasets. The real download is left for the
50-pair end-to-end run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.bunny_v1_1_loader import BunnyV11Dataset
from src.data.llava_instruct_loader import LLaVAInstruct150KDataset


# --------------------------------------------------------------------------
# Bunny-v1.1: multiple supported manifest schemas
# --------------------------------------------------------------------------


def test_bunny_caption_schema(tmp_path):
    """Direct ``{image, caption}`` rows."""
    rows = [
        {"image": "a.jpg", "caption": "alpha"},
        {"image": "b.jpg", "caption": "beta"},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(rows))
    ds = BunnyV11Dataset(root=tmp_path, image_root=tmp_path / "imgs")
    pairs = list(ds)
    assert len(pairs) == 2
    assert pairs[0].caption == "alpha"
    assert pairs[0].image_path == tmp_path / "imgs" / "a.jpg"
    assert pairs[1].caption == "beta"


def test_bunny_text_field_schema(tmp_path):
    """Alternative ``{image, text}`` schema."""
    rows = [{"image": "x.jpg", "text": "hello world"}]
    (tmp_path / "manifest.json").write_text(json.dumps(rows))
    pairs = list(BunnyV11Dataset(root=tmp_path))
    assert len(pairs) == 1
    assert pairs[0].caption == "hello world"


def test_bunny_llava_conversation_schema(tmp_path):
    """LLaVA-style conversation rows: prefer human turn, fall back to gpt."""
    rows = [
        {
            "image": "c.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe this."},
                {"from": "gpt", "value": "A cat."},
            ],
        },
        {
            # Human turn missing -> use gpt response.
            "image": "d.jpg",
            "conversations": [
                {"from": "gpt", "value": "A dog."},
            ],
        },
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(rows))
    pairs = list(BunnyV11Dataset(root=tmp_path))
    assert len(pairs) == 2
    assert pairs[0].caption == "Describe this."   # <image> stripped, trimmed
    assert pairs[1].caption == "A dog."


def test_bunny_jsonl_manifest(tmp_path):
    lines = [
        json.dumps({"image": "a.jpg", "caption": "alpha"}),
        json.dumps({"image": "b.jpg", "caption": "beta"}),
    ]
    (tmp_path / "manifest.jsonl").write_text("\n".join(lines))
    pairs = list(BunnyV11Dataset(root=tmp_path))
    assert len(pairs) == 2
    assert [p.caption for p in pairs] == ["alpha", "beta"]


def test_bunny_skips_rows_without_caption_or_image(tmp_path):
    rows = [
        {"image": "ok.jpg", "caption": "good"},
        {"image": "missing_caption.jpg"},        # dropped
        {"caption": "no image"},                  # dropped
        {"image": "", "caption": "empty image"},  # dropped (falsy image)
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(rows))
    pairs = list(BunnyV11Dataset(root=tmp_path))
    assert len(pairs) == 1
    assert pairs[0].caption == "good"


def test_bunny_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No manifest"):
        BunnyV11Dataset(root=tmp_path)


# --------------------------------------------------------------------------
# LLaVA-Instruct-150K
# --------------------------------------------------------------------------


def test_llava_instruct_iterates_conversations(tmp_path):
    rows = [
        {
            "id": "row-0",
            "image": "COCO_train2017_000001.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is this?"},
                {"from": "gpt", "value": "A photograph of a bird."},
            ],
        },
        {
            "id": "row-1",
            "image": "COCO_train2017_000002.jpg",
            "conversations": [
                {"from": "human", "value": "Describe the scene."},
                {"from": "gpt", "value": "Two people walking."},
            ],
        },
    ]
    (tmp_path / "llava_instruct_150k.json").write_text(json.dumps(rows))
    ds = LLaVAInstruct150KDataset(root=tmp_path, image_root=tmp_path / "imgs")
    items = list(ds)
    assert len(items) == 2
    assert items[0].item_id == "row-0"
    assert items[0].image_path == tmp_path / "imgs" / "COCO_train2017_000001.jpg"
    assert items[0].conversations[0]["value"].startswith("<image>")
    assert items[1].conversations[-1]["value"] == "Two people walking."


def test_llava_instruct_prefers_canonical_filename(tmp_path):
    """Multiple JSONs in the dir -> picks ``llava_instruct_150k.json`` when present."""
    (tmp_path / "other.json").write_text(json.dumps([]))
    canonical = [{"id": "x", "image": "x.jpg", "conversations": []}]
    (tmp_path / "llava_instruct_150k.json").write_text(json.dumps(canonical))
    ds = LLaVAInstruct150KDataset(root=tmp_path, image_root=tmp_path)
    assert ds.manifest.name == "llava_instruct_150k.json"


def test_llava_instruct_skips_rows_without_image(tmp_path):
    rows = [
        {"id": "ok", "image": "a.jpg", "conversations": []},
        {"id": "skip", "conversations": []},   # no image -> dropped
    ]
    (tmp_path / "llava_instruct_150k.json").write_text(json.dumps(rows))
    items = list(LLaVAInstruct150KDataset(root=tmp_path, image_root=tmp_path))
    assert len(items) == 1
    assert items[0].item_id == "ok"


def test_llava_instruct_missing_json_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No LLaVA-Instruct"):
        LLaVAInstruct150KDataset(root=tmp_path, image_root=tmp_path)
