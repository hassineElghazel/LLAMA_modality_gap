"""Bunny-v1.1-data loader for Stage 1 InfoNCE contrastive pretraining.

Source: ``BoyaWu10/Bunny-v1.1-data`` on HuggingFace. Yields (image, caption)
pairs for the symmetric InfoNCE loop in
``src/training/stage1_pretrain.py``.

The Bunny-v1.1-data release ships image archives + a JSONL manifest of
conversation-style entries. For the InfoNCE alignment objective we extract the
human prompt as the caption (or fall back to the GPT response). The exact
schema is verified at first download; this loader handles both ``{"image":
..., "conversations": [...]}`` (LLaVA-style) and ``{"image": ..., "caption":
...}`` (caption-only) shapes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image


@dataclass
class BunnyPair:
    image_path: Path
    caption: str


def _extract_caption(row: dict) -> str | None:
    if "caption" in row and row["caption"]:
        return str(row["caption"]).strip()
    if "text" in row and row["text"]:
        return str(row["text"]).strip()
    convs = row.get("conversations") or row.get("conversation")
    if isinstance(convs, list) and convs:
        # Prefer the human turn (the prompt); fall back to gpt response.
        human = next((t for t in convs if str(t.get("from", "")).lower() == "human"), None)
        if human and human.get("value"):
            return str(human["value"]).replace("<image>", "").strip()
        gpt = next((t for t in convs if str(t.get("from", "")).lower() in ("gpt", "assistant")), None)
        if gpt and gpt.get("value"):
            return str(gpt["value"]).strip()
    return None


class BunnyV11Dataset:
    """Iterates (image_path, caption) pairs.

    Looks for a single manifest file ending in ``.json`` or ``.jsonl`` at
    ``root``, and resolves image paths relative to ``image_root`` (defaults to
    the manifest's parent directory).
    """

    def __init__(
        self,
        root: str | Path,
        manifest_name: str | None = None,
        image_root: str | Path | None = None,
    ):
        self.root = Path(root)
        self.image_root = Path(image_root) if image_root else self.root
        if manifest_name is not None:
            self.manifest = self.root / manifest_name
        else:
            cands = list(self.root.glob("*.jsonl")) + list(self.root.glob("*.json"))
            if not cands:
                raise FileNotFoundError(
                    f"No manifest (.json/.jsonl) found under {self.root}. "
                    "Run scripts/01_download_data.sh first."
                )
            self.manifest = cands[0]

    def _iter_rows(self) -> Iterator[dict]:
        text = self.manifest.read_text()
        if self.manifest.suffix == ".jsonl":
            for line in text.splitlines():
                line = line.strip()
                if line:
                    yield json.loads(line)
        else:
            blob = json.loads(text)
            if isinstance(blob, list):
                yield from blob
            else:
                yield blob

    def __iter__(self) -> Iterator[BunnyPair]:
        for row in self._iter_rows():
            cap = _extract_caption(row)
            img = row.get("image") or row.get("image_path") or row.get("img")
            if not cap or not img:
                continue
            yield BunnyPair(image_path=self.image_root / img, caption=cap)


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")
