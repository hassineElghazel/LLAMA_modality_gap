"""Zero-shot captioning inference on COCO Karpathy test 5K.

Streams predictions to disk per batch with resume support — never accumulates
5K results in RAM (per the plan §8.4 D.1).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

from ..data.coco_val2017_loader import CocoCaptionItem, load_image
from ..models.vlm import VLM


def _format_prompt(template: str, image_token: str = "<image>") -> str:
    """LLaMA-2 base does not use an Instruct chat template; return the prompt
    verbatim. The connector replaces ``<image>`` at splice time
    (see ``VLM._build_input_embeddings``).
    """
    return template.strip()


@torch.no_grad()
def run_captioning(
    vlm: VLM,
    items: list[CocoCaptionItem],
    prompt_template: str,
    out_path: str | Path,
    batch_size: int = 8,
    gen_kwargs: dict | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: load already-completed image_ids from existing file.
    done: set[int] = set()
    existing: list[dict] = []
    if out_path.exists():
        with out_path.open() as f:
            existing = json.load(f)
        done = {row["image_id"] for row in existing}

    todo = [it for it in items if it.image_id not in done]
    full_prompt = _format_prompt(prompt_template)
    gen_kwargs = gen_kwargs or {"do_sample": False, "num_beams": 1, "max_new_tokens": 64}

    results: list[dict] = list(existing)
    for i in tqdm(range(0, len(todo), batch_size), desc="caption inference"):
        batch = todo[i : i + batch_size]
        images = [load_image(it.image_path) for it in batch]
        captions = vlm.generate(images, [full_prompt] * len(batch), **gen_kwargs)
        for it, cap in zip(batch, captions):
            results.append({"image_id": it.image_id, "caption": cap.strip()})
        # Flush after each batch.
        with out_path.open("w") as f:
            json.dump(results, f, indent=2)

    return out_path
