"""Projected-token-space embedding extraction.

The conceptual extension described in §2 of the plan: measure the gap at the
LLM input layer, not just in the encoder's contrastive space.

For each pair in the diagnostic manifest:
- Run the image through CLIP vision tower -> 576 tokens x 1024.
- Apply trained projector -> 576 tokens x 4096.
- Pool to a single 4096 vector. Default: mean across tokens.
- Save the raw 576-token tensor too, so alternative pooling can be tried later
  without re-extraction (mean-pool of 576 vs ~5-30 caption tokens is noisy).

Text side:
- Tokenize with Llama-3 tokenizer.
- Look up via ``llm.get_input_embeddings()`` -> L tokens x 4096.
- Pool: mean across CONTENT tokens (excluding BOS/EOS/pad).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

from ..data.coco_loader import CocoPair, load_image
from ..encoders.base import Encoder
from ..models.projector import MLP2xGELU


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


@torch.no_grad()
def extract_projected_embeddings(
    encoder: Encoder,
    projector: MLP2xGELU,
    llm,
    tokenizer,
    pairs: list[CocoPair],
    batch_size: int = 8,
    save_raw_tokens: bool = True,
) -> dict[str, torch.Tensor]:
    """Returns dict with keys: image_pooled, text_pooled, image_tokens (optional)."""
    device = next(projector.parameters()).device
    embed_layer = llm.get_input_embeddings()

    img_pooled: list[torch.Tensor] = []
    txt_pooled: list[torch.Tensor] = []
    img_token_tensors: list[torch.Tensor] = []

    for batch in tqdm(list(_batched(pairs, batch_size)), desc="extract projected embeddings"):
        images = [load_image(p.image_path) for p in batch]
        captions = [p.caption for p in batch]

        # ---------- visual side ----------
        vis_tokens = encoder.encode_image_tokens(images).to(device)        # (B, 576, 1024)
        proj_tokens = projector(vis_tokens.to(next(projector.parameters()).dtype))  # (B, 576, 4096)
        img_pooled.append(proj_tokens.mean(dim=1).to(torch.float64).cpu())
        if save_raw_tokens:
            img_token_tensors.append(proj_tokens.to(torch.float32).cpu())

        # ---------- text side ----------
        enc = tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(device)
        text_embs = embed_layer(enc["input_ids"])     # (B, L, 4096)
        # Build content mask: attention_mask AND not BOS AND not EOS.
        att = enc["attention_mask"].bool()
        special = torch.zeros_like(att)
        if tokenizer.bos_token_id is not None:
            special |= enc["input_ids"] == tokenizer.bos_token_id
        if tokenizer.eos_token_id is not None:
            special |= enc["input_ids"] == tokenizer.eos_token_id
        content_mask = att & ~special                  # (B, L)
        weights = content_mask.float().unsqueeze(-1)   # (B, L, 1)
        denom = weights.sum(dim=1).clamp(min=1.0)      # (B, 1)
        pooled = (text_embs * weights).sum(dim=1) / denom
        txt_pooled.append(pooled.to(torch.float64).cpu())

    out = {
        "image_pooled": torch.cat(img_pooled, dim=0),
        "text_pooled": torch.cat(txt_pooled, dim=0),
    }
    if save_raw_tokens:
        out["image_tokens"] = torch.cat(img_token_tensors, dim=0)
    return out


def save_projected(blob: dict[str, torch.Tensor], out_dir: str | Path, tag: str) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, tensor in blob.items():
        p = out_dir / f"projected_{tag}_{key}.pt"
        torch.save(tensor, p)
        paths[key] = p
    return paths
