"""Encoder-space embedding extraction for the gap diagnostics.

For each pair in the diagnostic manifest:
- Encode the image -> 768-dim L2-normalized vector (modality A).
- Encode the caption -> 768-dim L2-normalized vector (modality B).

Both saved as Float64 tensors to ``outputs/embeddings/encoder_{image,text}_embeds.pt``.
This is the input to ``compute_all_metrics`` in encoder-space.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

from ..data.coco_loader import CocoPair, load_image
from ..encoders.base import Encoder


def _batched(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


@torch.no_grad()
def extract_encoder_embeddings(
    encoder: Encoder,
    pairs: list[CocoPair],
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (image_embeds, text_embeds), both Float64, shape (N, output_dim)."""
    img_chunks: list[torch.Tensor] = []
    txt_chunks: list[torch.Tensor] = []
    for batch in tqdm(list(_batched(pairs, batch_size)), desc="extract encoder embeddings"):
        images = [load_image(p.image_path) for p in batch]
        captions = [p.caption for p in batch]
        img_chunks.append(encoder.encode_image(images).to(torch.float64).cpu())
        txt_chunks.append(encoder.encode_text(captions).to(torch.float64).cpu())
    return torch.cat(img_chunks, dim=0), torch.cat(txt_chunks, dim=0)


def save_embeddings(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    out_dir: str | Path,
    tag: str = "encoder",
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"{tag}_image_embeds.pt"
    txt_path = out_dir / f"{tag}_text_embeds.pt"
    torch.save(image_embeds, img_path)
    torch.save(text_embeds, txt_path)
    return {"image": img_path, "text": txt_path}
