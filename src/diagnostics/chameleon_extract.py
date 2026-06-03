"""Per-layer image/text hidden-state extraction for Chameleon (early fusion).

The LLaVA gap is measured at ONE interface (the connector output). Chameleon has
no connector: image and text tokens share one embedding table and one 32-layer
transformer from layer 0, so the gap is a TRAJECTORY over the 33 hidden states
(embedding output + 32 decoder layers).

For each (image, caption) pair we mean-pool the image-token hidden states and the
text-token hidden states at every layer, giving a paired (image, text) vector per
example per layer -- exactly the input ``compute_all_metrics`` (metrics.py) wants
(it requires equal-shape (n, 4096) tensors, row i of X paired with row i of Y).

Token splitting (verified against ChameleonConfig):
    image positions = input_ids == model.config.image_token_id   (1024 per image)
    text  positions = attention_mask & not(BOS|EOS|pad) & not image_token_id

Two modes:
    independent : image-only ("<image>") and caption-only forwards, pooled apart.
                  Comparable to the LLaVA single point (modalities embedded
                  independently, no cross-modal attention).
    fused       : image AND caption in one sequence ("<image>" + caption), split
                  by token type. Faithful to early fusion (cross-modal attention);
                  note the causal mask makes text attend to image but not vice
                  versa.

Mirrors src/diagnostics/extract_projected.py (batched, @torch.no_grad, content-
token masking) and casts pooled vectors to Float64 (ReAlign Appendix E.2).
"""
from __future__ import annotations

from typing import Iterable, Literal

import torch
from tqdm import tqdm

from ..data.coco_val2017_loader import CocoPair, load_image

Mode = Literal["independent", "fused"]
IMAGE_PLACEHOLDER = "<image>"


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _special_ids(model) -> set[int]:
    cfg = model.config
    ids = set()
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        v = getattr(cfg, name, None)
        if isinstance(v, int):
            ids.add(v)
    return ids


def _masked_mean_per_layer(hidden_states: tuple[torch.Tensor, ...],
                           mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool each layer's hidden states over the masked positions.

    Args:
        hidden_states: tuple of length L = num_layers+1, each (B, S, H).
        mask: (B, S) bool — positions to pool over (per example).
    Returns:
        (L, B, H) Float64 CPU tensor of per-layer, per-example pooled vectors.
    """
    w = mask.to(hidden_states[0].dtype).unsqueeze(-1)            # (B, S, 1)
    denom = w.sum(dim=1).clamp(min=1.0)                          # (B, 1)
    pooled_layers = []
    for h in hidden_states:
        pooled = (h * w).sum(dim=1) / denom                     # (B, H)
        pooled_layers.append(pooled.to(torch.float64).cpu())
    return torch.stack(pooled_layers, dim=0)                    # (L, B, H)


def _image_text_masks(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      image_token_id: int, special_ids: set[int]
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (image_mask, text_mask), each (B, S) bool."""
    image_mask = input_ids == image_token_id
    att = attention_mask.bool()
    special = torch.zeros_like(att)
    for sid in special_ids:
        special |= input_ids == sid
    text_mask = att & ~image_mask & ~special
    return image_mask, text_mask


@torch.no_grad()
def extract_trajectory(
    model,
    processor,
    pairs: list[CocoPair],
    mode: Mode,
    *,
    batch_size: int = 4,
    expected_image_tokens: int | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    """Extract per-layer paired (image, text) pooled vectors for one mode.

    Returns ``{layer_index: {"image_pooled": (N,H) f64, "text_pooled": (N,H) f64}}``
    with rows aligned across the two modalities (row i = pairs[i]).
    """
    device = model.device
    image_token_id = model.config.image_token_id
    if image_token_id is None:
        raise ValueError("model.config.image_token_id is None — cannot split image "
                         "vs text tokens. Check the checkpoint/processor.")
    special_ids = _special_ids(model)

    img_layers: list[torch.Tensor] = []   # each (L, b, H)
    txt_layers: list[torch.Tensor] = []

    for batch in tqdm(list(_batched(pairs, batch_size)), desc=f"chameleon[{mode}]"):
        images = [load_image(p.image_path) for p in batch]
        captions = [p.caption for p in batch]

        if mode == "independent":
            # --- image-only pass: "<image>" placeholder + pixel values ---
            img_in = processor(
                images=images, text=[IMAGE_PLACEHOLDER] * len(images),
                return_tensors="pt", padding=True,
            ).to(device)
            # processor returns float32 pixel_values; the VQ encoder convs are in
            # the model's (bf16) dtype, so cast to avoid a conv dtype mismatch.
            if "pixel_values" in img_in:
                img_in["pixel_values"] = img_in["pixel_values"].to(model.dtype)
            img_out = model(**img_in, output_hidden_states=True, use_cache=False)
            image_mask, _ = _image_text_masks(
                img_in["input_ids"], img_in["attention_mask"], image_token_id, special_ids)
            if expected_image_tokens is not None:
                counts = image_mask.sum(dim=1)
                if not bool((counts == expected_image_tokens).all()):
                    raise AssertionError(
                        f"expected {expected_image_tokens} image tokens/example, "
                        f"got {counts.tolist()}")
            img_layers.append(_masked_mean_per_layer(img_out.hidden_states, image_mask))

            # --- text-only pass: caption, no image ---
            txt_in = processor(
                text=captions, return_tensors="pt", padding=True,
            ).to(device)
            txt_out = model(**txt_in, output_hidden_states=True, use_cache=False)
            _, text_mask = _image_text_masks(
                txt_in["input_ids"], txt_in["attention_mask"], image_token_id, special_ids)
            txt_layers.append(_masked_mean_per_layer(txt_out.hidden_states, text_mask))

        elif mode == "fused":
            # --- one sequence: "<image>" + caption ---
            text = [IMAGE_PLACEHOLDER + c for c in captions]
            fused_in = processor(
                images=images, text=text, return_tensors="pt", padding=True,
            ).to(device)
            if "pixel_values" in fused_in:
                fused_in["pixel_values"] = fused_in["pixel_values"].to(model.dtype)
            out = model(**fused_in, output_hidden_states=True, use_cache=False)
            image_mask, text_mask = _image_text_masks(
                fused_in["input_ids"], fused_in["attention_mask"], image_token_id, special_ids)
            if expected_image_tokens is not None:
                counts = image_mask.sum(dim=1)
                if not bool((counts == expected_image_tokens).all()):
                    raise AssertionError(
                        f"expected {expected_image_tokens} image tokens/example, "
                        f"got {counts.tolist()}")
            img_layers.append(_masked_mean_per_layer(out.hidden_states, image_mask))
            txt_layers.append(_masked_mean_per_layer(out.hidden_states, text_mask))
        else:
            raise ValueError(f"unknown mode: {mode!r}")

    img = torch.cat(img_layers, dim=1)   # (L, N, H)
    txt = torch.cat(txt_layers, dim=1)   # (L, N, H)
    if img.shape != txt.shape:
        raise AssertionError(f"image/text trajectory shape mismatch: {img.shape} vs {txt.shape}")

    n_layers = img.shape[0]
    return {
        layer: {"image_pooled": img[layer], "text_pooled": txt[layer]}
        for layer in range(n_layers)
    }
