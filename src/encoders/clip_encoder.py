"""Plain CLIP ViT-L/14 vision encoder, frozen across all C0/C1/C2/C3 conditions.

Spec (Overleaf, §2 Architecture):
    image (224x224) -> CLIP ViT-L/14 -> 257 tokens of 1024-dim (1 CLS + 16x16 patches).

This wrapper exposes the raw per-token output of the ViT (``last_hidden_state``)
with no projection head and no text tower. The connector (configs/projector.yaml)
ingests these tokens directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image

from .base import VisionEncoder


@dataclass
class CLIPEncoderConfig:
    hf_id: str = "openai/clip-vit-large-patch14"
    image_size: int = 224
    num_visual_tokens: int = 257   # 1 CLS + 16*16 patches at 224 / 14
    vision_hidden_dim: int = 1024
    device: str = "cuda"
    weights_dtype: str = "bfloat16"


class CLIPViTL14Encoder(VisionEncoder):
    """Frozen CLIP ViT-L/14 wrapper returning per-token features (B, 257, 1024)."""

    def __init__(self, cfg: Optional[CLIPEncoderConfig] = None):
        self.cfg = cfg or CLIPEncoderConfig()
        self._vision = None
        self._image_processor = None

    def load(self) -> "CLIPViTL14Encoder":
        from transformers import CLIPImageProcessor, CLIPVisionModel

        dtype = getattr(torch, self.cfg.weights_dtype)
        self._vision = (
            CLIPVisionModel.from_pretrained(self.cfg.hf_id, torch_dtype=dtype)
            .to(self.cfg.device)
            .eval()
        )
        for p in self._vision.parameters():
            p.requires_grad = False
        self._image_processor = CLIPImageProcessor.from_pretrained(self.cfg.hf_id)
        return self

    def _require_loaded(self) -> None:
        if self._vision is None:
            raise RuntimeError("Encoder not loaded — call .load() first")

    # ------------------------------------------------------------------
    # VisionEncoder protocol
    # ------------------------------------------------------------------

    @property
    def vision_hidden_dim(self) -> int:
        return self.cfg.vision_hidden_dim

    @property
    def num_visual_tokens(self) -> int:
        return self.cfg.num_visual_tokens

    @property
    def image_size(self) -> int:
        return self.cfg.image_size

    def _preprocess(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        return self._image_processor(images=images, return_tensors="pt").to(self.cfg.device)

    @torch.no_grad()
    def encode_image_tokens(self, images) -> torch.Tensor:
        """Return last-hidden-state tokens shaped (B, 257, 1024).

        Token 0 is CLS; tokens 1..256 are the patch tokens. Spec Stage 1 uses
        ``tokens[:, 0]`` (CLS) for the InfoNCE image side; Stage 2 splices the
        full 257-token sequence into LLaMA's input.
        """
        self._require_loaded()
        inputs = self._preprocess(images)
        out = self._vision(**inputs)
        tokens = out.last_hidden_state  # (B, 257, 1024)
        if tokens.shape[1] != self.cfg.num_visual_tokens:
            raise RuntimeError(
                f"expected {self.cfg.num_visual_tokens} tokens, got {tokens.shape[1]} "
                f"(image_size={self.cfg.image_size}; check the processor config)"
            )
        if tokens.shape[-1] != self.cfg.vision_hidden_dim:
            raise RuntimeError(
                f"expected hidden_dim={self.cfg.vision_hidden_dim}, got {tokens.shape[-1]}"
            )
        return tokens


def build_clip_encoder(cfg_dict: dict | None = None) -> CLIPViTL14Encoder:
    """Construct a CLIPViTL14Encoder from a ``configs/encoders.yaml`` dict."""
    if cfg_dict is None:
        return CLIPViTL14Encoder()
    vm = cfg_dict.get("vision_model", {})
    inf = cfg_dict.get("inference", {})
    cfg = CLIPEncoderConfig(
        hf_id=vm.get("hf_id", "openai/clip-vit-large-patch14"),
        image_size=int(vm.get("image_size", 224)),
        num_visual_tokens=int(vm.get("num_visual_tokens", 257)),
        vision_hidden_dim=int(vm.get("vision_hidden_dim", 1024)),
        device=inf.get("device", "cuda"),
        weights_dtype=inf.get("weights_dtype", "bfloat16"),
    )
    return CLIPViTL14Encoder(cfg)
