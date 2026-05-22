"""Abstract vision-encoder interface.

The connector ablation study (C0/C1/C2/C3) uses a single frozen vision encoder
across all conditions. This protocol exposes only what downstream code needs:
per-token features for the connector to consume, and the configured input size
/ output shape.

Text encoding is NOT the vision encoder's job in this experiment — the text
side reads from LLaMA-2's frozen embedding layer (see
``src/diagnostics/extract_projected.py`` for Stage 1 / measurement, and
``src/models/vlm.py`` for Stage 2 splicing).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class VisionEncoder(ABC):
    """Vision-encoder interface used by the connector and gap diagnostics."""

    @property
    @abstractmethod
    def vision_hidden_dim(self) -> int:
        """Per-token hidden dim of the ViT (input to the connector)."""

    @property
    @abstractmethod
    def num_visual_tokens(self) -> int:
        """Number of tokens emitted per image (e.g. 257 for ViT-L/14 @ 224)."""

    @property
    @abstractmethod
    def image_size(self) -> int:
        """Configured square input resolution."""

    @abstractmethod
    def encode_image_tokens(self, images) -> torch.Tensor:
        """Return per-token vision-tower features of shape
        ``(B, num_visual_tokens, vision_hidden_dim)``. The first token is the
        CLS token; callers that want Stage-1-style CLS pooling should slice
        ``tokens[:, 0]``.
        """
