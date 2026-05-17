"""Abstract encoder interface.

Both modality encoders (image and text) live behind the same ``Encoder``
protocol so the diagnostic pipeline can swap implementations without changing
calling code. ``LLM2CLIPEncoder`` (the only implementation in this phase) wraps
the HuggingFace LLM2CLIP model.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Encoder(ABC):
    """Encoder interface for modality-gap diagnostics."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the contrastive (L2-normalized) output space."""

    @property
    @abstractmethod
    def vision_hidden_dim(self) -> int:
        """Per-token hidden dim of the ViT vision tower (input to projector)."""

    @abstractmethod
    def encode_image(self, images) -> torch.Tensor:
        """Return L2-normalized contrastive image embeddings, shape (B, output_dim)."""

    @abstractmethod
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Return L2-normalized contrastive text embeddings, shape (B, output_dim)."""

    @abstractmethod
    def encode_image_tokens(self, images) -> torch.Tensor:
        """Return per-token vision-tower features, shape (B, num_visual_tokens, vision_hidden_dim).

        These are the features the projector ingests; distinct from
        ``encode_image`` which returns the contrastive-pooled output.
        """
