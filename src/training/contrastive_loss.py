"""Symmetric InfoNCE loss with a learnable temperature, CLIP-style.

Per Overleaf spec §3.3, Stage 1 trains the connector with the loss

    L = 0.5 * [ CE( (z_img @ z_txt.T) / tau, y ) + CE( (z_txt @ z_img.T) / tau, y ) ]

where ``y = [0, 1, ..., N-1]`` (diagonal labels) and ``tau`` is a learnable
temperature initialised at 0.07.

This module mirrors CLIP's parameterisation: ``log_logit_scale = log(1 / tau)``
so ``logit_scale = exp(log_logit_scale)`` is naturally positive without
constraints, and we apply CLIP's clamp at 100 to prevent the contrastive head
from diverging early in training.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LearnableTemperature(nn.Module):
    """Learnable ``tau`` parameterised as ``tau = 1 / exp(log_logit_scale)``.

    Default ``temperature_init=0.07`` matches CLIP and the Overleaf spec.
    The CLIP-style upper clamp prevents ``logit_scale`` from exceeding 100,
    which empirically stabilises InfoNCE training in the first few hundred
    steps.
    """

    def __init__(self, temperature_init: float = 0.07, max_logit_scale: float = 100.0):
        super().__init__()
        if temperature_init <= 0:
            raise ValueError("temperature_init must be positive")
        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature_init)))
        self._log_max = math.log(max_logit_scale)

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.log_logit_scale.clamp(max=self._log_max).exp()

    @property
    def temperature(self) -> float:
        return float(1.0 / self.logit_scale.detach())

    def forward(self) -> torch.Tensor:
        return self.logit_scale


def symmetric_infonce(
    z_img: torch.Tensor,
    z_txt: torch.Tensor,
    logit_scale: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Symmetric InfoNCE loss over a batch of paired (image, text) embeddings.

    Args:
        z_img: (B, D) image-side embeddings.
        z_txt: (B, D) text-side embeddings.
        logit_scale: scalar tensor (positive), typically ``1 / tau``.
        normalize: if True (default), L2-normalize both sides before the dot
            product. CLIP and the Overleaf InfoNCE objective assume unit-norm
            inputs; only set False when feeding pre-normalised features.

    Returns:
        Scalar loss tensor.
    """
    if z_img.shape != z_txt.shape:
        raise ValueError(f"z_img {tuple(z_img.shape)} != z_txt {tuple(z_txt.shape)}")
    if normalize:
        z_img = F.normalize(z_img, dim=-1)
        z_txt = F.normalize(z_txt, dim=-1)
    logits_per_image = logit_scale * (z_img @ z_txt.T)
    logits_per_text = logits_per_image.T
    labels = torch.arange(z_img.size(0), device=z_img.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return 0.5 * (loss_i + loss_t)
