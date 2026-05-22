"""Projector shape, init, and gradient-flow tests."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.projector import MLP2xGELU, ProjectorConfig, build_projector


def test_default_shapes():
    cfg = ProjectorConfig()
    assert cfg.in_dim == 1024
    assert cfg.hidden_dim == 4096
    assert cfg.out_dim == 4096


def test_forward_shape():
    proj = build_projector()
    # ViT-L/14 @ 224 -> 257 tokens (1 CLS + 16*16 patches), 1024-d each.
    x = torch.randn(2, 257, 1024)
    y = proj(x)
    assert y.shape == (2, 257, 4096)


def test_backward_flows():
    proj = build_projector()
    x = torch.randn(1, 257, 1024, requires_grad=True)
    y = proj(x)
    y.sum().backward()
    for name, p in proj.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert p.grad.abs().sum() > 0, f"zero grad on {name}"


def test_bias_init_zero():
    proj = build_projector()
    assert proj.fc1.bias.abs().sum().item() == 0.0
    assert proj.fc2.bias.abs().sum().item() == 0.0


def test_param_count_reasonable():
    proj = build_projector()
    n = proj.num_parameters()
    # Rough envelope: 1024*4096 + 4096 + 4096*4096 + 4096 ~= 21M
    assert 15_000_000 < n < 25_000_000
