"""Symmetric InfoNCE — known-answer + gradient-flow tests."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from src.training.contrastive_loss import LearnableTemperature, symmetric_infonce


def test_temperature_init_matches_spec():
    """tau init 0.07 -> logit_scale = 1/0.07 ~ 14.286."""
    t = LearnableTemperature(temperature_init=0.07)
    assert math.isclose(t.temperature, 0.07, rel_tol=1e-5)
    assert math.isclose(float(t.logit_scale.detach()), 1.0 / 0.07, rel_tol=1e-5)


def test_temperature_rejects_nonpositive_init():
    with pytest.raises(ValueError):
        LearnableTemperature(temperature_init=0.0)


def test_loss_lower_bounded_at_perfect_alignment():
    """When z_img == z_txt (and rows are distinct), CE collapses to its lower
    bound: ``log(N) - logit_scale`` for large logit_scale. We just verify the
    perfectly-aligned case beats a random baseline by a large margin."""
    torch.manual_seed(0)
    N, D = 16, 64
    z = torch.randn(N, D)
    t = LearnableTemperature(temperature_init=0.07)
    perfect = symmetric_infonce(z, z.clone(), t())
    random = symmetric_infonce(z, torch.randn(N, D), t())
    assert perfect.item() < random.item()
    # Perfect alignment with large logit_scale -> near-zero loss.
    assert perfect.item() < 0.05


def test_loss_chance_level_when_targets_random():
    """When targets are independent, the symmetric loss should hover near
    ``log(N)`` (chance level)."""
    torch.manual_seed(0)
    N, D = 32, 64
    z_a = torch.randn(N, D)
    z_b = torch.randn(N, D)
    t = LearnableTemperature(temperature_init=1.0)   # mild temperature
    loss = symmetric_infonce(z_a, z_b, t())
    chance = math.log(N)
    # Loose tolerance — the small batch is noisy.
    assert abs(loss.item() - chance) < 1.5


def test_grad_flows_to_connector_like_params():
    """Synthetic connector (a Linear projecting z_img features) must receive
    gradients via the loss."""
    torch.manual_seed(0)
    N, D = 8, 32
    proj = torch.nn.Linear(D, D)
    z_raw = torch.randn(N, D)
    z_txt = torch.randn(N, D)
    z_img = proj(z_raw)
    t = LearnableTemperature(temperature_init=0.07)
    loss = symmetric_infonce(z_img, z_txt, t())
    loss.backward()
    assert proj.weight.grad is not None and proj.weight.grad.abs().sum() > 0
    assert t.log_logit_scale.grad is not None


def test_logit_scale_clamped_at_max():
    """CLIP-style clamp keeps logit_scale below max even when log_logit_scale
    is pushed very large."""
    t = LearnableTemperature(temperature_init=0.07, max_logit_scale=100.0)
    with torch.no_grad():
        t.log_logit_scale.fill_(20.0)   # exp(20) >> 100
    assert float(t.logit_scale.detach()) <= 100.0 + 1e-4
