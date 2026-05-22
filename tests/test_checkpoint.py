"""Round-trip tests for projector checkpoint I/O."""
from __future__ import annotations

import torch

from src.models.checkpoint import load_projector, save_projector
from src.models.projector import ProjectorConfig, build_projector


def test_projector_roundtrip_preserves_weights_and_config(tmp_path):
    cfg = ProjectorConfig(in_dim=1024, hidden_dim=4096, out_dim=4096)
    proj = build_projector(cfg)
    # Mutate weights so we know we're loading the saved version (not a fresh init).
    with torch.no_grad():
        for p in proj.parameters():
            p.add_(0.1)

    ckpt_path = tmp_path / "proj.pt"
    save_projector(proj, ckpt_path)
    assert ckpt_path.exists()

    proj2 = load_projector(ckpt_path)
    # Config matches.
    assert proj2.cfg.in_dim == 1024
    assert proj2.cfg.hidden_dim == 4096
    assert proj2.cfg.out_dim == 4096
    # Weights match exactly.
    for (n1, p1), (n2, p2) in zip(
        proj.state_dict().items(), proj2.state_dict().items()
    ):
        assert n1 == n2
        assert torch.equal(p1, p2), f"weight mismatch on {n1}"


def test_projector_roundtrip_small_config(tmp_path):
    """A small projector with non-default dims still round-trips cleanly."""
    cfg = ProjectorConfig(in_dim=8, hidden_dim=16, out_dim=12)
    proj = build_projector(cfg)
    save_projector(proj, tmp_path / "small.pt")
    proj2 = load_projector(tmp_path / "small.pt")
    assert proj2.cfg.in_dim == 8
    assert proj2.cfg.hidden_dim == 16
    assert proj2.cfg.out_dim == 12
    # Forward agreement on a random input.
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        assert torch.allclose(proj(x), proj2(x))


def test_save_creates_parent_directories(tmp_path):
    cfg = ProjectorConfig(in_dim=4, hidden_dim=8, out_dim=4)
    proj = build_projector(cfg)
    nested = tmp_path / "deep" / "nested" / "path" / "proj.pt"
    save_projector(proj, nested)
    assert nested.exists()
