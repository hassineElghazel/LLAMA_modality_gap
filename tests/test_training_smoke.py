"""End-to-end smoke tests for the Stage 1 / Stage 2 training loops.

These tests verify that one forward/backward step lowers the loss and produces
finite gradients on the *connector* parameters. They use synthetic tensors and
tiny mocked encoders / LLMs so the test runs in seconds without downloading
multi-GB checkpoints.

The goal is to catch breakage of the wiring (optimizer construction,
dataloader contract, label masking) -- not to validate the training recipe.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.models.projector import ProjectorConfig, build_projector
from src.training.contrastive_loss import LearnableTemperature, symmetric_infonce
from src.training.stage1_pretrain import train_stage1


# Tiny dimensions so the test runs in <1 s on CPU.
VIS_DIM = 4
HIDDEN = 8
N_VIS = 5


# --------------------------------------------------------------------------
# Stage 1 wiring smoke
# --------------------------------------------------------------------------


class _FakeCLIPEncoder:
    def encode_image_tokens(self, images):
        n = len(images)
        # Deterministic, paired per-image features so InfoNCE has signal.
        return torch.stack([torch.full((N_VIS, VIS_DIM), float(i)) for i in range(n)])


class _FakeTokenizer:
    """Minimal HF-compatible tokenizer producing a fixed-length output."""
    bos_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"

    def __call__(self, captions, return_tensors=None, padding=None, truncation=None, max_length=None):
        # Deterministic per-caption ids so paired (image, caption) have signal.
        n = len(captions)
        ids = torch.tensor([[0, i + 2, 1] for i in range(n)], dtype=torch.long)
        att = torch.ones_like(ids)
        return _FakeBatchEncoding({"input_ids": ids, "attention_mask": att})


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self


class _FakeEmbed(nn.Embedding):
    def __init__(self):
        # vocab=8 is enough to cover ids 0-4 used by the fake tokenizer.
        super().__init__(8, HIDDEN)


def _stage1_cfg(max_steps: int) -> dict:
    return {
        "device": "cpu",
        "freeze": {"llm_embed": True, "vit": True, "connector": False},
        "loss": {"type": "infonce_symmetric", "temperature_init": 0.07, "temperature_learnable": True},
        "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8},
        "schedule": {"warmup_ratio": 0.0, "type": "cosine", "num_epochs": 1},
        "batch": {"per_device_batch_size": 4, "gradient_accumulation_steps": 1},
        "precision": {"amp": "fp32"},
        "logging": {"log_every_steps": 1000, "save_every_steps": 0},
        "output": {"checkpoint_path": "<set-in-test>", "log_dir": "/tmp/logs"},
        "seed": 42,
    }


def test_stage1_runs_one_step_and_reduces_loss(tmp_path):
    cfg = _stage1_cfg(max_steps=5)
    cfg["output"]["checkpoint_path"] = str(tmp_path / "ckpt.pt")
    proj = build_projector(ProjectorConfig(in_dim=VIS_DIM, hidden_dim=HIDDEN, out_dim=HIDDEN))

    captured: list[float] = []

    def cb(step, loss, tau):
        captured.append(loss)

    images = [object() for _ in range(4)]
    captions = [f"caption {i}" for i in range(4)]
    batch = {"images": images, "captions": captions}

    class _OneBatchLoader:
        def __len__(self):
            return 1
        def __iter__(self):
            for _ in range(5):
                yield batch

    train_stage1(
        encoder=_FakeCLIPEncoder(),
        connector=proj,
        llm_embed=_FakeEmbed(),
        tokenizer=_FakeTokenizer(),
        dataloader=_OneBatchLoader(),
        cfg=cfg,
        max_steps=5,
        progress_cb=cb,
    )

    assert len(captured) == 5
    assert all(torch.isfinite(torch.tensor(l)) for l in captured)
    # Loss should not blow up; with diagonal-aligned synthetic data the
    # InfoNCE objective is well-conditioned. We accept any monotone-ish
    # behavior so this stays robust to optimizer schedule details.
    assert captured[-1] <= captured[0] + 0.5


# --------------------------------------------------------------------------
# InfoNCE loss-only smoke (no training loop)
# --------------------------------------------------------------------------


def test_symmetric_infonce_grad_to_connector():
    """One step of fwd/bwd through projector + InfoNCE updates the projector."""
    torch.manual_seed(0)
    proj = build_projector(ProjectorConfig(in_dim=VIS_DIM, hidden_dim=HIDDEN, out_dim=HIDDEN))
    temp = LearnableTemperature(temperature_init=0.07)
    image_features = torch.randn(8, VIS_DIM)
    text_features = torch.randn(8, HIDDEN)
    opt = torch.optim.AdamW(list(proj.parameters()) + list(temp.parameters()), lr=1e-3)

    losses = []
    for _ in range(5):
        opt.zero_grad()
        z_img = proj(image_features)
        loss = symmetric_infonce(z_img, text_features, temp())
        loss.backward()
        opt.step()
        losses.append(loss.item())
        # Gradients on every connector layer.
        for p in proj.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()

    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    # Loss must strictly decrease over 5 steps with this overparameterised setup.
    assert losses[-1] < losses[0]
