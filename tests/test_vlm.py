"""Unit tests for VLM image-token splicing.

The splice is the critical part of ``VLM._build_inputs``: every ``<image>``
placeholder must be replaced with exactly ``num_visual_tokens`` projected
tokens, and (when training) the corresponding label positions must be set to
-100 so the AR loss is computed on text tokens only.

These tests mock the encoder, projector, LLM, and tokenizer with the smallest
possible torch.nn modules so the assertions exercise the splice arithmetic
without downloading 7B weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from src.models.vlm import VLM, VLMConfig


HIDDEN = 8       # tiny LLM hidden size
VIS_DIM = 4      # tiny ViT hidden size
N_VIS = 5        # tiny visual-token count (would be 257 in production)
VOCAB = 32       # tiny vocabulary
IMAGE_TOKEN_ID = 31


class _DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_image_count = None

    def encode_image_tokens(self, images):
        n = len(images)
        self.last_image_count = n
        # Deterministic per-batch tensor so tests can match on shape only.
        return torch.arange(n * N_VIS * VIS_DIM, dtype=torch.float32).reshape(n, N_VIS, VIS_DIM)


class _DummyProjector(nn.Module):
    """VIS_DIM -> HIDDEN passthrough that stays differentiable."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(VIS_DIM, HIDDEN, bias=False)

    def forward(self, x):
        return self.lin(x)


class _DummyEmbed(nn.Embedding):
    def __init__(self):
        super().__init__(VOCAB, HIDDEN)


class _DummyLLM(nn.Module):
    """Minimal stand-in for HF AutoModelForCausalLM."""

    def __init__(self):
        super().__init__()
        self.embed = _DummyEmbed()
        self.head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.last_inputs_embeds_shape = None
        self.last_labels_shape = None

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None, **_):
        self.last_inputs_embeds_shape = tuple(inputs_embeds.shape)
        self.last_labels_shape = None if labels is None else tuple(labels.shape)
        logits = self.head(inputs_embeds)
        if labels is None:
            return type("Out", (), {"logits": logits})()
        # Compute the same masked AR cross-entropy that HF would.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.reshape(-1, VOCAB),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )
        return type("Out", (), {"logits": logits, "loss": loss})()


def _build_vlm() -> VLM:
    encoder = _DummyEncoder()
    projector = _DummyProjector()
    vlm = VLM(encoder, projector, VLMConfig(device="cpu", weights_dtype="float32"))
    vlm._llm = _DummyLLM()
    vlm._image_token_id = IMAGE_TOKEN_ID
    return vlm


# ----- shape / splice arithmetic -----


def test_splice_inserts_n_vis_tokens_and_expands_labels():
    vlm = _build_vlm()
    B, L = 2, 6
    # Place <image> at position 1 in both rows.
    input_ids = torch.tensor(
        [[5, IMAGE_TOKEN_ID, 7, 8, 9, 10],
         [4, IMAGE_TOKEN_ID, 6, 7, 8, 9]],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [[-100, -100, 7, 8, 9, 10],
         [-100, -100, 6, 7, 8, 9]],
        dtype=torch.long,
    )
    images = [object(), object()]   # only len() matters for _DummyEncoder

    inputs_embeds, attn, exp_labels = vlm._build_inputs(images, input_ids, labels)
    # text tokens (L) + visual tokens (N_VIS) - 1 placeholder swapped out.
    expected_seq = L + N_VIS - 1
    assert inputs_embeds.shape == (B, expected_seq, HIDDEN)
    assert attn.shape == (B, expected_seq)
    assert exp_labels.shape == (B, expected_seq)
    # All attention mask positions inside each row should be 1 (we padded
    # nothing here since both rows have the same length).
    assert torch.all(attn == 1)


def test_visual_positions_are_label_ignored():
    vlm = _build_vlm()
    input_ids = torch.tensor([[5, IMAGE_TOKEN_ID, 7, 8]], dtype=torch.long)
    labels = torch.tensor([[-100, 99, 7, 8]], dtype=torch.long)   # 99 marks the placeholder slot
    inputs_embeds, attn, exp_labels = vlm._build_inputs([object()], input_ids, labels)
    # Visual positions span [1, 1+N_VIS) and must be -100.
    assert torch.all(exp_labels[0, 1 : 1 + N_VIS] == -100)
    # Text positions before and after retain original label values.
    assert exp_labels[0, 0].item() == -100
    assert exp_labels[0, 1 + N_VIS].item() == 7
    assert exp_labels[0, 1 + N_VIS + 1].item() == 8


def test_missing_image_token_raises():
    vlm = _build_vlm()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)   # no IMAGE_TOKEN_ID
    with pytest.raises(ValueError, match="expected exactly 1 image token"):
        vlm._build_inputs([object()], input_ids, labels=None)


def test_multiple_image_tokens_in_one_row_rejected():
    vlm = _build_vlm()
    input_ids = torch.tensor([[IMAGE_TOKEN_ID, 2, IMAGE_TOKEN_ID, 4]], dtype=torch.long)
    with pytest.raises(ValueError, match="expected exactly 1 image token"):
        vlm._build_inputs([object()], input_ids, labels=None)


def test_padding_handles_heterogeneous_row_lengths():
    """Different placeholder positions still yield rectangular outputs."""
    vlm = _build_vlm()
    # Row 0 length 6, row 1 length 4; both contain a single placeholder.
    row0 = torch.tensor([5, IMAGE_TOKEN_ID, 7, 8, 9, 10], dtype=torch.long)
    row1 = torch.tensor([IMAGE_TOKEN_ID, 6, 7, 8], dtype=torch.long)
    # Right-pad row1 with a non-image token so input_ids stays rectangular.
    PAD = 0
    row1_padded = torch.cat([row1, torch.tensor([PAD, PAD], dtype=torch.long)])
    input_ids = torch.stack([row0, row1_padded])
    inputs_embeds, attn, _ = vlm._build_inputs([object(), object()], input_ids, labels=None)
    assert inputs_embeds.shape[0] == 2
    # The padded row gets the same flat sequence length but attention mask
    # equals the actual non-padded length per row.
    assert attn[0].sum().item() >= attn[1].sum().item()


# ----- end-to-end forward producing a finite loss -----


def test_forward_with_labels_returns_finite_loss_and_gradients_flow():
    vlm = _build_vlm()
    input_ids = torch.tensor([[5, IMAGE_TOKEN_ID, 7, 8, 9]], dtype=torch.long)
    labels = torch.tensor([[-100, -100, 7, 8, 9]], dtype=torch.long)
    out = vlm([object()], input_ids, labels=labels)
    assert out.loss is not None
    assert torch.isfinite(out.loss).item()
    out.loss.backward()
    # Projector gets gradients (it's between encoder output and LLM input).
    assert vlm.projector.lin.weight.grad is not None
    assert torch.isfinite(vlm.projector.lin.weight.grad).all()
