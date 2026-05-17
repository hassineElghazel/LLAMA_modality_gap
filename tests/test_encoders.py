"""Encoder interface contract tests.

The actual ``LLM2CLIPEncoder`` requires HuggingFace weights to load — those
checks are gated by the ``slow`` and ``gpu`` markers. Quick CI runs only
exercise the abstract interface contract.
"""
from __future__ import annotations

import pytest

from src.encoders.base import Encoder
from src.encoders.llm2clip_encoder import LLM2CLIPConfig, LLM2CLIPEncoder


def test_llm2clip_implements_encoder_interface():
    enc = LLM2CLIPEncoder(LLM2CLIPConfig(device="cpu"))
    assert isinstance(enc, Encoder)
    assert enc.output_dim == 768
    # Before any forward, the property returns the expected (canonical) value.
    assert enc.vision_hidden_dim == 1024


def test_encoder_methods_require_load():
    enc = LLM2CLIPEncoder(LLM2CLIPConfig(device="cpu"))
    with pytest.raises(RuntimeError):
        enc.encode_image([])
    with pytest.raises(RuntimeError):
        enc.encode_text(["hi"])
    with pytest.raises(RuntimeError):
        enc.encode_image_tokens([])


def test_two_distinct_model_ids():
    """LLM2CLIP has separate vision and text checkpoints — config must keep them
    distinct from the captioning LLM backbone (which lives in configs/llm.yaml)."""
    cfg = LLM2CLIPConfig()
    assert cfg.vision_hf_id == "microsoft/LLM2CLIP-Openai-L-14-336"
    assert cfg.text_hf_id == "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"
    assert cfg.vision_hf_id != cfg.text_hf_id
    # The text-side workaround id must be the canonical Meta-Llama id (LLM2Vec
    # rejects anything else).
    assert cfg.llm2vec_name_workaround == "meta-llama/Meta-Llama-3-8B-Instruct"


@pytest.mark.slow
@pytest.mark.gpu
def test_llm2clip_smoke_load():
    """Real HF load — runs only with the 'slow' marker enabled.

    On first run this also probes the ACTUAL vision_hidden_dim. If it differs
    from 1024 (the canonical CLIP ViT-L value), update encoders.yaml +
    ProjectorConfig.in_dim to match the probed value before any training.
    """
    from PIL import Image
    enc = LLM2CLIPEncoder(LLM2CLIPConfig(device="cuda")).load()
    assert enc.output_dim == 768
    dummy = Image.new("RGB", (336, 336), (0, 0, 0))
    tokens = enc.encode_image_tokens([dummy])
    assert tokens.shape == (1, 576, enc.vision_hidden_dim)
