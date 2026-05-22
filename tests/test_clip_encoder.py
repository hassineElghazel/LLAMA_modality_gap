"""CLIP ViT-L/14 encoder smoke tests.

The fast subset (no `-m gpu`) only exercises the config / build path, since
loading the actual HF weights requires network + GPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.encoders.clip_encoder import CLIPEncoderConfig, CLIPViTL14Encoder, build_clip_encoder


def test_default_config_matches_spec():
    cfg = CLIPEncoderConfig()
    assert cfg.hf_id == "openai/clip-vit-large-patch14"
    assert cfg.image_size == 224
    assert cfg.num_visual_tokens == 257
    assert cfg.vision_hidden_dim == 1024


def test_build_from_yaml_dict_overrides_defaults():
    cfg_dict = {
        "vision_model": {
            "hf_id": "openai/clip-vit-large-patch14",
            "image_size": 224,
            "num_visual_tokens": 257,
            "vision_hidden_dim": 1024,
        },
        "inference": {"device": "cpu", "weights_dtype": "float32"},
    }
    enc = build_clip_encoder(cfg_dict)
    assert isinstance(enc, CLIPViTL14Encoder)
    assert enc.vision_hidden_dim == 1024
    assert enc.num_visual_tokens == 257
    assert enc.image_size == 224
    assert enc.cfg.device == "cpu"


def test_require_loaded_raises_before_load():
    enc = CLIPViTL14Encoder()
    with pytest.raises(RuntimeError, match="not loaded"):
        enc.encode_image_tokens(None)


@pytest.mark.gpu
@pytest.mark.slow
def test_forward_shape_real_weights(tmp_path):
    """Loads real CLIP ViT-L/14 weights; requires HF cache + a CUDA device."""
    from PIL import Image
    import numpy as np

    enc = CLIPViTL14Encoder(
        CLIPEncoderConfig(device="cuda" if torch.cuda.is_available() else "cpu",
                          weights_dtype="float32")
    ).load()
    rng = np.random.default_rng(0)
    img = Image.fromarray((rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)))
    tokens = enc.encode_image_tokens([img, img])
    assert tokens.shape == (2, 257, 1024)
