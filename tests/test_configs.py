"""Smoke tests for the spec-critical values in all project YAML configs.

Each config must:
1. Load cleanly via the project's own ``load_yaml`` utility.
2. Contain the architecture-critical keys required by the spec (image size,
   token counts, hidden dims, HF model IDs) so that a mistyped value is caught
   before any expensive download or training run starts.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.io import load_yaml


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(filename: str) -> dict:
    return load_yaml(CONFIGS_DIR / filename)


# ---------------------------------------------------------------------------
# configs/encoders.yaml
# ---------------------------------------------------------------------------

def test_encoders_yaml_loads():
    cfg = _load("encoders.yaml")
    vm = cfg["vision_model"]
    assert vm["hf_id"] == "openai/clip-vit-large-patch14"
    assert vm["image_size"] == 224
    assert vm["num_visual_tokens"] == 257
    assert vm["vision_hidden_dim"] == 1024


# ---------------------------------------------------------------------------
# configs/llm.yaml
# ---------------------------------------------------------------------------

def test_llm_yaml_loads():
    cfg = _load("llm.yaml")
    model = cfg["model"]
    assert model["hf_id"] == "meta-llama/Llama-2-7b-hf"
    assert model["hidden_size"] == 4096
    assert "<image>" in model["image_token"]


# ---------------------------------------------------------------------------
# configs/projector.yaml
# ---------------------------------------------------------------------------

def test_projector_yaml_loads():
    cfg = _load("projector.yaml")
    arch = cfg["architecture"]
    assert arch["in_dim"] == 1024    # CLIP ViT-L/14 hidden dim
    assert arch["hidden_dim"] == 4096
    assert arch["out_dim"] == 4096   # LLaMA-2-7B hidden size
    assert cfg["sequence"]["num_visual_tokens"] == 257


# ---------------------------------------------------------------------------
# configs/training_stage1.yaml
# ---------------------------------------------------------------------------

def test_training_stage1_yaml_loads():
    cfg = _load("training_stage1.yaml")
    assert cfg["data"]["dataset"] == "bunny_v1_1"
    loss = cfg["loss"]
    assert loss["type"] == "infonce_symmetric"
    assert abs(loss["temperature_init"] - 0.07) < 1e-9
    assert loss["temperature_learnable"] is True
    freeze = cfg["freeze"]
    assert freeze["llm_embed"] is True
    assert freeze["vit"] is True
    assert freeze["connector"] is False


# ---------------------------------------------------------------------------
# configs/training_stage2.yaml
# ---------------------------------------------------------------------------

def test_training_stage2_yaml_loads():
    cfg = _load("training_stage2.yaml")
    assert cfg["data"]["dataset"] == "llava_instruct_150k"
    lora = cfg["lora"]
    assert lora["enabled"] is True
    assert lora["r"] == 16
    assert lora["alpha"] == 32
    assert abs(lora["dropout"] - 0.05) < 1e-9
    targets = set(lora["target_modules"])
    for mod in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert mod in targets, f"LoRA target missing: {mod}"
    freeze = cfg["freeze"]
    assert freeze["vit"] is True


# ---------------------------------------------------------------------------
# configs/captioning.yaml
# ---------------------------------------------------------------------------

def test_captioning_yaml_loads():
    cfg = _load("captioning.yaml")
    es = cfg["eval_set"]
    assert es["name"] == "coco_val2017"
    assert es["num_images"] == 5000
    gen = cfg["generation"]
    assert gen["max_new_tokens"] > 0


# ---------------------------------------------------------------------------
# configs/data.yaml
# ---------------------------------------------------------------------------

def test_data_yaml_loads():
    cfg = _load("data.yaml")
    assert cfg["coco_val2017"]["expected_num_images"] == 5000
    bunny = cfg["bunny_v1_1"]
    assert "BoyaWu10" in bunny["hf_repo"]
    llava = cfg["llava_instruct_150k"]
    assert "liuhaotian" in llava["hf_repo"]
    diag = cfg["diagnostic_sample"]
    assert diag["num_pairs"] > 0
    assert diag["seed"] == 42
