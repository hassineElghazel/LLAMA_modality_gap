"""Smoke tests for the VLMEvalKit adapter's message-parsing logic.

Tests ``LlamaConnectorVLM._parse_message`` against the three message shapes
the adapter must accept, plus the error-path cases (missing image, unknown
shape). A mock ``VLM.generate`` verifies the dispatch wiring without loading
any model weights.
"""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image_file(tmp_path: Path, name: str = "img.jpg") -> Path:
    """Write a tiny valid JPEG to tmp_path and return its path."""
    p = tmp_path / name
    img = Image.new("RGB", (4, 4), color=(128, 64, 32))
    img.save(p, format="JPEG")
    return p


# ---------------------------------------------------------------------------
# _parse_message: three accepted shapes
# ---------------------------------------------------------------------------


def test_parse_message_dict_shape(tmp_path):
    """``{"image": path, "text": "..."}`` dict is parsed correctly."""
    from src.evaluation.vlmevalkit_adapter import LlamaConnectorVLM

    img_path = _make_image_file(tmp_path)
    msg = {"image": str(img_path), "text": "What is in the image?"}

    img, prompt = LlamaConnectorVLM._parse_message(msg)

    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert "What is in the image?" in prompt
    assert prompt.startswith("<image>")


def test_parse_message_list_of_dicts_shape(tmp_path):
    """VLMEvalKit canonical ``[{"type": "image", ...}, {"type": "text", ...}]`` list."""
    from src.evaluation.vlmevalkit_adapter import LlamaConnectorVLM

    img_path = _make_image_file(tmp_path)
    msg = [
        {"type": "image", "value": str(img_path)},
        {"type": "text", "value": "Describe the scene."},
    ]

    img, prompt = LlamaConnectorVLM._parse_message(msg)

    assert isinstance(img, Image.Image)
    assert "Describe the scene." in prompt
    assert prompt.startswith("<image>")


def test_parse_message_tuple_shape(tmp_path):
    """``(image_path, text)`` 2-tuple is parsed correctly."""
    from src.evaluation.vlmevalkit_adapter import LlamaConnectorVLM

    img_path = _make_image_file(tmp_path)
    msg = (str(img_path), "Is there a dog?")

    img, prompt = LlamaConnectorVLM._parse_message(msg)

    assert isinstance(img, Image.Image)
    assert "Is there a dog?" in prompt
    assert prompt.startswith("<image>")


# ---------------------------------------------------------------------------
# _parse_message: error paths
# ---------------------------------------------------------------------------


def test_parse_message_missing_image_raises():
    """Dict without an image key raises ValueError."""
    from src.evaluation.vlmevalkit_adapter import LlamaConnectorVLM

    with pytest.raises(ValueError, match="no image part"):
        LlamaConnectorVLM._parse_message({"text": "no image here"})


def test_parse_message_unknown_shape_raises():
    """An unrecognised message type raises ValueError."""
    from src.evaluation.vlmevalkit_adapter import LlamaConnectorVLM

    with pytest.raises(ValueError, match="unrecognised"):
        LlamaConnectorVLM._parse_message(42)


# ---------------------------------------------------------------------------
# generate: dispatch wiring (mock VLM, no weights loaded)
# ---------------------------------------------------------------------------


def test_generate_calls_vlm_and_returns_string(tmp_path):
    """``generate`` calls ``vlm.generate`` and returns the first element."""
    from src.evaluation.vlmevalkit_adapter import LlamaConnectorVLM

    img_path = _make_image_file(tmp_path)

    # Build a minimal adapter instance without loading real weights.
    adapter = object.__new__(LlamaConnectorVLM)
    adapter.gen_kwargs = {"max_new_tokens": 20}
    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = ["a cute cat sitting on a mat"]
    adapter.vlm = mock_vlm

    msg = [
        {"type": "image", "value": str(img_path)},
        {"type": "text", "value": "Describe."},
    ]
    result = adapter.generate(msg)

    assert result == "a cute cat sitting on a mat"
    mock_vlm.generate.assert_called_once()
    call_kwargs = mock_vlm.generate.call_args
    # First positional arg is the list containing the PIL image.
    images_arg = call_kwargs[0][0]
    assert len(images_arg) == 1
    assert isinstance(images_arg[0], Image.Image)
