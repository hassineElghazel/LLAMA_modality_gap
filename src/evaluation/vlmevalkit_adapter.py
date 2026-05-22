"""Adapter exposing our CLIP+connector+LLaMA-2 VLM to VLMEvalKit.

VLMEvalKit (``vlmeval`` package) expects models to implement a ``generate``
method that consumes a list of multimodal "message" parts of the form
``{"type": "image", "value": <path>}`` / ``{"type": "text", "value": <str>}``
and returns a plain string answer. This adapter:

1. Reconstructs the VLM (encoder + connector + LLaMA-2-7B + LoRA) from a
   Stage-2 checkpoint (or C0/C2 zero-shot configuration).
2. Translates a VLMEvalKit message into a (PIL image, prompt) pair.
3. Calls ``VLM.generate`` and returns the decoded string.

The exact VLMEvalKit ``BaseModel`` API shifts across releases; we keep the
glue minimal and tolerant by mirroring the interface used in their reference
custom-model examples.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

from ..encoders.clip_encoder import build_clip_encoder
from ..models.checkpoint import load_projector
from ..models.projector import build_projector
from ..models.vlm import VLM, VLMConfig
from ..utils.io import load_yaml


@dataclass
class AdapterConfig:
    encoders_yaml: str = "configs/encoders.yaml"
    projector_yaml: str = "configs/projector.yaml"
    llm_yaml: str = "configs/llm.yaml"
    captioning_yaml: str = "configs/captioning.yaml"


class LlamaConnectorVLM:
    """VLMEvalKit-compatible wrapper around our VLM.

    Args:
        vlm_checkpoint: path to a Stage-2 checkpoint (or "random" for C0).
        lora_target_modules: required to recreate the LoRA-wrapped LLM when
            loading from a Stage-2 checkpoint. If None, the LLM is loaded
            without LoRA (suitable for C0 / C2 zero-shot).
    """

    INSTALL_REQ = False    # we don't pull anything from VLMEvalKit's model zoo

    def __init__(
        self,
        vlm_checkpoint: str | None = None,
        adapter_cfg: Optional[AdapterConfig] = None,
        lora_cfg: Optional[dict] = None,
        device: Optional[str] = None,
    ):
        self.adapter_cfg = adapter_cfg or AdapterConfig()
        enc_cfg = load_yaml(self.adapter_cfg.encoders_yaml)
        proj_cfg = load_yaml(self.adapter_cfg.projector_yaml)
        llm_cfg = load_yaml(self.adapter_cfg.llm_yaml)
        self.cap_cfg = load_yaml(self.adapter_cfg.captioning_yaml)
        self.device = device or enc_cfg["inference"]["device"]
        self.gen_kwargs = self.cap_cfg["generation"]

        encoder = build_clip_encoder(enc_cfg).load()

        if vlm_checkpoint is None or str(vlm_checkpoint).lower() == "random":
            connector = build_projector(proj_cfg["architecture"]).to(self.device)
        else:
            blob = torch.load(vlm_checkpoint, map_location="cpu")
            if "config" in blob:
                connector = load_projector(vlm_checkpoint).to(self.device)
            else:
                connector = build_projector(proj_cfg["architecture"])
                connector.load_state_dict(blob["connector"])
                connector = connector.to(self.device)

        self.vlm = VLM(encoder, connector, VLMConfig(
            llm_hf_id=llm_cfg["model"]["hf_id"],
            weights_dtype=llm_cfg["dtype"]["weights"],
            device=self.device,
        )).load_llm()

        # If checkpoint contains LoRA params, wrap and load them.
        if vlm_checkpoint and str(vlm_checkpoint).lower() != "random":
            blob = torch.load(vlm_checkpoint, map_location="cpu")
            llm_trainable = blob.get("llm_trainable") or {}
            if llm_trainable and lora_cfg:
                from peft import LoraConfig, get_peft_model
                peft_cfg = LoraConfig(
                    r=int(lora_cfg["r"]),
                    lora_alpha=int(lora_cfg["alpha"]),
                    lora_dropout=float(lora_cfg["dropout"]),
                    target_modules=list(lora_cfg["target_modules"]),
                    bias=str(lora_cfg.get("bias", "none")),
                    task_type="CAUSAL_LM",
                )
                self.vlm._llm = get_peft_model(self.vlm._llm, peft_cfg)
                missing, unexpected = self.vlm._llm.load_state_dict(llm_trainable, strict=False)
                if missing:
                    print(f"[adapter] LoRA load missing keys: {len(missing)} (expected: base weights)")
                if unexpected:
                    print(f"[adapter] LoRA load unexpected keys: {unexpected[:5]}")

        self.vlm.eval()

    # ------------------------------------------------------------------
    # VLMEvalKit interface
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_message(message: Any) -> tuple[Image.Image, str]:
        """Accept VLMEvalKit's ``[{"type": "image", "value": path}, {"type": "text", "value": str}]``
        OR a plain ``(image_path, text)`` tuple OR ``{"image": ..., "text": ...}``.
        """
        image_path = None
        text_parts: list[str] = []
        if isinstance(message, dict):
            image_path = message.get("image") or message.get("image_path")
            if message.get("text"):
                text_parts.append(str(message["text"]))
        elif isinstance(message, (list, tuple)) and message and isinstance(message[0], dict):
            for part in message:
                t = part.get("type")
                v = part.get("value")
                if t == "image":
                    image_path = v
                elif t == "text":
                    text_parts.append(str(v))
        elif isinstance(message, (list, tuple)) and len(message) == 2:
            image_path, txt = message
            text_parts.append(str(txt))
        else:
            raise ValueError(f"unrecognised VLMEvalKit message shape: {type(message).__name__}")
        if image_path is None:
            raise ValueError("VLMEvalKit message contained no image part")
        img = Image.open(image_path).convert("RGB")
        prompt = "<image>\n" + "\n".join([t for t in text_parts if t])
        return img, prompt

    @torch.no_grad()
    def generate(self, message: Any, dataset: Optional[str] = None) -> str:
        img, prompt = self._parse_message(message)
        out = self.vlm.generate([img], [prompt], **self.gen_kwargs)
        return out[0] if isinstance(out, list) else str(out)

    # Some VLMEvalKit code paths call ``generate_inner`` instead of ``generate``.
    generate_inner = generate
