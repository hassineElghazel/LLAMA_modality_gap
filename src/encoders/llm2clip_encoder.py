"""LLM2CLIP encoder wrapper.

LLM2CLIP has a non-trivial two-checkpoint architecture:
  - vision tower: ``microsoft/LLM2CLIP-Openai-L-14-336``
  - text tower:   ``microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned``,
                  which is a CC-finetuned Llama-3 wrapped via the ``llm2vec``
                  library to produce text features that the vision tower's
                  contrastive head consumes.

This wrapper mirrors the reference ``embed.py`` pipeline (see
``references/ReVision/embed.py``):

  text  -> LLM2Vec(text-tower, mean pool) -> llm_feats
        -> vision_tower.get_text_features(llm_feats) -> 768-dim
        -> L2-normalize

  image -> CLIPImageProcessor (fallback to plain CLIP ViT-L/14-336 because
           LLM2CLIP-Openai-L-14-336 does not ship its own preprocessor)
        -> vision_tower.get_image_features(...) -> 768-dim
        -> L2-normalize

For projected-token-space diagnostics (§7.5), ``encode_image_tokens`` returns
per-token vision-tower features (B, num_visual_tokens, vision_hidden_dim);
this is what the projector ingests.

Embeddings handed to the gap-diagnostic metrics are cast to Float64 at the
extraction-script boundary (see ``src/diagnostics/extract_embeddings.py``) to
avoid the Float32 ~1e-8 error floor (ReAlign Appendix E.2).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
from PIL import Image

from .base import Encoder


@dataclass
class LLM2CLIPConfig:
    # Vision tower
    vision_hf_id: str = "microsoft/LLM2CLIP-Openai-L-14-336"
    image_size: int = 336
    num_visual_tokens: int = 576
    expected_vision_hidden_dim: int = 1024
    contrastive_dim: int = 768

    # Text tower (LLM2CLIP CC-finetuned Llama-3, wrapped via LLM2Vec).
    text_hf_id: str = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"
    # Workaround: LLM2Vec only accepts the canonical Meta-Llama id; we patch
    # the loaded config to satisfy it (per reference embed.py:130-132).
    llm2vec_name_workaround: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    text_pooling_mode: str = "mean"
    text_max_length: int = 512
    text_doc_max_length: int = 512

    # Image processor fallback for when the vision repo doesn't ship one.
    image_processor_fallback_hf_id: str = "openai/clip-vit-large-patch14-336"

    device: str = "cuda"
    weights_dtype: str = "bfloat16"


class LLM2CLIPEncoder(Encoder):
    def __init__(self, cfg: Optional[LLM2CLIPConfig] = None):
        self.cfg = cfg or LLM2CLIPConfig()
        self._vision = None
        self._text_llm = None
        self._text_l2v = None
        self._image_processor = None
        # Probed on first forward — see _probe_vision_hidden_dim.
        self._vision_hidden_dim: Optional[int] = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def load(self) -> "LLM2CLIPEncoder":
        from transformers import AutoConfig, AutoModel, AutoTokenizer, CLIPImageProcessor

        dtype = getattr(torch, self.cfg.weights_dtype)

        # ---- vision tower -------------------------------------------------
        self._vision = AutoModel.from_pretrained(
            self.cfg.vision_hf_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.cfg.device).eval()

        # ---- image processor (with fallback) ------------------------------
        try:
            self._image_processor = CLIPImageProcessor.from_pretrained(self.cfg.vision_hf_id)
        except Exception as e:   # noqa: BLE001
            warnings.warn(
                f"Could not load preprocessor from {self.cfg.vision_hf_id} ({e}). "
                f"Falling back to {self.cfg.image_processor_fallback_hf_id}."
            )
            self._image_processor = CLIPImageProcessor.from_pretrained(
                self.cfg.image_processor_fallback_hf_id
            )

        # ---- text tower (LLM2Vec-wrapped CC-finetuned Llama-3) -----------
        from llm2vec import LLM2Vec   # heavy import deferred

        text_cfg = AutoConfig.from_pretrained(self.cfg.text_hf_id, trust_remote_code=True)
        # LLM2Vec only accepts the canonical Llama-3 id — patch in place.
        if getattr(text_cfg, "_name_or_path", "") != self.cfg.llm2vec_name_workaround:
            text_cfg._name_or_path = self.cfg.llm2vec_name_workaround
        try:
            text_cfg._attn_implementation = "sdpa"
        except Exception:
            pass

        self._text_llm = AutoModel.from_pretrained(
            self.cfg.text_hf_id,
            torch_dtype=dtype,
            config=text_cfg,
            trust_remote_code=True,
        ).to(self.cfg.device).eval()
        text_tok = AutoTokenizer.from_pretrained(self.cfg.text_hf_id, trust_remote_code=True)

        self._text_l2v = LLM2Vec(
            self._text_llm,
            text_tok,
            pooling_mode=self.cfg.text_pooling_mode,
            max_length=self.cfg.text_max_length,
            doc_max_length=self.cfg.text_doc_max_length,
        )
        return self

    def _require_loaded(self):
        if self._vision is None:
            raise RuntimeError("Encoder not loaded — call .load() first")

    # ------------------------------------------------------------------
    # Encoder interface
    # ------------------------------------------------------------------

    @property
    def output_dim(self) -> int:
        return self.cfg.contrastive_dim

    @property
    def vision_hidden_dim(self) -> int:
        # Returns the expected value before the first probe. After the first
        # forward through the vision tower, returns the actual value.
        return self._vision_hidden_dim or self.cfg.expected_vision_hidden_dim

    # ------------------------------------------------------------------
    # image side
    # ------------------------------------------------------------------

    def _preprocess_images(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        return self._image_processor(images=images, return_tensors="pt").to(self.cfg.device)

    @torch.no_grad()
    def encode_image(self, images) -> torch.Tensor:
        """L2-normalized 768-dim contrastive image embeddings."""
        self._require_loaded()
        inputs = self._preprocess_images(images)
        feats = self._vision.get_image_features(**inputs)
        den = feats.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (feats / den.to(dtype=feats.dtype))

    @torch.no_grad()
    def encode_image_tokens(self, images) -> torch.Tensor:
        """Per-token vision-tower features (B, num_visual_tokens, vision_hidden_dim).

        The exact API for pre-projection ViT token features depends on the
        LLM2CLIP repo's vision_model interface; we use ``last_hidden_state``
        and drop the CLS token. The reference repo accesses these inside
        bunny/model/bunny_arch.py — verify shapes there if anything breaks.
        """
        self._require_loaded()
        inputs = self._preprocess_images(images)
        vision_out = self._vision.vision_model(**inputs, output_hidden_states=False)
        tokens = vision_out.last_hidden_state[:, 1:, :]   # drop CLS
        self._probe_vision_hidden_dim(tokens.shape[-1])
        if tokens.shape[1] != self.cfg.num_visual_tokens:
            raise RuntimeError(
                f"expected {self.cfg.num_visual_tokens} visual tokens, got {tokens.shape[1]}"
            )
        return tokens

    def _probe_vision_hidden_dim(self, observed: int) -> None:
        if self._vision_hidden_dim is None:
            self._vision_hidden_dim = observed
            if observed != self.cfg.expected_vision_hidden_dim:
                warnings.warn(
                    f"vision_hidden_dim probed={observed}, expected={self.cfg.expected_vision_hidden_dim}. "
                    f"Update configs/encoders.yaml::vision_model.expected_vision_hidden_dim "
                    f"and ProjectorConfig.in_dim to match before training."
                )

    # ------------------------------------------------------------------
    # text side (LLM2Vec -> contrastive head)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_text(self, texts) -> torch.Tensor:
        """L2-normalized 768-dim contrastive text embeddings via LLM2Vec.

        Pipeline (per reference embed.py:180-189):
          texts -> LLM2Vec(text-tower, mean pool) -> llm_feats
                -> vision_tower.get_text_features(llm_feats) -> 768-dim
                -> L2-normalize
        """
        self._require_loaded()
        if isinstance(texts, str):
            texts = [texts]

        llm_feats = self._text_l2v.encode(list(texts), convert_to_tensor=True)
        llm_feats = llm_feats.to(device=self.cfg.device, dtype=next(self._vision.parameters()).dtype)

        clip_feats = self._vision.get_text_features(llm_feats)
        den = clip_feats.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return clip_feats / den.to(dtype=clip_feats.dtype)
