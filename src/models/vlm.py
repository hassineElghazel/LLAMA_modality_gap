"""VLM assembly: encoder + projector + Llama-3.

Standard LLaVA-style splice: tokenize the prompt around an ``<image>``
placeholder, run the image through encoder + projector to get
(num_visual_tokens, hidden) projected tokens, splice those into the LLM input
embedding sequence at the placeholder position.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..encoders.base import Encoder
from .projector import MLP2xGELU


@dataclass
class VLMConfig:
    llm_hf_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    image_token: str = "<image>"
    weights_dtype: str = "bfloat16"
    device: str = "cuda"


class VLM(nn.Module):
    """Encoder + projector + Llama-3 wrapper.

    Forward signature is intentionally minimal — callers pass either a
    ``forward(images, input_ids, labels)`` for training or use ``generate``
    for inference. Heavy lifting (image-token splicing) lives in
    ``_build_input_embeddings``.
    """

    def __init__(
        self,
        encoder: Encoder,
        projector: MLP2xGELU,
        cfg: Optional[VLMConfig] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.cfg = cfg or VLMConfig()
        self._llm = None
        self._tokenizer = None
        self._image_token_id: Optional[int] = None

    def load_llm(self) -> "VLM":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self.cfg.weights_dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.llm_hf_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Add image placeholder as a special token so it survives tokenization
        # as a single contiguous token id.
        self._tokenizer.add_special_tokens({"additional_special_tokens": [self.cfg.image_token]})
        self._image_token_id = self._tokenizer.convert_tokens_to_ids(self.cfg.image_token)

        self._llm = AutoModelForCausalLM.from_pretrained(
            self.cfg.llm_hf_id, torch_dtype=dtype
        ).to(self.cfg.device)
        # Resize embeddings to account for the new special token.
        self._llm.resize_token_embeddings(len(self._tokenizer))
        return self

    # ---------- core splice ----------

    def _build_input_embeddings(
        self, images, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace each <image> token in input_ids with the projected visual
        token sequence. Returns (inputs_embeds, attention_mask).

        Constraint: every row of input_ids must contain exactly one image
        placeholder. (Multi-image not supported in this phase.)
        """
        assert self._llm is not None, "call .load_llm() first"
        assert self._image_token_id is not None
        embed_layer = self._llm.get_input_embeddings()

        # Visual tokens: (B, num_visual_tokens, hidden_llm)
        with torch.no_grad():
            vis_tokens = self.encoder.encode_image_tokens(images)
        proj_tokens = self.projector(vis_tokens.to(next(self.projector.parameters()).dtype))

        B, L = input_ids.shape
        text_embeds = embed_layer(input_ids)

        new_rows, new_masks = [], []
        for b in range(B):
            ids = input_ids[b]
            pos = (ids == self._image_token_id).nonzero(as_tuple=True)[0]
            if pos.numel() != 1:
                raise ValueError(
                    f"row {b}: expected exactly 1 image token, found {pos.numel()}"
                )
            i = int(pos.item())
            row = torch.cat([text_embeds[b, :i], proj_tokens[b], text_embeds[b, i + 1 :]], dim=0)
            mask = torch.ones(row.shape[0], dtype=torch.long, device=row.device)
            new_rows.append(row)
            new_masks.append(mask)

        # Right-pad rows to common length.
        max_len = max(r.shape[0] for r in new_rows)
        hidden = new_rows[0].shape[-1]
        padded = torch.zeros(B, max_len, hidden, dtype=new_rows[0].dtype, device=new_rows[0].device)
        att = torch.zeros(B, max_len, dtype=torch.long, device=new_rows[0].device)
        for b, (r, m) in enumerate(zip(new_rows, new_masks)):
            padded[b, : r.shape[0]] = r
            att[b, : m.shape[0]] = m
        return padded, att

    # ---------- forward / generate ----------

    def forward(
        self,
        images,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        inputs_embeds, attention_mask = self._build_input_embeddings(images, input_ids)
        return self._llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, images, prompt: str, **gen_kwargs) -> list[str]:
        assert self._llm is not None and self._tokenizer is not None, "call .load_llm() first"
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt)
        enc = self._tokenizer(prompts, return_tensors="pt", padding=True).to(self.cfg.device)
        inputs_embeds, attention_mask = self._build_input_embeddings(images, enc["input_ids"])
        out_ids = self._llm.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_kwargs
        )
        return self._tokenizer.batch_decode(out_ids, skip_special_tokens=True)
