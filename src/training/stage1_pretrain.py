"""Stage 1: contrastive connector pre-training (InfoNCE).

Per Overleaf spec §3:
- **Image side**: image -> Frozen CLIP ViT-L/14 -> CLS token (1024-d) ->
  Connector -> z_img (4096-d).
- **Text side**: caption -> LLaMA-2 tokenizer -> LLaMA embed layer (frozen) ->
  mean-pool over content tokens -> z_txt (4096-d).
- **Loss**: symmetric InfoNCE with learnable temperature, tau init 0.07.

Trains the connector parameters + log_logit_scale only. ViT and LLaMA's embed
layer are frozen targets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
from rich.console import Console
from torch import nn

from ..encoders.clip_encoder import CLIPViTL14Encoder
from ..models.checkpoint import save_projector
from ..models.projector import MLP2xGELU
from .contrastive_loss import LearnableTemperature, symmetric_infonce
from .trainer_utils import build_adamw, cosine_with_warmup, freeze_module

console = Console()


# ----- text-side helper -------------------------------------------------

def encode_text_mean_pool(
    captions: list[str],
    tokenizer,
    embed_layer: nn.Module,
    device: torch.device,
    max_length: int = 64,
) -> torch.Tensor:
    """Mean-pool LLaMA-2 token embeddings over content tokens (excludes pad / BOS / EOS)."""
    enc = tokenizer(
        captions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    embs = embed_layer(enc["input_ids"])               # (B, L, 4096)
    mask = enc["attention_mask"].bool()
    special = torch.zeros_like(mask)
    if tokenizer.bos_token_id is not None:
        special |= enc["input_ids"] == tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        special |= enc["input_ids"] == tokenizer.eos_token_id
    content = mask & ~special                          # (B, L)
    w = content.float().unsqueeze(-1)                  # (B, L, 1)
    denom = w.sum(dim=1).clamp(min=1.0)                # (B, 1)
    return (embs * w).sum(dim=1) / denom               # (B, 4096)


# ----- training loop ----------------------------------------------------

def train_stage1(
    encoder: CLIPViTL14Encoder,
    connector: MLP2xGELU,
    llm_embed: nn.Module,
    tokenizer,
    dataloader: Iterable,
    cfg: dict,
    *,
    max_steps: Optional[int] = None,
    progress_cb: Optional[Callable[[int, float, float], None]] = None,
) -> Path:
    """Run symmetric InfoNCE on (image, caption) batches.

    The ``dataloader`` is expected to yield dicts of the form
    ``{"images": list[PIL.Image], "captions": list[str]}``.
    """
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype_amp = torch.bfloat16 if cfg["precision"]["amp"] == "bf16" else torch.float32

    # Freeze targets. Connector + temperature are the only trainable params.
    freeze_module(llm_embed)
    connector.to(device).train()
    temp = LearnableTemperature(
        temperature_init=cfg["loss"]["temperature_init"],
    ).to(device)
    temp.train()

    trainable_params = list(connector.parameters())
    if cfg["loss"].get("temperature_learnable", True):
        trainable_params += list(temp.parameters())

    opt_cfg = cfg["optimizer"]
    sched_cfg = cfg["schedule"]
    n_batches = len(dataloader) if hasattr(dataloader, "__len__") else (max_steps or 1000)
    total_steps = max_steps or (n_batches * sched_cfg["num_epochs"])
    warmup_steps = int(total_steps * sched_cfg["warmup_ratio"])

    optimizer = build_adamw(
        trainable_params,
        lr=opt_cfg["lr"], wd=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]), eps=opt_cfg["eps"],
    )
    scheduler = cosine_with_warmup(optimizer, warmup_steps, total_steps)

    log_every = cfg["logging"]["log_every_steps"]
    save_every = cfg["logging"]["save_every_steps"]
    ckpt_path = Path(cfg["output"]["checkpoint_path"])

    step = 0
    done = False
    for epoch in range(sched_cfg["num_epochs"]):
        if done:
            break
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            images = batch["images"]
            captions = batch["captions"]

            # Image side: ViT -> CLS (1024) -> connector -> z_img (4096)
            with torch.no_grad():
                vis_tokens = encoder.encode_image_tokens(images)   # (B, 257, 1024)
            cls = vis_tokens[:, 0, :].to(device=device, dtype=next(connector.parameters()).dtype)

            with torch.amp.autocast("cuda", dtype=dtype_amp, enabled=torch.cuda.is_available()):
                z_img = connector(cls)                              # (B, 4096)

                # Text side: tokenize -> embed (frozen) -> mean pool -> z_txt
                with torch.no_grad():
                    z_txt = encode_text_mean_pool(
                        captions, tokenizer, llm_embed, device=device
                    )
                z_txt = z_txt.to(dtype=z_img.dtype)

                loss = symmetric_infonce(z_img, z_txt, temp())

            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if step % log_every == 0:
                console.log(
                    f"[stage1] epoch={epoch} step={step} "
                    f"loss={loss.item():.4f} tau={temp.temperature:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
            if progress_cb is not None:
                progress_cb(step, float(loss.item()), float(temp.temperature))
            if save_every and step % save_every == 0:
                save_projector(connector, ckpt_path)
            if max_steps is not None and step >= max_steps:
                done = True
                break

    save_projector(connector, ckpt_path)
    return ckpt_path
