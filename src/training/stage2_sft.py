"""Stage 2: full visual instruction tuning on InternVL-Chat-V1.2-SFT.

Full-parameter LLM fine-tuning. A100 required. Fallback: LoRA r=16 a=32 on
attention + MLP projections fits on 24GB.

Skeleton — like stage1, the data collator and forward construction are
delegated to the caller because they depend on the InternVL conversation schema
(verified at first integration).
"""
from __future__ import annotations

from pathlib import Path

import torch
from rich.console import Console

from ..models.vlm import VLM
from .trainer_utils import build_adamw, cosine_with_warmup, freeze_module

console = Console()


def train_stage2(
    vlm: VLM,
    sft_dataloader,
    cfg: dict,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vlm.to(device)

    if cfg["freeze"]["encoder"]:
        freeze_module(vlm.encoder)
    if cfg.get("lora_fallback", {}).get("enabled"):
        from peft import LoraConfig, get_peft_model
        lora_cfg = cfg["lora_fallback"]
        peft_cfg = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        vlm._llm = get_peft_model(vlm._llm, peft_cfg)
        console.log("[stage2] LoRA fallback enabled")

    trainable = [p for p in vlm.parameters() if p.requires_grad]
    opt_cfg = cfg["optimizer"]
    sched_cfg = cfg["schedule"]
    total_steps = len(sft_dataloader) * sched_cfg["num_epochs"]
    warmup_steps = int(total_steps * sched_cfg["warmup_ratio"])
    optimizer = build_adamw(
        trainable, lr=opt_cfg["lr"], wd=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]), eps=opt_cfg["eps"],
    )
    scheduler = cosine_with_warmup(optimizer, warmup_steps, total_steps)

    log_every = cfg["logging"]["log_every_steps"]
    save_every = cfg["logging"]["save_every_steps"]
    ckpt_path = Path(cfg["output"]["checkpoint_path"])

    step = 0
    for epoch in range(sched_cfg["num_epochs"]):
        for batch in sft_dataloader:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = vlm(batch["images"], batch["input_ids"], labels=batch["labels"])
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % log_every == 0:
                console.log(f"[stage2] epoch={epoch} step={step} loss={out.loss.item():.4f}")
            if save_every and step % save_every == 0:
                torch.save({"vlm_state_dict": vlm.state_dict()}, ckpt_path)

    torch.save({"vlm_state_dict": vlm.state_dict()}, ckpt_path)
    return ckpt_path
