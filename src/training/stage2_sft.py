"""Stage 2: autoregressive image captioning with LoRA fine-tuning.

Per Overleaf spec §4:
- CLIP ViT-L/14 frozen.
- Connector refined (loaded from Stage 1 checkpoint for C3, random for C1).
- LLaMA-2-7B trained via LoRA adapters (base weights frozen).
- AR cross-entropy on text positions only; visual-token positions masked.

The data side is LLaVA-Instruct-150K: each item has an image and a
conversation. The collator flattens the conversation into a single
prompt + response and produces ``input_ids`` / ``labels`` with non-response
positions set to -100. ``VLM._build_inputs`` further masks the visual
positions when expanding the ``<image>`` placeholder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
from rich.console import Console

from ..models.vlm import VLM
from ..utils import notify
from .trainer_utils import build_adamw, cosine_with_warmup, freeze_module

console = Console()


def _apply_lora(vlm: VLM, lora_cfg: dict) -> VLM:
    from peft import LoraConfig, get_peft_model

    peft_cfg = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias=str(lora_cfg.get("bias", "none")),
        task_type="CAUSAL_LM",
    )
    vlm._llm = get_peft_model(vlm._llm, peft_cfg)
    console.log(
        f"[stage2] LoRA enabled (r={peft_cfg.r}, alpha={peft_cfg.lora_alpha}, "
        f"targets={peft_cfg.target_modules})"
    )
    return vlm


def train_stage2(
    vlm: VLM,
    dataloader: Iterable,
    cfg: dict,
    *,
    max_steps: Optional[int] = None,
    progress_cb: Optional[Callable[[int, float], None]] = None,
) -> Path:
    """LoRA Stage 2 training loop.

    Expects ``dataloader`` to yield batches shaped
    ``{"images": list[PIL.Image], "input_ids": LongTensor, "labels": LongTensor}``
    where the (input_ids, labels) sequences are pre-formatted by the LLaVA
    collator (see ``scripts/06_train_stage2.py::_llava_collate``).
    """
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype_amp = torch.bfloat16 if cfg["precision"]["amp"] == "bf16" else torch.float32
    vlm.to(device)

    if cfg["freeze"].get("vit", True):
        freeze_module(vlm.encoder._vision)
    if cfg["freeze"].get("connector", False):
        freeze_module(vlm.projector)
    else:
        for p in vlm.projector.parameters():
            p.requires_grad = True
        vlm.projector.train()

    lora_cfg = cfg.get("lora") or {}
    if lora_cfg.get("enabled", True):
        vlm = _apply_lora(vlm, lora_cfg)
    if cfg["freeze"].get("llm", True):
        # Freeze the base LLM weights; PEFT keeps adapter params trainable.
        for n, p in vlm._llm.named_parameters():
            if "lora" not in n.lower():
                p.requires_grad = False

    trainable = [p for p in vlm.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    console.log(f"[stage2] trainable params: {n_trainable / 1e6:.2f}M")

    opt_cfg = cfg["optimizer"]
    sched_cfg = cfg["schedule"]
    # Precedence: max_steps (smoke runs) > cfg["total_steps"] (injected by the
    # launcher when the dataset size is known) > dataloader __len__ > 1000.
    # Required because the cosine schedule oscillates back up if step > total_steps.
    if max_steps is not None:
        total_steps = max_steps
    elif cfg.get("total_steps") is not None:
        total_steps = int(cfg["total_steps"])
    else:
        n_batches = len(dataloader) if hasattr(dataloader, "__len__") else 1000
        total_steps = n_batches * sched_cfg["num_epochs"]
    warmup_steps = int(total_steps * sched_cfg["warmup_ratio"])
    optimizer = build_adamw(
        trainable, lr=opt_cfg["lr"], wd=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]), eps=opt_cfg["eps"],
    )
    scheduler = cosine_with_warmup(optimizer, warmup_steps, total_steps)

    log_every = cfg["logging"]["log_every_steps"]
    save_every = cfg["logging"]["save_every_steps"]
    notify_every = cfg["logging"].get("notify_every_steps", 0)
    ckpt_path = Path(cfg["output"]["checkpoint_path"])
    accum = max(1, int(cfg["batch"].get("gradient_accumulation_steps", 1)))

    step = 0
    done = False
    for epoch in range(sched_cfg["num_epochs"]):
        if done:
            break
        optimizer.zero_grad(set_to_none=True)
        for micro_idx, batch in enumerate(dataloader):
            with torch.amp.autocast("cuda", dtype=dtype_amp, enabled=torch.cuda.is_available()):
                out = vlm(
                    batch["images"],
                    batch["input_ids"].to(device),
                    labels=batch["labels"].to(device),
                )
            loss = out.loss / accum
            loss.backward()

            if (micro_idx + 1) % accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % log_every == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    console.log(
                        f"[stage2] epoch={epoch} step={step}/{total_steps} "
                        f"loss={out.loss.item():.4f} lr={lr_now:.2e}"
                    )
                    if notify_every and step % notify_every == 0:
                        pct = 100 * step / total_steps
                        notify.send(
                            f"[Stage2 C1] step {step}/{total_steps} ({pct:.0f}%)\n"
                            f"loss={out.loss.item():.4f}  lr={lr_now:.2e}"
                        )
                if progress_cb is not None:
                    progress_cb(step, float(out.loss.item()))
                if save_every and step % save_every == 0:
                    _save_vlm(vlm, ckpt_path)
                if max_steps is not None and step >= max_steps:
                    done = True
                    break

    _save_vlm(vlm, ckpt_path)
    return ckpt_path


def _save_vlm(vlm: VLM, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Store connector weights + LoRA adapters separately; the base LLM is
    # the upstream HF checkpoint and not duplicated here.
    blob = {
        "connector": vlm.projector.state_dict(),
        "llm_trainable": {
            n: p.detach().cpu() for n, p in vlm._llm.named_parameters() if p.requires_grad
        },
    }
    torch.save(blob, path)
