"""C4: joint autoregressive + contrastive Stage-2 training.

Extends Stage-2 SFT (AR captioning with LoRA) with a contrastive term applied
*during* fine-tuning, to probe whether keeping connector alignment pressure
active changes captioning (a dose-response over lambda). See
``c4_experiment_plan.tex``.

Design (kept identical to the established conditions so C4 stays comparable):
- **AR path** is byte-for-byte the Stage-2 path (same VLM forward, same LoRA
  setup via ``stage2_sft._apply_lora``, same teacher-forced collate). At
  ``lambda=0`` this reduces to C3 exactly.
- **Contrastive path** is byte-for-byte the Stage-1 path: image side is the
  CLS token through the connector, text side is the mean-pooled frozen
  ``embed_tokens`` of the caption, scored with the same ``symmetric_infonce``
  and a learnable temperature. It runs on a *separate* batch (no LLaMA body),
  recovering in-batch negatives cheaply.

Loss (per optimizer step):
    convex   : L = (1-lambda) * L_AR + lambda * L_NCE
    kendall  : L = exp(-s1) L_AR + exp(-s2) L_NCE + 0.5 (s1 + s2)

The AR term is back-propagated per micro-batch (scaled by w_ar / accum) so the
gradient-checkpointed + 4-bit LLaMA forward never has to hold all micro-batches
at once; the contrastive term is back-propagated once per optimizer step.

This module deliberately does NOT modify ``stage2_sft`` — the C0--C3 pipeline
must stay frozen for a clean comparison.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import torch
from rich.console import Console
from torch import nn

from ..models.vlm import VLM
from ..utils import notify
from .contrastive_loss import LearnableTemperature, symmetric_infonce
from .stage1_pretrain import encode_text_mean_pool
from .stage2_sft import _apply_lora, _save_vlm
from .trainer_utils import build_adamw, cosine_with_warmup, freeze_module

console = Console()


def train_stage2_joint(
    vlm: VLM,
    ar_dataloader: Iterable,
    contrastive_iter: Iterator[dict],
    cfg: dict,
    *,
    lambda_contrastive: float,
    use_kendall: bool = False,
    max_steps: Optional[int] = None,
    progress_cb: Optional[Callable[[int, float, float], None]] = None,
    resume_from: Optional[Path] = None,
) -> Path:
    """Joint AR + contrastive Stage-2 loop.

    ``ar_dataloader`` yields the same batches as Stage 2
    (``{"images", "input_ids", "labels"}``). ``contrastive_iter`` is an
    *infinite* iterator yielding ``{"images": list[PIL], "captions": list[str]}``
    of the contrastive batch size; one batch is consumed per optimizer step.
    """
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype_amp = torch.bfloat16 if cfg["precision"]["amp"] == "bf16" else torch.float32
    vlm.to(device)

    # ----- freeze / trainable setup (identical to Stage 2) -----
    if cfg["freeze"].get("vit", True):
        freeze_module(vlm.encoder._vision)
    if cfg["freeze"].get("connector", False):
        freeze_module(vlm.projector)
    else:
        for p in vlm.projector.parameters():
            p.requires_grad = True
        vlm.projector.train()

    if hasattr(vlm._llm, "gradient_checkpointing_enable"):
        if hasattr(vlm._llm, "config"):
            vlm._llm.config.use_cache = False
        vlm._llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(vlm._llm, "enable_input_require_grads"):
            vlm._llm.enable_input_require_grads()
        console.log("[c4] gradient checkpointing enabled (non-reentrant, use_cache=False)")

    lora_cfg = cfg.get("lora") or {}
    if lora_cfg.get("enabled", True):
        vlm = _apply_lora(vlm, lora_cfg)
    if cfg["freeze"].get("llm", True):
        for n, p in vlm._llm.named_parameters():
            if "lora" not in n.lower():
                p.requires_grad = False

    # ----- contrastive head: temperature (+ Kendall log-variances) -----
    c_cfg = cfg.get("contrastive", {})
    temp = LearnableTemperature(
        temperature_init=float(c_cfg.get("temperature_init", 0.07))
    ).to(device)
    temp.train()
    max_cap_len = int(c_cfg.get("max_caption_length", 64))

    s_ar = s_nce = None
    if use_kendall:
        # log-variances; init 0 -> sigma^2 = 1 -> unit weight at start.
        s_ar = nn.Parameter(torch.zeros((), device=device))
        s_nce = nn.Parameter(torch.zeros((), device=device))

    trainable = [p for p in vlm.parameters() if p.requires_grad]
    trainable += list(temp.parameters())
    if use_kendall:
        trainable += [s_ar, s_nce]
    n_trainable = sum(p.numel() for p in trainable)
    console.log(
        f"[c4] trainable params: {n_trainable / 1e6:.2f}M  "
        f"(mode={'kendall' if use_kendall else f'convex lambda={lambda_contrastive}'})"
    )

    vlm.train()
    if cfg["freeze"].get("vit", True):
        vlm.encoder._vision.eval()

    embed_layer = vlm._llm.get_input_embeddings()
    tokenizer = vlm._tokenizer
    conn_dtype = next(vlm.projector.parameters()).dtype

    # ----- optimizer / schedule (Stage-2 settings) -----
    opt_cfg = cfg["optimizer"]
    sched_cfg = cfg["schedule"]
    if max_steps is not None:
        total_steps = max_steps
    elif cfg.get("total_steps") is not None:
        total_steps = int(cfg["total_steps"])
    else:
        n_batches = len(ar_dataloader) if hasattr(ar_dataloader, "__len__") else 1000
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

    # ----- resume (weights + optimizer/scheduler/step, plus contrastive head) -----
    start_step = 0
    if resume_from is not None and Path(resume_from).exists():
        blob = torch.load(str(resume_from), map_location="cpu")
        vlm.projector.load_state_dict(blob["connector"])
        own = dict(vlm._llm.named_parameters())
        with torch.no_grad():
            for n, tensor in blob.get("llm_trainable", {}).items():
                if n in own:
                    own[n].data.copy_(tensor.to(own[n].device, dtype=own[n].dtype))
        if "temp" in blob:
            temp.load_state_dict(blob["temp"])
        if use_kendall and "s_ar" in blob:
            with torch.no_grad():
                s_ar.copy_(torch.as_tensor(blob["s_ar"], device=device))
                s_nce.copy_(torch.as_tensor(blob["s_nce"], device=device))
        if "optimizer" in blob and "scheduler" in blob:
            optimizer.load_state_dict(blob["optimizer"])
            scheduler.load_state_dict(blob["scheduler"])
            start_step = int(blob.get("step", 0))
        console.log(f"[c4] resumed from {resume_from} at step={start_step}/{total_steps}")

    def _weights() -> tuple[torch.Tensor | float, torch.Tensor | float]:
        if use_kendall:
            return torch.exp(-s_ar), torch.exp(-s_nce)
        return (1.0 - lambda_contrastive), lambda_contrastive

    def _lambda_eff() -> float:
        if not use_kendall:
            return float(lambda_contrastive)
        w_ar = float(torch.exp(-s_ar).detach())
        w_nce = float(torch.exp(-s_nce).detach())
        return w_nce / (w_ar + w_nce + 1e-12)

    def _contrastive_step() -> torch.Tensor:
        """One InfoNCE forward on a fresh batch-64 (no LLaMA body)."""
        cbatch = next(contrastive_iter)
        with torch.no_grad():
            vis = vlm.encoder.encode_image_tokens(cbatch["images"])  # (B,257,1024)
        cls = vis[:, 0, :].to(device=device, dtype=conn_dtype)        # CLS token
        with torch.amp.autocast("cuda", dtype=dtype_amp, enabled=torch.cuda.is_available()):
            z_img = vlm.projector(cls)                                # (B,4096)
            with torch.no_grad():
                z_txt = encode_text_mean_pool(
                    cbatch["captions"], tokenizer, embed_layer, device, max_length=max_cap_len
                )
            z_txt = z_txt.to(dtype=z_img.dtype)
            return symmetric_infonce(z_img, z_txt, temp())

    run_contrastive = use_kendall or lambda_contrastive > 0.0

    step = start_step
    done = False
    last_nce = 0.0
    for epoch in range(sched_cfg["num_epochs"]):
        if done:
            break
        optimizer.zero_grad(set_to_none=True)
        for micro_idx, batch in enumerate(ar_dataloader):
            # --- AR micro-batch (scaled by w_ar / accum) ---
            with torch.amp.autocast("cuda", dtype=dtype_amp, enabled=torch.cuda.is_available()):
                out = vlm(
                    batch["images"],
                    batch["input_ids"].to(device),
                    labels=batch["labels"].to(device),
                )
            w_ar, w_nce = _weights()
            (w_ar * out.loss / accum).backward()

            if (micro_idx + 1) % accum == 0:
                # --- contrastive term: one forward+backward per optimizer step ---
                if run_contrastive:
                    l_nce = _contrastive_step()
                    last_nce = float(l_nce.detach())
                    if use_kendall:
                        (w_nce * l_nce + 0.5 * (s_ar + s_nce)).backward()
                    else:
                        (lambda_contrastive * l_nce).backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % log_every == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    console.log(
                        f"[c4] epoch={epoch} step={step}/{total_steps} "
                        f"L_ar={out.loss.item():.4f} L_nce={last_nce:.4f} "
                        f"lambda_eff={_lambda_eff():.3f} tau={temp.temperature:.4f} lr={lr_now:.2e}"
                    )
                    if notify_every and step % notify_every == 0:
                        pct = 100 * step / total_steps
                        notify.send(
                            f"[C4] step {step}/{total_steps} ({pct:.0f}%)\n"
                            f"L_ar={out.loss.item():.4f} L_nce={last_nce:.4f} "
                            f"lambda_eff={_lambda_eff():.3f}"
                        )
                if progress_cb is not None:
                    progress_cb(step, float(out.loss.item()), last_nce)
                if save_every and step % save_every == 0:
                    _save_vlm(vlm, ckpt_path, optimizer=optimizer,
                              scheduler=scheduler, step=step, epoch=epoch)
                if max_steps is not None and step >= max_steps:
                    done = True
                    break

    _save_vlm(vlm, ckpt_path, optimizer=optimizer, scheduler=scheduler,
              step=step, epoch=epoch)

    # Sidecar: record the contrastive-head state so the learned balance is
    # reportable (lambda_eff is the Kendall point overlaid on the sweep curve).
    sidecar = {
        "mode": "kendall" if use_kendall else "convex",
        "lambda_fixed": None if use_kendall else float(lambda_contrastive),
        "lambda_eff": _lambda_eff(),
        "temperature": float(temp.temperature),
        "steps": int(step),
    }
    if use_kendall:
        sidecar["s_ar"] = float(s_ar.detach())
        sidecar["s_nce"] = float(s_nce.detach())
    side_path = Path(str(ckpt_path).replace(".pt", "_contrastive.json"))
    side_path.parent.mkdir(parents=True, exist_ok=True)
    side_path.write_text(json.dumps(sidecar, indent=2))
    console.log(f"[c4] contrastive-head summary -> {side_path}: {sidecar}")
    return ckpt_path
