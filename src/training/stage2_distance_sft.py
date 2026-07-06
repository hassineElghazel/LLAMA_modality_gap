"""C5: joint autoregressive + distance Stage-2 training.

Identical to C4 (``stage2_joint_sft.py``) in every respect EXCEPT the geometric
term: C4's angular InfoNCE (orientation) is replaced by a batch-mean *distance*
term (location). The AR path, LoRA setup, optimizer/schedule, gradient
checkpointing, logging, save and resume are byte-for-byte the C4/Stage-2 paths.

Loss (per optimizer step), convex form (so ``lambda_d`` is comparable to C4's
lambda; ``lambda_d=0`` reduces to C3 exactly):

    L = (1 - lambda_d) * L_AR  +  lambda_d * L_dist
    L_dist = || mean_b(z_img) - mu_y ||^2 / trace_x

with ``z_img = projector(CLS)`` (the SAME CLS->connector path C4 uses), ``mu_y``
the frozen global text centroid and ``trace_x`` the frozen C3 image-cloud trace.

Why the *batch mean* (not per-instance): its per-point gradient is the identical
translation ``(2 / (B * trace_x)) (mean_b(z) - mu_y)`` for every point, so the
covariance is invariant — it moves the centroid (location) without shrinking the
spread. A per-instance ``mean_i ||z_i - mu_y||^2`` is globally minimised at
``z_i = mu_y`` for all i (collapse). See ``c5_experiment_plan``.

Why divide by ``trace_x``: it makes the term dimensionless ((gap)^2/(spread)^2),
so lambda_d is O(1); ``trace_x`` is a FROZEN constant (a live denominator could be
gamed by inflating the spread instead of closing the gap).

This module does NOT modify ``stage2_sft`` or ``stage2_joint_sft`` — the C0--C4
pipeline stays frozen for a clean comparison.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import torch
from rich.console import Console

from ..models.vlm import VLM
from ..utils import notify
from .stage2_sft import _apply_lora, _save_vlm
from .trainer_utils import build_adamw, cosine_with_warmup, freeze_module

console = Console()


def train_stage2_distance(
    vlm: VLM,
    ar_dataloader: Iterable,
    image_iter: Iterator[dict],
    cfg: dict,
    *,
    lambda_d: float,
    mu_y: torch.Tensor,
    trace_x: float,
    lambda_s: float = 0.0,
    btrace0: Optional[float] = None,
    lambda_r: float = 0.0,
    effrank0: Optional[float] = None,
    pool: str = "cls",
    max_steps: Optional[int] = None,
    progress_cb: Optional[Callable[[int, float, float], None]] = None,
    resume_from: Optional[Path] = None,
) -> Path:
    """Joint AR + distance Stage-2 loop (C5 / C5b).

    ``ar_dataloader`` yields the same batches as Stage 2
    (``{"images", "input_ids", "labels"}``). ``image_iter`` is an *infinite*
    iterator yielding ``{"images": list[PIL], ...}`` of the distance batch size
    (the SAME stream C4's contrastive iterator produces; any caption field is
    ignored). ``mu_y`` is a frozen (4096,) tensor; ``trace_x`` a frozen scalar.

    ``lambda_s`` (default 0) adds the C5b scale-pin term
    ``L_scale = (btrace / btrace0 - 1)^2`` that holds the CLS-cloud spread at
    its baseline ``btrace0`` (the C2-init value), forcing the distance term to
    close location by TRANSLATION rather than by shrinking magnitude. At
    ``lambda_s=0`` this is byte-identical to C5 (the scale term is never built).
    """
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype_amp = torch.bfloat16 if cfg["precision"]["amp"] == "bf16" else torch.float32
    vlm.to(device)

    # Frozen geometry targets (never updated, no gradient).
    mu_y = mu_y.to(device=device, dtype=torch.float32).detach()
    trace_x = float(trace_x)
    if trace_x <= 0:
        raise ValueError(f"trace_x must be positive, got {trace_x}")
    # pool = "cls": close mean_b(projector(CLS)) (original). pool = "all257": close
    # mean_b(mean_257(projector(vis))) -- the SAME pooled-257 vector that G_mu is
    # measured on and that the decoder ingests, so control == measurement. At the
    # pooled level the batch-mean gradient is a uniform translation, so location
    # closes WITHOUT compressing trace_image by construction.
    if pool not in ("cls", "all257"):
        raise ValueError(f"pool must be 'cls' or 'all257', got {pool!r}")

    # C5b scale pin: hold the CLS-cloud spread (btrace) at its baseline btrace0.
    # btrace0 is FROZEN once set. If not provided, it is AUTO-MEASURED from the
    # first distance step (where the connector is still C2-init), so it is always
    # correct for whatever the init/data is — no hardcoded constant.
    use_scale_pin = lambda_s > 0.0
    if use_scale_pin and btrace0 is not None:
        btrace0 = float(btrace0)
        if btrace0 <= 0:
            raise ValueError(f"btrace0 must be positive, got {btrace0}")

    # Rank pin (participation ratio of the centered batch cloud). Holds
    # eff_rank at its baseline effrank0 so the distance term closes location
    # WITHOUT the anisotropic squeeze that co-moves rank down in plain C5bp.
    use_rank_pin = lambda_r > 0.0
    if use_rank_pin and effrank0 is not None:
        effrank0 = float(effrank0)
        if effrank0 <= 0:
            raise ValueError(f"effrank0 must be positive, got {effrank0}")

    # ----- freeze / trainable setup (identical to Stage 2 / C4) -----
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
        console.log("[c5] gradient checkpointing enabled (non-reentrant, use_cache=False)")

    lora_cfg = cfg.get("lora") or {}
    if lora_cfg.get("enabled", True):
        vlm = _apply_lora(vlm, lora_cfg)
    if cfg["freeze"].get("llm", True):
        for n, p in vlm._llm.named_parameters():
            if "lora" not in n.lower():
                p.requires_grad = False

    # No contrastive head: the distance loss carries no learnable parameters
    # (mu_y / trace_x are frozen constants). Trainable set = connector + LoRA.
    trainable = [p for p in vlm.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    if use_scale_pin:
        b0_str = f"{btrace0:.1f}" if btrace0 is not None else "auto@step1"
        pin_msg = f"  scale-pin lambda_s={lambda_s} btrace0={b0_str}"
    else:
        pin_msg = ""
    if use_rank_pin:
        r0_str = f"{effrank0:.2f}" if effrank0 is not None else "auto@step1"
        pin_msg += f"  rank-pin lambda_r={lambda_r} effrank0={r0_str}"
    console.log(
        f"[c5] trainable params: {n_trainable / 1e6:.2f}M  "
        f"(convex lambda_d={lambda_d}, pool={pool}, trace_x={trace_x:.1f}){pin_msg}"
    )

    vlm.train()
    if cfg["freeze"].get("vit", True):
        vlm.encoder._vision.eval()

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

    # ----- resume (weights + optimizer/scheduler/step) -----
    start_step = 0
    if resume_from is not None and Path(resume_from).exists():
        blob = torch.load(str(resume_from), map_location="cpu")
        vlm.projector.load_state_dict(blob["connector"])
        own = dict(vlm._llm.named_parameters())
        with torch.no_grad():
            for n, tensor in blob.get("llm_trainable", {}).items():
                if n in own:
                    own[n].data.copy_(tensor.to(own[n].device, dtype=own[n].dtype))
        if "optimizer" in blob and "scheduler" in blob:
            optimizer.load_state_dict(blob["optimizer"])
            scheduler.load_state_dict(blob["scheduler"])
            start_step = int(blob.get("step", 0))
        console.log(f"[c5] resumed from {resume_from} at step={start_step}/{total_steps}")

    # convex weights (matches C4's non-Kendall path).
    w_ar = 1.0 - lambda_d

    diag = {"gap": 0.0, "btrace": 0.0, "effrank": 0.0}

    def _distance_step():
        """One distance forward on a fresh image batch (no LLaMA body).

        Image path (encode -> CLS -> connector) is copied verbatim from C4's
        ``_contrastive_step``. Returns ``(l_dist, l_scale)``; ``l_scale`` is
        ``None`` unless the C5b scale pin is active (``lambda_s>0``).
        """
        ibatch = next(image_iter)
        with torch.no_grad():
            vis = vlm.encoder.encode_image_tokens(ibatch["images"])  # (B,257,1024)
        vis = vis.to(device=device, dtype=conn_dtype)
        with torch.amp.autocast("cuda", dtype=dtype_amp, enabled=torch.cuda.is_available()):
            if pool == "all257":
                # control == measurement: pool ALL 257 projected tokens exactly as
                # extract_projected / G_mu do (proj_tokens.mean(dim=1)).
                z_img = vlm.projector(vis).mean(dim=1)                # (B,4096)
            else:
                z_img = vlm.projector(vis[:, 0, :])                   # CLS token (original)
        # Centroid + distance computed in fp32 for an accurate, well-conditioned
        # geometric term (gradient still flows back through z_img / connector).
        nonlocal btrace0, effrank0
        zf = z_img.float()
        zbar = zf.mean(dim=0)                                         # (4096,)
        l_dist = ((zbar - mu_y) ** 2).sum() / trace_x
        if use_scale_pin:
            # grad-enabled batch spread (CLS-cloud trace); pin it to btrace0 so
            # the optimizer must TRANSLATE the cloud, not shrink it.
            btrace = ((zf - zbar) ** 2).sum(dim=1).mean()
            if btrace0 is None:
                # first distance step: connector is still C2-init -> this IS the
                # baseline spread. Freeze it (no hardcoded constant needed).
                btrace0 = float(btrace.detach())
                console.log(f"[c5] scale-pin: captured btrace0={btrace0:.1f} (C2-init spread)")
            l_scale = (btrace / btrace0 - 1.0) ** 2
            diag["btrace"] = float(btrace.detach())
        else:
            l_scale = None
            with torch.no_grad():
                diag["btrace"] = float(((zf - zbar) ** 2).sum(dim=1).mean())
        if use_rank_pin:
            zc = zf - zbar                                            # (B,4096)
            G = zc @ zc.t()                                           # (B,B) Gram
            pr = torch.diagonal(G).sum() ** 2 / (G * G).sum().clamp_min(1e-12)
            if effrank0 is None:
                effrank0 = float(pr.detach())
                console.log(f"[c5] rank-pin: captured effrank0={effrank0:.2f} (C2-init participation ratio)")
            l_rank = (pr / effrank0 - 1.0) ** 2
            diag["effrank"] = float(pr.detach())
        else:
            l_rank = None
            with torch.no_grad():
                zc = zf - zbar
                G = zc @ zc.t()
                diag["effrank"] = float(torch.diagonal(G).sum() ** 2 / (G * G).sum().clamp_min(1e-12))
        with torch.no_grad():
            diag["gap"] = float((zbar - mu_y).norm())
        return l_dist, l_scale, l_rank

    run_distance = lambda_d > 0.0 or use_scale_pin or use_rank_pin

    step = start_step
    done = False
    last_dist = 0.0
    last_scale = 0.0
    last_rank = 0.0
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
            (w_ar * out.loss / accum).backward()

            if (micro_idx + 1) % accum == 0:
                # --- distance (+ optional scale-pin) term: one fwd+bwd / step ---
                if run_distance:
                    l_dist, l_scale, l_rank = _distance_step()
                    last_dist = float(l_dist.detach())
                    geo = lambda_d * l_dist
                    if l_scale is not None:
                        last_scale = float(l_scale.detach())
                        geo = geo + lambda_s * l_scale
                    if l_rank is not None:
                        last_rank = float(l_rank.detach())
                        geo = geo + lambda_r * l_rank
                    geo.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % log_every == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    scale_str = f"L_scale={last_scale:.4f} " if use_scale_pin else ""
                    rank_str = f"L_rank={last_rank:.4f} effrank={diag['effrank']:.1f} " if use_rank_pin else ""
                    console.log(
                        f"[c5] epoch={epoch} step={step}/{total_steps} "
                        f"L_ar={out.loss.item():.4f} L_dist={last_dist:.4f} {scale_str}{rank_str}"
                        f"lambda_d={lambda_d:.3f} gap={diag['gap']:.4f} "
                        f"btrace={diag['btrace']:.1f} lr={lr_now:.2e}"
                    )
                    if notify_every and step % notify_every == 0:
                        pct = 100 * step / total_steps
                        notify.send(
                            f"[C5] step {step}/{total_steps} ({pct:.0f}%)\n"
                            f"L_ar={out.loss.item():.4f} L_dist={last_dist:.4f} "
                            f"gap={diag['gap']:.4f} btrace={diag['btrace']:.1f}"
                        )
                if progress_cb is not None:
                    progress_cb(step, float(out.loss.item()), last_dist)
                if save_every and step % save_every == 0:
                    _save_vlm(vlm, ckpt_path, optimizer=optimizer,
                              scheduler=scheduler, step=step, epoch=epoch)
                if max_steps is not None and step >= max_steps:
                    done = True
                    break

    _save_vlm(vlm, ckpt_path, optimizer=optimizer, scheduler=scheduler,
              step=step, epoch=epoch)

    # Sidecar: record the distance-head state so the run is reportable.
    sidecar = {
        "mode": "distance_pinned" if use_scale_pin else "distance",
        "pool": pool,
        "lambda_d": float(lambda_d),
        "lambda_s": float(lambda_s),
        "btrace0": btrace0 if use_scale_pin else None,
        "trace_x": trace_x,
        "final_gap": diag["gap"],
        "final_btrace": diag["btrace"],
        "steps": int(step),
    }
    side_path = Path(str(ckpt_path).replace(".pt", "_distance.json"))
    side_path.parent.mkdir(parents=True, exist_ok=True)
    side_path.write_text(json.dumps(sidecar, indent=2))
    console.log(f"[c5] distance-head summary -> {side_path}: {sidecar}")
    return ckpt_path
