"""C6: pure-orientation Stage-2 training (joint AR + InfoNCE, location & scale pinned).

This is C4 (``stage2_joint_sft.py``) with TWO extra constraint terms that hold the
two non-orientation axes of the gap fixed at their C2-init baseline, so that the
InfoNCE term can ONLY move orientation:

    L = (1 - lambda_o) * L_AR
        + lambda_o * L_NCE                                  # orientation (C4's term)
        + lambda_p * || mean_b(z) - mu_x0 ||^2 / trace_x    # location pin  (hold mean)
        + lambda_s * (btrace / btrace0 - 1)^2               # scale pin     (hold spread)

``z = projector(CLS)`` is the SAME CLS->connector path C4's InfoNCE uses; all three
geometric terms are computed off that one forward. ``L_NCE`` runs on L2-normalised
vectors, so it only sees direction (orientation) and is blind to location/scale;
the two pins constrain exactly the axes NCE leaks into through the nonlinear
connector (C4 incidentally closed G_mu 246->198 and compressed the cloud).

Why this experiment: C4's captioning gain is confounded — it moved location,
orientation AND shape at once. Pinning location and scale isolates orientation.
A rotation about the mean preserves both the mean and the trace, so pinning those
two leaves orientation fully free to rotate; the pins only cancel NCE's incidental
translation/compression. If C6 ~ C4 then orientation alone is the lever; if C6
drops toward baseline then C4's win was the joint move.

Pin targets are FROZEN baselines:
- ``mu_x0``   : baseline CLS centroid. If not provided, AUTO-MEASURED at the first
  contrastive step (connector still C2-init -> that IS the baseline location).
- ``btrace0`` : baseline CLS spread. Same auto-measure rule (matches C5b).

At ``lambda_p=0`` and ``lambda_s=0`` this loop is byte-identical to C4 (the pins are
never built). This module does NOT modify ``stage2_sft`` or ``stage2_joint_sft`` —
the C0--C5 pipeline stays frozen for a clean comparison.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import torch
from rich.console import Console

from ..models.vlm import VLM
from ..utils import notify
from .contrastive_loss import LearnableTemperature, symmetric_infonce
from .stage1_pretrain import encode_text_mean_pool
from .stage2_sft import _apply_lora, _save_vlm
from .trainer_utils import build_adamw, cosine_with_warmup, freeze_module

console = Console()


def train_stage2_orientation_pinned(
    vlm: VLM,
    ar_dataloader: Iterable,
    contrastive_iter: Iterator[dict],
    cfg: dict,
    *,
    lambda_contrastive: float,
    trace_x: float,
    lambda_p: float = 0.0,
    lambda_s: float = 0.0,
    pool: str = "cls",
    mu_x0: Optional[torch.Tensor] = None,
    btrace0: Optional[float] = None,
    max_steps: Optional[int] = None,
    progress_cb: Optional[Callable[[int, float, float], None]] = None,
    resume_from: Optional[Path] = None,
) -> Path:
    """Joint AR + InfoNCE Stage-2 loop with location & scale pins (C6).

    ``ar_dataloader`` yields the same batches as Stage 2
    (``{"images", "input_ids", "labels"}``). ``contrastive_iter`` is an *infinite*
    iterator yielding ``{"images": list[PIL], "captions": list[str]}`` of the
    contrastive batch size (the SAME stream C4 uses); one batch is consumed per
    optimizer step. ``trace_x`` is the frozen scalar that normalises the location
    pin (kept on the same scale as C5's ``L_dist``).

    ``lambda_p>0`` adds the location pin to ``mu_x0`` and ``lambda_s>0`` adds the
    scale pin to ``btrace0``; either target, if ``None``, is auto-measured at the
    first contrastive step (connector still C2-init = the baseline). With both
    lambdas 0 this is byte-identical to C4 (convex path).
    """
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype_amp = torch.bfloat16 if cfg["precision"]["amp"] == "bf16" else torch.float32
    vlm.to(device)

    # Frozen pin normaliser + optional explicit baselines.
    trace_x = float(trace_x)
    if trace_x <= 0:
        raise ValueError(f"trace_x must be positive, got {trace_x}")
    # pool = "cls": orientation + pins act on projector(CLS) (original C4/C6 path).
    # pool = "all257": they act on mean_257(projector(vis)) -- the SAME pooled-257
    # vector G_mu / subspace_overlap are measured on and the decoder ingests, so
    # control == measurement (mirrors the C5/C5bp distance trainer).
    if pool not in ("cls", "all257"):
        raise ValueError(f"pool must be 'cls' or 'all257', got {pool!r}")
    use_loc_pin = lambda_p > 0.0
    use_scale_pin = lambda_s > 0.0
    if use_loc_pin and mu_x0 is not None:
        mu_x0 = mu_x0.to(device=device, dtype=torch.float32).detach()
        if mu_x0.ndim != 1:
            raise ValueError(f"mu_x0 must be 1-D (D,), got shape {tuple(mu_x0.shape)}")
    if use_scale_pin and btrace0 is not None:
        btrace0 = float(btrace0)
        if btrace0 <= 0:
            raise ValueError(f"btrace0 must be positive, got {btrace0}")

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
        console.log("[c6] gradient checkpointing enabled (non-reentrant, use_cache=False)")

    lora_cfg = cfg.get("lora") or {}
    if lora_cfg.get("enabled", True):
        vlm = _apply_lora(vlm, lora_cfg)
    if cfg["freeze"].get("llm", True):
        for n, p in vlm._llm.named_parameters():
            if "lora" not in n.lower():
                p.requires_grad = False

    # ----- contrastive head: learnable temperature (same as C4) -----
    c_cfg = cfg.get("contrastive", {})
    temp = LearnableTemperature(
        temperature_init=float(c_cfg.get("temperature_init", 0.07))
    ).to(device)
    temp.train()
    max_cap_len = int(c_cfg.get("max_caption_length", 64))

    trainable = [p for p in vlm.parameters() if p.requires_grad]
    trainable += list(temp.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    pin_msg = ""
    if use_loc_pin:
        m0 = f"||mu_x0||={float(mu_x0.norm()):.1f}" if mu_x0 is not None else "auto@step1"
        pin_msg += f"  loc-pin lambda_p={lambda_p} ({m0})"
    if use_scale_pin:
        b0 = f"{btrace0:.1f}" if btrace0 is not None else "auto@step1"
        pin_msg += f"  scale-pin lambda_s={lambda_s} btrace0={b0}"
    console.log(
        f"[c6] trainable params: {n_trainable / 1e6:.2f}M  "
        f"(convex lambda_o={lambda_contrastive}, pool={pool}, trace_x={trace_x:.1f}){pin_msg}"
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

    # ----- resume (weights + optimizer/scheduler/step + temperature) -----
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
        if "optimizer" in blob and "scheduler" in blob:
            optimizer.load_state_dict(blob["optimizer"])
            scheduler.load_state_dict(blob["scheduler"])
            start_step = int(blob.get("step", 0))
        console.log(f"[c6] resumed from {resume_from} at step={start_step}/{total_steps}")

    # convex weights (matches C4's non-Kendall path).
    w_ar = 1.0 - lambda_contrastive

    diag = {"loc_drift": 0.0, "btrace": 0.0}

    def _contrastive_step():
        """One InfoNCE forward on a fresh contrastive batch (no LLaMA body),
        plus the optional location/scale pins computed off the same z_img.

        Image path (encode -> CLS -> connector) is copied verbatim from C4.
        Returns ``(l_nce, l_loc, l_scale)``; ``l_loc``/``l_scale`` are ``None``
        unless their pin is active.
        """
        nonlocal mu_x0, btrace0
        cbatch = next(contrastive_iter)
        with torch.no_grad():
            vis = vlm.encoder.encode_image_tokens(cbatch["images"])   # (B,257,1024)
        vis = vis.to(device=device, dtype=conn_dtype)
        with torch.amp.autocast("cuda", dtype=dtype_amp, enabled=torch.cuda.is_available()):
            if pool == "all257":
                # control == measurement: pool ALL 257 projected tokens exactly as
                # extract_projected / G_mu do (proj_tokens.mean(dim=1)).
                z_img = vlm.projector(vis).mean(dim=1)                 # (B,4096)
            else:
                z_img = vlm.projector(vis[:, 0, :])                   # CLS token (original)
            with torch.no_grad():
                z_txt = encode_text_mean_pool(
                    cbatch["captions"], tokenizer, embed_layer, device, max_length=max_cap_len
                )
            z_txt = z_txt.to(dtype=z_img.dtype)
            l_nce = symmetric_infonce(z_img, z_txt, temp())

        # Pins computed in fp32 for a well-conditioned geometric term (gradient
        # still flows back through z_img / connector). NCE saw only DIRECTION
        # (it normalises); these pins act on the un-normalised mean / spread.
        zf = z_img.float()
        zbar = zf.mean(dim=0)                                          # (4096,)
        l_loc = l_scale = None
        if use_loc_pin:
            if mu_x0 is None:
                # first contrastive step: connector still C2-init -> baseline
                # location. Freeze it (no hardcoded constant needed).
                mu_x0 = zbar.detach().clone()
                console.log(
                    f"[c6] loc-pin: captured mu_x0 ||mu_x0||={float(mu_x0.norm()):.1f} "
                    f"(C2-init CLS centroid)"
                )
            l_loc = ((zbar - mu_x0) ** 2).sum() / trace_x
            diag["loc_drift"] = float((zbar.detach() - mu_x0).norm())
        else:
            diag["loc_drift"] = 0.0
        if use_scale_pin:
            btrace = ((zf - zbar) ** 2).sum(dim=1).mean()
            if btrace0 is None:
                btrace0 = float(btrace.detach())
                console.log(f"[c6] scale-pin: captured btrace0={btrace0:.1f} (C2-init CLS spread)")
            l_scale = (btrace / btrace0 - 1.0) ** 2
            diag["btrace"] = float(btrace.detach())
        else:
            with torch.no_grad():
                diag["btrace"] = float(((zf - zbar) ** 2).sum(dim=1).mean())
        return l_nce, l_loc, l_scale

    run_contrastive = lambda_contrastive > 0.0 or use_loc_pin or use_scale_pin

    step = start_step
    done = False
    last_nce = last_loc = last_scale = 0.0
    for epoch in range(sched_cfg["num_epochs"]):
        if done:
            break
        optimizer.zero_grad(set_to_none=True)
        for micro_idx, batch in enumerate(ar_dataloader):
            # --- AR micro-batch (scaled by w_ar / accum), identical to C4 ---
            with torch.amp.autocast("cuda", dtype=dtype_amp, enabled=torch.cuda.is_available()):
                out = vlm(
                    batch["images"],
                    batch["input_ids"].to(device),
                    labels=batch["labels"].to(device),
                )
            (w_ar * out.loss / accum).backward()

            if (micro_idx + 1) % accum == 0:
                # --- orientation (+ pins) term: one fwd+bwd per optimizer step ---
                if run_contrastive:
                    l_nce, l_loc, l_scale = _contrastive_step()
                    last_nce = float(l_nce.detach())
                    geo = lambda_contrastive * l_nce
                    if l_loc is not None:
                        last_loc = float(l_loc.detach())
                        geo = geo + lambda_p * l_loc
                    if l_scale is not None:
                        last_scale = float(l_scale.detach())
                        geo = geo + lambda_s * l_scale
                    geo.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % log_every == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    loc_str = f"L_loc={last_loc:.4f} drift={diag['loc_drift']:.2f} " if use_loc_pin else ""
                    scale_str = f"L_scale={last_scale:.4f} " if use_scale_pin else ""
                    console.log(
                        f"[c6] epoch={epoch} step={step}/{total_steps} "
                        f"L_ar={out.loss.item():.4f} L_nce={last_nce:.4f} {loc_str}{scale_str}"
                        f"lambda_o={lambda_contrastive:.3f} btrace={diag['btrace']:.1f} "
                        f"tau={temp.temperature:.4f} lr={lr_now:.2e}"
                    )
                    if notify_every and step % notify_every == 0:
                        pct = 100 * step / total_steps
                        notify.send(
                            f"[C6] step {step}/{total_steps} ({pct:.0f}%)\n"
                            f"L_ar={out.loss.item():.4f} L_nce={last_nce:.4f} "
                            f"drift={diag['loc_drift']:.2f} btrace={diag['btrace']:.1f}"
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

    # Sidecar: record the orientation head + pin state so the run is reportable.
    sidecar = {
        "mode": "orientation_pinned",
        "pool": pool,
        "lambda_o": float(lambda_contrastive),
        "lambda_p": float(lambda_p),
        "lambda_s": float(lambda_s),
        "trace_x": trace_x,
        "mu_x0_norm": float(mu_x0.norm()) if (use_loc_pin and mu_x0 is not None) else None,
        "btrace0": btrace0 if use_scale_pin else None,
        "temperature": float(temp.temperature),
        "final_loc_drift": diag["loc_drift"],
        "final_btrace": diag["btrace"],
        "steps": int(step),
    }
    side_path = Path(str(ckpt_path).replace(".pt", "_orientation_pinned.json"))
    side_path.parent.mkdir(parents=True, exist_ok=True)
    side_path.write_text(json.dumps(sidecar, indent=2))
    console.log(f"[c6] orientation-pinned summary -> {side_path}: {sidecar}")
    return ckpt_path
