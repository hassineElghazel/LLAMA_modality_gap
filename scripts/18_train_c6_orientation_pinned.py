"""C6: pure-orientation Stage-2 training (joint AR + InfoNCE, location & scale pinned).

Clone of ``scripts/14_train_c4_joint.py`` (the C4 driver: connector init from C2,
LoRA on LLaMA-2-7B, teacher-forced LLaVA collate, the SAME COCO-caption contrastive
stream) with two extra constraint terms bolted on so InfoNCE can only move
orientation:

    L = (1 - lambda_o) L_AR + lambda_o L_NCE
        + lambda_p || mean_b(z) - mu_x0 ||^2 / trace_x   # hold location (baseline CLS centroid)
        + lambda_s (btrace / btrace0 - 1)^2              # hold scale    (baseline CLS spread)

``mu_x0`` and ``btrace0`` are FROZEN baselines auto-measured at the first
contrastive step (connector still C2-init), exactly like C5b's btrace0; pass
``--btrace0`` / config to override. ``trace_x`` is the frozen normaliser, computed
once from the C3 image embeddings (kept on the same scale as C5's distance term).
At ``--lambda-p 0 --lambda-s 0`` this reproduces C4 exactly.

Examples
--------
    python scripts/18_train_c6_orientation_pinned.py --lambda 0.9 \\
        --lambda-p 0.5 --lambda-s 1.0 \\
        --output-name stage2_vlm_C6_lam0p9.pt --max-steps 450
"""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
from typing import Iterator

import torch

from src.data.coco_train_captions import CocoTrainCaptions
from src.data.llava_instruct_loader import LLaVAInstruct150KDataset, load_image
from src.encoders.clip_encoder import build_clip_encoder
from src.models.checkpoint import load_projector
from src.models.projector import build_projector
from src.models.vlm import VLM, VLMConfig
from src.training.stage2_orientation_pinned_sft import train_stage2_orientation_pinned
from src.utils import notify
from src.utils.io import load_yaml, snapshot_run_metadata
from src.utils.reproducibility import set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_stage2_entry():
    """Import scripts/06_train_stage2.py (numeric filename -> importlib) to reuse
    its AR collate helpers, keeping C6's AR path identical to C1/C3/C4/C5."""
    path = REPO_ROOT / "scripts" / "06_train_stage2.py"
    spec = importlib.util.spec_from_file_location("stage2_entry", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _contrastive_batches(
    dataset: LLaVAInstruct150KDataset,
    captions: CocoTrainCaptions,
    batch_size: int,
) -> Iterator[dict]:
    """Infinite stream of {"images": [PIL], "captions": [str]} of ``batch_size``.

    Byte-identical to C4's ``_contrastive_batches`` (same skip rules, same cycling)
    so the InfoNCE / pin stream feeding C6 matches C4 exactly.
    """
    buf_img: list = []
    buf_cap: list[str] = []
    while True:
        for item in dataset:
            cap = captions.caption_for_filename(item.image_path.name)
            if cap is None:
                continue
            try:
                img = load_image(item.image_path)
            except FileNotFoundError:
                continue
            buf_img.append(img)
            buf_cap.append(cap)
            if len(buf_img) == batch_size:
                yield {"images": buf_img, "captions": buf_cap}
                buf_img, buf_cap = [], []


def _resolve_trace_x(p_cfg: dict) -> float:
    """Frozen image-cloud trace normaliser for the location pin. Prefer computing
    from the saved C3 image embeddings (self-consistent with the gap report and
    with C5's L_dist); else use the config scalar."""
    src = p_cfg.get("trace_x_source")
    if src and Path(src).exists():
        X = torch.load(str(src), map_location="cpu").to(torch.float64)  # (N,4096)
        mu = X.mean(dim=0)
        tr = float(((X - mu) ** 2).sum(dim=1).mean())
        print(f"[c6] trace_x computed from {src}: {tr:.2f}")
        return tr
    tr = float(p_cfg.get("trace_x", 4582.0))
    print(f"[c6] trace_x from config scalar: {tr:.2f}")
    return tr


def _load_mu_y(path: str, device: str) -> torch.Tensor:
    """Frozen global text centroid = mean over rows of the C3 text embeddings
    (same source and form as C5/Cloc's distance target)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"mu_y source not found: {path}. Run the C3 extraction "
            f"(scripts/07_extract_projected.py --condition C3_stage2) first."
        )
    t = torch.load(str(p), map_location="cpu")           # (N, 4096)
    return t.mean(dim=0).to(device=device, dtype=torch.float32)


def main():
    stage2_entry = _load_stage2_entry()
    stage2_entry._maybe_apply_liger()

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training_c6.yaml")
    p.add_argument("--projector-config", default="configs/projector.yaml")
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--llm-config", default="configs/llm.yaml")
    p.add_argument("--data-config", default="configs/data.yaml")
    p.add_argument("--init-connector", default=None,
                   help="connector checkpoint to init from (default: C2 from config)")
    p.add_argument("--lambda", dest="lambda_contrastive", type=float, default=None,
                   help="orientation (InfoNCE) weight in (1-l)L_AR + l*L_NCE (overrides config)")
    p.add_argument("--lambda-p", dest="lambda_p", type=float, default=None,
                   help="location-pin weight: + l_p*||mean_b(z)-mu_x0||^2/trace_x holds the "
                        "CLS centroid at baseline (overrides config). 0 = no pin (plain C4)")
    p.add_argument("--lambda-s", dest="lambda_s", type=float, default=None,
                   help="scale-pin weight: + l_s*(btrace/btrace0 - 1)^2 holds the CLS spread "
                        "at btrace0 (overrides config). 0 = no pin")
    p.add_argument("--btrace0", type=float, default=None,
                   help="baseline CLS spread to pin to (default: config; null = auto@step1)")
    p.add_argument("--lambda-rank", dest="lambda_r", type=float, default=None,
                   help="rank-pin weight: + l_r*(eff_rank/effrank0 - 1)^2 holds the "
                        "participation-ratio eff_rank at effrank0 (scale-invariant, "
                        "orthogonal to the scale pin). 0 = no pin")
    p.add_argument("--effrank0", type=float, default=None,
                   help="explicit rank-pin target (participation ratio); null = auto@step1")
    p.add_argument("--pool", choices=("cls", "all257"), default=None,
                   help="geometry object for InfoNCE + both pins: 'cls' (token 0, "
                        "original) or 'all257' (mean of all 257 projected tokens = "
                        "control==measurement). Default: config 'pool' or 'cls'.")
    p.add_argument("--close-location", dest="close_location", action="store_true",
                   help="Clocorient: RETARGET the location leg from the baseline "
                        "centroid mu_x0 (pin) to the frozen text centroid mu_y "
                        "(CLOSE G_mu, = Cloc's distance drive). --lambda-p becomes "
                        "the closure weight lambda_d and joins the convex AR budget "
                        "(w_ar = 1 - lambda_o - lambda_d). Combines with --lambda "
                        "(orientation) + --lambda-s + --lambda-rank.")
    p.add_argument("--mu-y-source", dest="mu_y_source",
                   default="outputs/embeddings/projected_C3_stage2_text_pooled.pt",
                   help="text-centroid source for --close-location (same as Cloc/C5).")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--subset-size", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--output-name", default=None,
                   help="checkpoint filename, e.g. stage2_vlm_C6_lam0p9.pt")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    proj_cfg = load_yaml(args.projector_config)
    enc_cfg = load_yaml(args.encoders_config)
    llm_cfg = load_yaml(args.llm_config)
    data_cfg = load_yaml(args.data_config)
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device

    c_cfg = cfg.setdefault("contrastive", {})
    p_cfg = cfg.setdefault("pins", {})
    lambda_contrastive = (
        args.lambda_contrastive if args.lambda_contrastive is not None
        else float(c_cfg.get("lambda", 0.9))
    )
    lambda_p = (
        args.lambda_p if args.lambda_p is not None
        else float(p_cfg.get("lambda_p", 0.5))
    )
    lambda_s = (
        args.lambda_s if args.lambda_s is not None
        else float(p_cfg.get("lambda_s", 1.0))
    )
    btrace0 = (
        args.btrace0 if args.btrace0 is not None
        else p_cfg.get("btrace0")  # may be None -> auto-measure at step 1
    )
    lambda_r = (
        args.lambda_r if args.lambda_r is not None
        else float(p_cfg.get("lambda_r", 0.0))
    )
    effrank0 = (
        args.effrank0 if args.effrank0 is not None
        else p_cfg.get("effrank0")  # may be None -> auto-measure at step 1
    )
    pool = args.pool if args.pool is not None else str(cfg.get("pool", "cls"))

    # Display tag for logs/notifications. This module is the shared pinned-orientation
    # trainer (C6 = CLS, C4bp = pooled, future pinned variants); derive the label from
    # the checkpoint name so each run self-labels (e.g. C4bp_lam0p1) instead of always
    # printing "[C6]". Falls back to "C6" when no --output-name is given.
    run_tag = (
        Path(args.output_name).stem.replace("stage2_vlm_", "")
        if args.output_name else "C6"
    )
    cfg["run_tag"] = run_tag

    if args.output_name:
        out_dir = Path(cfg["output"]["checkpoint_path"]).parent
        cfg["output"]["checkpoint_path"] = str(out_dir / args.output_name)
        print(f"[{run_tag}] checkpoint output -> {cfg['output']['checkpoint_path']}")

    # Encoder (frozen CLIP ViT-L/14).
    encoder = build_clip_encoder(enc_cfg).load()

    # Connector: init from C2's Stage-1 connector (or override) — same as C3/C4/C5.
    init = args.init_connector or cfg["init_from"]["connector_checkpoint"]
    if str(init).lower() == "random":
        connector = build_projector(proj_cfg["architecture"])
        print("[c6] connector init: random")
    else:
        connector = load_projector(init)
        print(f"[c6] connector init: {init}")

    quant_cfg = llm_cfg.get("quantization", {})
    vlm = VLM(encoder, connector, VLMConfig(
        llm_hf_id=llm_cfg["model"]["hf_id"],
        weights_dtype=llm_cfg["dtype"]["weights"],
        device=device,
        load_in_4bit=bool(quant_cfg.get("load_in_4bit", False)),
    )).load_llm()
    tokenizer = vlm._tokenizer
    image_token_id = vlm._image_token_id

    # ----- AR data (LLaVA-Instruct-150K), identical to Stage 2 / C4 -----
    llava_cfg = data_cfg["llava_instruct_150k"]
    ar_dataset = LLaVAInstruct150KDataset(
        root=llava_cfg["local_path"], image_root=llava_cfg["image_root"],
        limit=args.subset_size, shuffle=True, seed=cfg["seed"],
    )
    batch_size = cfg["batch"]["per_device_batch_size"]
    accum = max(1, int(cfg["batch"].get("gradient_accumulation_steps", 1)))
    num_epochs = cfg["schedule"]["num_epochs"]
    n_items = len(ar_dataset)
    cfg["total_steps"] = math.ceil(n_items / (batch_size * accum)) * num_epochs
    print(f"[c6] AR schedule: total_steps={cfg['total_steps']:,} "
          f"(items={n_items:,} eff_batch={batch_size * accum} epochs={num_epochs})")
    ar_dataloader = stage2_entry._iter_batches(ar_dataset, tokenizer, image_token_id, batch_size)

    # ----- contrastive data: same images, COCO train2017 caption target (as C4) -----
    captions = CocoTrainCaptions(c_cfg["caption_annotations"])
    print(f"[c6] COCO train2017 captions loaded: {len(captions):,} images")
    c_batch = int(c_cfg.get("batch_size", 64))
    c_dataset = LLaVAInstruct150KDataset(
        root=llava_cfg["local_path"], image_root=llava_cfg["image_root"],
        limit=None, shuffle=True, seed=cfg["seed"] + 1,
    )
    contrastive_iter = _contrastive_batches(c_dataset, captions, c_batch)

    # ----- frozen pin normaliser (mu_x0 / btrace0 auto-measured at step 1) -----
    trace_x = _resolve_trace_x(p_cfg)

    # Clocorient: load the frozen text centroid so the location leg CLOSES G_mu
    # (toward mu_y) instead of pinning it to the baseline image centroid.
    mu_close = None
    if args.close_location:
        mu_close = _load_mu_y(args.mu_y_source, device)
        print(f"[c6] close-location: mu_y loaded from {args.mu_y_source} "
              f"||mu_y||={float(mu_close.norm()):.2f} (lambda_d={lambda_p})")

    pins = []
    if args.close_location and lambda_p > 0:
        pins.append(f"loc-CLOSE->mu_y(lambda_d={lambda_p})")
    elif lambda_p > 0:
        pins.append(f"loc(lambda_p={lambda_p})")
    if lambda_s > 0:
        b0 = f"{float(btrace0):.1f}" if btrace0 is not None else "auto@step1"
        pins.append(f"scale(lambda_s={lambda_s},btrace0={b0})")
    if lambda_r > 0:
        r0 = f"{float(effrank0):.2f}" if effrank0 is not None else "auto@step1"
        pins.append(f"rank(lambda_r={lambda_r},effrank0={r0})")
    pin_label = " + ".join(pins) if pins else "none (== C4)"
    mode = f"convex lambda_o={lambda_contrastive}  pool={pool}  pins=[{pin_label}]"
    device_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    notify.send(
        f"[{run_tag}] training started on {device_label}\n"
        f"mode={mode}  c_batch={c_batch}  eff_batch={batch_size * accum}\n"
        f"trace_x={trace_x:.1f}  total_steps={cfg['total_steps']:,}  max_steps={args.max_steps}"
    )
    try:
        ckpt = train_stage2_orientation_pinned(
            vlm, ar_dataloader, contrastive_iter, cfg,
            lambda_contrastive=lambda_contrastive,
            trace_x=trace_x,
            lambda_p=lambda_p,
            lambda_s=lambda_s,
            pool=pool,
            mu_x0=None,  # auto-measure baseline centroid at step 1 (CLS or pooled)
            mu_close=mu_close,  # non-None => location leg CLOSES toward mu_y (Clocorient)
            btrace0=float(btrace0) if btrace0 is not None else None,
            lambda_r=lambda_r,
            effrank0=float(effrank0) if effrank0 is not None else None,
            max_steps=args.max_steps,
            resume_from=Path(args.resume) if args.resume else None,
        )
    except Exception as exc:
        notify.send(f"[{run_tag}] FAILED ({mode}): {exc}")
        raise
    notify.send(f"[{run_tag}] training complete ({mode}) — checkpoint {ckpt}")
    snapshot_run_metadata(
        {"c6": cfg, "args": vars(args), "lambda_o": lambda_contrastive,
         "lambda_p": lambda_p, "lambda_s": lambda_s, "btrace0": btrace0,
         "lambda_r": lambda_r, "effrank0": effrank0, "trace_x": trace_x},
        Path(cfg["output"]["log_dir"]),
    )
    print(f"[ok] C6 VLM checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
