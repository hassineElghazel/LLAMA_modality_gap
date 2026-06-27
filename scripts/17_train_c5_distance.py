"""C5: joint autoregressive + distance Stage-2 training (sweep on lambda_d).
Clone of ``scripts/14_train_c4_joint.py``. The ONLY functional change is the
geometric term: C4's angular InfoNCE (orientation) is replaced by a batch-mean
distance term (location) closing the connector output onto the frozen text
centroid mu_y. The AR pipeline, connector init (from C2), LoRA setup and the
image stream are identical to C4, so ``lambda_d=0`` reproduces C3 exactly and
the C4-vs-C5 difference is distance-vs-orientation only.

mu_y and trace_x are FROZEN constants, precomputed once at startup from the
existing C3 diagnostic embeddings (the same ones behind gap_C3_stage2.json):
    mu_y    = mean over rows of projected_C3_stage2_text_pooled.pt
    trace_x = mean_j || x_j - mu_x ||^2 on projected_C3_stage2_image_pooled.pt

Examples
--------
    python scripts/17_train_c5_distance.py --lambda-d 0.5 \\
        --output-name stage2_vlm_C5_lam0p5.pt --max-steps 450
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
from src.training.stage2_distance_sft import train_stage2_distance
from src.utils import notify
from src.utils.io import load_yaml, snapshot_run_metadata
from src.utils.reproducibility import set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_stage2_entry():
    """Import scripts/06_train_stage2.py (numeric filename -> importlib) to reuse
    its AR collate helpers, keeping C5's AR path identical to C1/C3/C4."""
    path = REPO_ROOT / "scripts" / "06_train_stage2.py"
    spec = importlib.util.spec_from_file_location("stage2_entry", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _image_batches(
    dataset: LLaVAInstruct150KDataset,
    captions: CocoTrainCaptions,
    batch_size: int,
) -> Iterator[dict]:
    """Infinite stream of {"images": [PIL], "captions": [str]} of ``batch_size``.

    Identical to C4's ``_contrastive_batches`` (same skip rules, same cycling)
    so the image stream feeding the distance term matches C4 exactly. The
    distance loss only consumes ``images``; ``captions`` is carried solely to
    reproduce C4's per-item filtering and ordering.
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


def _load_mu_y(path: str, device: str) -> torch.Tensor:
    """Frozen global text centroid = mean over rows of the C3 text embeddings."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"mu_y source not found: {path}. Run the C3 extraction "
            f"(scripts/07_extract_projected.py --condition C3_stage2) first."
        )
    t = torch.load(str(p), map_location="cpu")          # (N, 4096) float64
    return t.mean(dim=0).to(device=device, dtype=torch.float32)


def _resolve_trace_x(d_cfg: dict) -> float:
    """Frozen image-cloud trace. Prefer computing from the saved C3 image
    embeddings (self-consistent with the gap report); else use the config scalar."""
    src = d_cfg.get("trace_x_source")
    if src and Path(src).exists():
        X = torch.load(str(src), map_location="cpu").to(torch.float64)  # (N,4096)
        mu = X.mean(dim=0)
        tr = float(((X - mu) ** 2).sum(dim=1).mean())
        print(f"[c5] trace_x computed from {src}: {tr:.2f}")
        return tr
    tr = float(d_cfg.get("trace_x", 4582.0))
    print(f"[c5] trace_x from config scalar: {tr:.2f}")
    return tr


def main():
    stage2_entry = _load_stage2_entry()
    stage2_entry._maybe_apply_liger()

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training_c5.yaml")
    p.add_argument("--projector-config", default="configs/projector.yaml")
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--llm-config", default="configs/llm.yaml")
    p.add_argument("--data-config", default="configs/data.yaml")
    p.add_argument("--init-connector", default=None,
                   help="connector checkpoint to init from (default: C2 from config)")
    p.add_argument("--lambda-d", dest="lambda_d", type=float, default=None,
                   help="distance weight in (1-l)L_AR + l*L_dist (overrides config)")
    p.add_argument("--lambda-s", dest="lambda_s", type=float, default=None,
                   help="C5b scale-pin weight: + l_s*(btrace/btrace0 - 1)^2 holds the "
                        "CLS spread at btrace0 (overrides config). 0 = plain C5")
    p.add_argument("--btrace0", type=float, default=None,
                   help="C5b baseline CLS spread to pin to (default: config/29282)")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--subset-size", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--output-name", default=None,
                   help="checkpoint filename, e.g. stage2_vlm_C5_lam0p5.pt")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    proj_cfg = load_yaml(args.projector_config)
    enc_cfg = load_yaml(args.encoders_config)
    llm_cfg = load_yaml(args.llm_config)
    data_cfg = load_yaml(args.data_config)
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device

    d_cfg = cfg.setdefault("distance", {})
    lambda_d = (
        args.lambda_d if args.lambda_d is not None
        else float(d_cfg.get("lambda_d", 0.5))
    )
    lambda_s = (
        args.lambda_s if args.lambda_s is not None
        else float(d_cfg.get("lambda_s", 0.0))
    )
    btrace0 = (
        args.btrace0 if args.btrace0 is not None
        else d_cfg.get("btrace0")  # may be None when scale pin is off
    )

    if args.output_name:
        out_dir = Path(cfg["output"]["checkpoint_path"]).parent
        cfg["output"]["checkpoint_path"] = str(out_dir / args.output_name)
        print(f"[c5] checkpoint output -> {cfg['output']['checkpoint_path']}")

    # Encoder (frozen CLIP ViT-L/14).
    encoder = build_clip_encoder(enc_cfg).load()

    # Connector: init from C2's Stage-1 connector (or override) — same as C3/C4.
    init = args.init_connector or cfg["init_from"]["connector_checkpoint"]
    if str(init).lower() == "random":
        connector = build_projector(proj_cfg["architecture"])
        print("[c5] connector init: random")
    else:
        connector = load_projector(init)
        print(f"[c5] connector init: {init}")

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
    print(f"[c5] AR schedule: total_steps={cfg['total_steps']:,} "
          f"(items={n_items:,} eff_batch={batch_size * accum} epochs={num_epochs})")
    ar_dataloader = stage2_entry._iter_batches(ar_dataset, tokenizer, image_token_id, batch_size)

    # ----- distance data: same image stream as C4's contrastive iterator -----
    captions = CocoTrainCaptions(d_cfg["caption_annotations"])
    print(f"[c5] COCO train2017 captions loaded: {len(captions):,} images")
    c_batch = int(d_cfg.get("batch_size", 64))
    c_dataset = LLaVAInstruct150KDataset(
        root=llava_cfg["local_path"], image_root=llava_cfg["image_root"],
        limit=None, shuffle=True, seed=cfg["seed"] + 1,
    )
    image_iter = _image_batches(c_dataset, captions, c_batch)

    # ----- frozen geometry targets -----
    mu_y = _load_mu_y(d_cfg["mu_y_source"], device)
    trace_x = _resolve_trace_x(d_cfg)
    print(f"[c5] mu_y loaded: shape={tuple(mu_y.shape)} ||mu_y||={float(mu_y.norm()):.4f}")

    pin = f" + scale-pin lambda_s={lambda_s} btrace0={btrace0}" if lambda_s > 0 else ""
    mode = f"convex lambda_d={lambda_d}{pin}"
    if lambda_s > 0 and btrace0 is None:
        btrace0 = 29282.0  # observed C2-init CLS spread (step-1 btrace in C5 runs)
        print(f"[c5] lambda_s>0 but no btrace0 given; using default {btrace0}")
    device_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    notify.send(
        f"[C5] training started on {device_label}\n"
        f"mode={mode}  d_batch={c_batch}  eff_batch={batch_size * accum}\n"
        f"trace_x={trace_x:.1f}  total_steps={cfg['total_steps']:,}  max_steps={args.max_steps}"
    )
    try:
        ckpt = train_stage2_distance(
            vlm, ar_dataloader, image_iter, cfg,
            lambda_d=lambda_d,
            mu_y=mu_y,
            trace_x=trace_x,
            lambda_s=lambda_s,
            btrace0=float(btrace0) if btrace0 is not None else None,
            max_steps=args.max_steps,
            resume_from=Path(args.resume) if args.resume else None,
        )
    except Exception as exc:
        notify.send(f"[C5] FAILED ({mode}): {exc}")
        raise
    notify.send(f"[C5] training complete ({mode}) — checkpoint {ckpt}")
    snapshot_run_metadata(
        {"c5": cfg, "args": vars(args), "lambda_d": lambda_d,
         "lambda_s": lambda_s, "btrace0": btrace0, "trace_x": trace_x},
        Path(cfg["output"]["log_dir"]),
    )
    print(f"[ok] C5 VLM checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
