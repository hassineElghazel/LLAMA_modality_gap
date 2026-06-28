"""C4: joint autoregressive + contrastive Stage-2 training (dose-response on lambda).

Reuses the Stage-2 AR pipeline verbatim (connector init from C2, LoRA on
LLaMA-2-7B, teacher-forced LLaVA-Instruct collate) and adds a contrastive term
on a separate batch of (image, COCO-train2017 caption) pairs. At ``--lambda 0``
this is exactly C3; ``--kendall`` learns the balance instead (reference point).

The AR collate helpers are imported from ``scripts/06_train_stage2.py`` so the
AR path is byte-identical to C1/C3 (no second copy that could drift).

Examples
--------
    python scripts/14_train_c4_joint.py --lambda 0.1 \\
        --output-name stage2_vlm_C4_lam0p1.pt --max-steps 450
    python scripts/14_train_c4_joint.py --kendall \\
        --output-name stage2_vlm_C4_kendall.pt --max-steps 450
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
from src.training.stage2_joint_sft import train_stage2_joint
from src.utils import notify
from src.utils.io import load_yaml, snapshot_run_metadata
from src.utils.reproducibility import set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_stage2_entry():
    """Import scripts/06_train_stage2.py (numeric filename -> importlib) to reuse
    its AR collate helpers, keeping C4's AR path identical to C1/C3."""
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

    Skips items whose image_id has no COCO caption or whose image fails to load.
    Cycles the (shuffled) dataset so we never run dry over the sweep.
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


def main():
    stage2_entry = _load_stage2_entry()
    stage2_entry._maybe_apply_liger()

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training_c4.yaml")
    p.add_argument("--projector-config", default="configs/projector.yaml")
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--llm-config", default="configs/llm.yaml")
    p.add_argument("--data-config", default="configs/data.yaml")
    p.add_argument("--init-connector", default=None,
                   help="connector checkpoint to init from (default: C2 from config)")
    p.add_argument("--lambda", dest="lambda_contrastive", type=float, default=None,
                   help="fixed contrastive weight in (1-l)L_AR + l*L_NCE (overrides config)")
    p.add_argument("--kendall", action="store_true",
                   help="learn the balance via uncertainty weighting instead of a fixed lambda")
    p.add_argument("--pool", choices=["cls", "all257"], default=None,
                   help="image-side pooling for the InfoNCE term: 'cls' (token 0, "
                        "original) or 'all257' (mean of all 257 projected tokens = the "
                        "SAME pooled cloud subspace_overlap is measured on). Default: config or 'cls'")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--subset-size", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--output-name", default=None,
                   help="checkpoint filename, e.g. stage2_vlm_C4_lam0p1.pt")
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
    lambda_contrastive = (
        args.lambda_contrastive if args.lambda_contrastive is not None
        else float(c_cfg.get("lambda", 0.1))
    )
    use_kendall = bool(args.kendall or c_cfg.get("use_kendall", False))
    pool = args.pool if args.pool is not None else str(c_cfg.get("pool", "cls"))

    if args.output_name:
        out_dir = Path(cfg["output"]["checkpoint_path"]).parent
        cfg["output"]["checkpoint_path"] = str(out_dir / args.output_name)
        print(f"[c4] checkpoint output -> {cfg['output']['checkpoint_path']}")

    # Encoder (frozen CLIP ViT-L/14).
    encoder = build_clip_encoder(enc_cfg).load()

    # Connector: init from C2's Stage-1 connector (or override).
    init = args.init_connector or cfg["init_from"]["connector_checkpoint"]
    if str(init).lower() == "random":
        connector = build_projector(proj_cfg["architecture"])
        print("[c4] connector init: random")
    else:
        connector = load_projector(init)
        print(f"[c4] connector init: {init}")

    quant_cfg = llm_cfg.get("quantization", {})
    vlm = VLM(encoder, connector, VLMConfig(
        llm_hf_id=llm_cfg["model"]["hf_id"],
        weights_dtype=llm_cfg["dtype"]["weights"],
        device=device,
        load_in_4bit=bool(quant_cfg.get("load_in_4bit", False)),
    )).load_llm()
    tokenizer = vlm._tokenizer
    image_token_id = vlm._image_token_id

    # ----- AR data (LLaVA-Instruct-150K), identical to Stage 2 -----
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
    print(f"[c4] AR schedule: total_steps={cfg['total_steps']:,} "
          f"(items={n_items:,} eff_batch={batch_size * accum} epochs={num_epochs})")
    ar_dataloader = stage2_entry._iter_batches(ar_dataset, tokenizer, image_token_id, batch_size)

    # ----- contrastive data: same images, COCO train2017 caption target -----
    captions = CocoTrainCaptions(c_cfg["caption_annotations"])
    print(f"[c4] COCO train2017 captions loaded: {len(captions):,} images")
    c_batch = int(c_cfg.get("batch_size", 64))
    # Independent shuffle (seed+1) so the contrastive batch is not aligned with
    # the AR micro-batches — InfoNCE needs distinct in-batch negatives.
    c_dataset = LLaVAInstruct150KDataset(
        root=llava_cfg["local_path"], image_root=llava_cfg["image_root"],
        limit=None, shuffle=True, seed=cfg["seed"] + 1,
    )
    contrastive_iter = _contrastive_batches(c_dataset, captions, c_batch)

    mode = ("kendall" if use_kendall else f"convex lambda={lambda_contrastive}") + f" pool={pool}"
    device_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    notify.send(
        f"[C4] training started on {device_label}\n"
        f"mode={mode}  c_batch={c_batch}  eff_batch={batch_size * accum}\n"
        f"total_steps={cfg['total_steps']:,}  max_steps={args.max_steps}"
    )
    try:
        ckpt = train_stage2_joint(
            vlm, ar_dataloader, contrastive_iter, cfg,
            lambda_contrastive=lambda_contrastive,
            use_kendall=use_kendall,
            pool=pool,
            max_steps=args.max_steps,
            resume_from=Path(args.resume) if args.resume else None,
        )
    except Exception as exc:
        notify.send(f"[C4] FAILED ({mode}): {exc}")
        raise
    notify.send(f"[C4] training complete ({mode}) — checkpoint {ckpt}")
    snapshot_run_metadata(
        {"c4": cfg, "args": vars(args), "lambda": lambda_contrastive, "kendall": use_kendall, "pool": pool},
        Path(cfg["output"]["log_dir"]),
    )
    print(f"[ok] C4 VLM checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
