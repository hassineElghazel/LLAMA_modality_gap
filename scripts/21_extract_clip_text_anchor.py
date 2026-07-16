"""Build the CLIP-anchor text cloud for the dd256 gap set (Cloc's new mu_y source).

Encodes the 1300 captions of the dd256 gap manifest through the FROZEN CLIP text
tower + frozen semi-orthogonal 768->4096 lift (src/encoders/clip_text_encoder.py),
and saves the (1300, 4096) cloud. Cloc's distance trainer (17_train_c5_distance.py)
reads it via ``_load_mu_y`` -> ``mean(dim=0)`` = the CLIP text centroid mu_y.

Loading the tower here also MATERIALISES the shared lift artifact
(outputs/anchors/clip_lift.pt) the first time; Corient's live InfoNCE positive then
loads that same W, so both models anchor to one identical CLIP geometry.

The scale diagnostics matter: Cloc's loss is ||mean(z_img) - mu_y||^2 / trace_x with
trace_x frozen from the C3 IMAGE cloud. If ||mu_y|| here is wildly off the LLaMA-anchor
scale, lambda_d's effective pull changes — eyeball the printed norms before launching
the 20h retrain.

Example:
    python scripts/21_extract_clip_text_anchor.py \
        --data-config configs/data_1300.yaml \
        --out outputs/embeddings_1300/clipanchor_text_1300.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.coco_val2017_loader import load_diagnostic_manifest
from src.encoders.clip_text_encoder import build_clip_text_tower
from src.utils.io import load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-config", default="configs/data_1300.yaml")
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--out", default="outputs/embeddings_1300/clipanchor_text_1300.pt")
    p.add_argument("--lift-path", default="outputs/anchors/clip_lift.pt",
                   help="shared frozen-lift artifact; keep IDENTICAL across all jobs")
    p.add_argument("--lift-method", default="semi_orthogonal",
                   choices=["semi_orthogonal", "random"])
    p.add_argument("--proj-seed", type=int, default=42)
    p.add_argument("--match-norm", type=float, default=None,
                   help="scale the saved anchor so ||mu_y_clip|| equals this value "
                        "(e.g. the image centroid norm 177.85). CLIP's semantic DIRECTION "
                        "is preserved; only the magnitude changes so the direction actually "
                        "drives Cloc's location move. Cloc-only (Corient's InfoNCE is "
                        "scale-invariant, reads the unscaled lift). None = native CLIP scale.")
    p.add_argument("--device", default=None, help="override encoders.yaml inference.device")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    data_cfg = load_yaml(args.data_config)
    enc_cfg = load_yaml(args.encoders_config)
    if args.device:
        enc_cfg.setdefault("inference", {})["device"] = args.device

    pairs = load_diagnostic_manifest(data_cfg["diagnostic_sample"]["manifest_path"])
    captions = [pr.caption for pr in pairs]
    print(f"[anchor] {len(captions)} captions from {data_cfg['diagnostic_sample']['manifest_path']}")

    tower = build_clip_text_tower(
        enc_cfg, out_dim=4096, proj_seed=args.proj_seed,
        lift_method=args.lift_method, lift_path=args.lift_path,
    ).load()
    print(f"[anchor] lift ready ({args.lift_method}, seed={args.proj_seed}) -> {args.lift_path}  "
          f"W={tuple(tower.W.shape)}")

    cloud = tower.encode(captions, batch_size=args.batch_size).to(torch.float32).cpu()  # (N,4096)

    raw_mu_norm = float(cloud.mean(dim=0).norm())
    if args.match_norm is not None:
        s = float(args.match_norm) / max(raw_mu_norm, 1e-12)
        cloud = cloud * s
        print(f"[anchor] match-norm: scaled lift output by s={s:.4f} so "
              f"||mu_y_clip|| {raw_mu_norm:.4f} -> {args.match_norm:.4f} "
              f"(CLIP direction preserved; magnitude set to image-cloud scale)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cloud, str(out_path))

    mu = cloud.mean(dim=0)
    sample_norm = cloud.norm(dim=1).mean()
    print(f"[anchor] saved {tuple(cloud.shape)} -> {out_path}")
    print(f"[anchor] ||mu_y_clip||={float(mu.norm()):.4f}  mean ||row||={float(sample_norm):.4f}")
    print("[anchor] calibration: Cloc normalises the distance by the frozen image "
          "trace_x (~4253 for C3pinr); keep lambda_d fixed for dosage-comparability. "
          "The scale only affects Cloc's target — Corient reads the unscaled shared lift.")


if __name__ == "__main__":
    main()
