"""Compute the spec-defined gap metrics for one of the 5 measurement points.

Per Overleaf Table 3 the conditions are: C0_random, C1_stage2, C2_stage1,
C3_stage1, C3_stage2. The script loads the saved 4096-d image/text embeddings
for the chosen condition and runs ``compute_all_metrics``.

Outputs: ``outputs/metrics/gap_<condition>.json``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.diagnostics.metrics import compute_all_metrics
from src.utils.io import save_json


CONDITIONS = ("C0_random", "C1_stage2", "C2_stage1", "C3_stage1", "C3_stage2",
              "C4_lam0p1", "C4_lam0p5", "C4_kendall")


def _embed_paths(condition: str, embeddings_dir: Path) -> tuple[Path, Path]:
    img = embeddings_dir / f"projected_{condition}_image_pooled.pt"
    txt = embeddings_dir / f"projected_{condition}_text_pooled.pt"
    return img, txt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=CONDITIONS)
    p.add_argument("--embeddings-dir", default="outputs/embeddings")
    p.add_argument("--out-dir", default="outputs/metrics")
    args = p.parse_args()

    img_path, txt_path = _embed_paths(args.condition, Path(args.embeddings_dir))
    X = torch.load(img_path)
    Y = torch.load(txt_path)
    metrics = compute_all_metrics(X, Y)

    out_path = Path(args.out_dir) / f"gap_{args.condition}.json"
    save_json(metrics.to_dict(), out_path)
    spec = metrics.to_dict()["spec_metrics"]
    print(
        f"[ok] {args.condition}: "
        f"G_mu={spec['G_mu']:.4f}  alpha_img={spec['alpha_image']:.2f}  "
        f"JS={spec['js_divergence_angular']:.4f}  "
        f"knn={spec['knn_mixing_rate_k20']:.4f}"
    )
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
