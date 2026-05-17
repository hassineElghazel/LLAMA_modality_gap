"""Compute modality-gap diagnostic metrics from saved embeddings.

Works for any measurement point (encoder / projected_untrained /
projected_stage1 / projected_stage2). The --measurement-point flag picks
which embedding files to load.

Outputs:
  outputs/metrics/gap_<point>.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.diagnostics.metrics import compute_all_metrics
from src.utils.io import save_json


_FILE_TAGS = {
    "encoder": ("encoder_image_embeds.pt", "encoder_text_embeds.pt"),
    "projected_untrained": ("projected_untrained_image_pooled.pt", "projected_untrained_text_pooled.pt"),
    "projected_stage1": ("projected_stage1_image_pooled.pt", "projected_stage1_text_pooled.pt"),
    "projected_stage2": ("projected_stage2_image_pooled.pt", "projected_stage2_text_pooled.pt"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--measurement-point", required=True, choices=sorted(_FILE_TAGS))
    p.add_argument("--embeddings-dir", default="outputs/embeddings")
    p.add_argument("--out-dir", default="outputs/metrics")
    args = p.parse_args()

    img_name, txt_name = _FILE_TAGS[args.measurement_point]
    edir = Path(args.embeddings_dir)
    X = torch.load(edir / img_name)
    Y = torch.load(edir / txt_name)
    metrics = compute_all_metrics(X, Y)

    out_path = Path(args.out_dir) / f"gap_{args.measurement_point}.json"
    save_json(metrics.to_dict(), out_path)
    print(f"[ok] {args.measurement_point}: G_mu={metrics.G_mu:.4f}  A_r={metrics.A_r:.2f}  "
          f"d_eff/d={metrics.d_eff_over_d:.3f}  knn_mix={metrics.knn_mixing_rate_k20:.4f}")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
