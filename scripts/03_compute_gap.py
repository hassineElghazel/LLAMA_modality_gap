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
              "C4_lam0p1", "C4_lam0p3", "C4_lam0p5", "C4_lam0p7", "C4_lam0p9",
              "C4_kendall",
              "C5_lam0p1", "C5_lam0p3", "C5_lam0p5", "C5_lam0p7", "C5_lam0p9",
              "C5b_lam0p5",
              "C6_lam0p9",
              # pooled-257 control==measurement variants (p = pooled):
              "C5p_lam0p1", "C4p_lam0p9",
              # C5bp = pooled distance + scale-pin (isolate location from compression):
              "C5bp_lam0p1",
              # C4bp = pooled InfoNCE + location-pin + scale-pin (isolate orientation):
              "C4bp_lam0p1", "C4bp_lam0p9",
              # C3pin = pins-only control (lambda_o=0, location+scale pinned): location-177 anchor:
              "C3pin",
              # C5bp location dose-response sweep (vary lambda_d, hold scale-pin lambda_s=1.0):
              "C5bp_lam0p9", "C5bp_lam0p7", "C5bp_lam0p5", "C5bp_lam0p3",
              # trace-held (low-lambda_d) points -- the clean location curve:
              "C5bp_lam0p05", "C5bp_lam0p02", "C5bp_lam0p01",
              # single-axis SCALE dose-response: hold loc@177 + rank + no InfoNCE, drive trace to targets:
              "Cscale1500", "Cscale2500", "Cscale3500",
              # C3pinr = C3pin + rank-pin: rank-matched anchor (trace held at baseline, rank ~39):
              "C3pinr",
              # Crank15 = single-axis RANK test: match C3pin loc+trace+orient, drive eff_rank down:
              "Crank15",
              # Cloc = clean-location: distance drive + scale-pin + rank-pin (isolate LOCATION,
              # hold scale AND rank) -> assumption-free location causality.
              "Cloc",
              # Corient = clean-orientation: InfoNCE drive (lambda_o=0.9) + location-pin +
              # scale-pin + rank-pin (isolate ORIENTATION, hold the other 3 axes).
              "Corient",
              # CLIP-text-anchor arm: Cloc/Corient retrained toward the frozen CLIP text tower.
              "Cloc_clip", "Corient_clip",
              # Cloc_clip_native: location drive toward the CLIP centroid at natural scale (~7.9).
              "Cloc_clip_native",
              # Clocorient = combined: Cloc + Corient dosages in one model -- InfoNCE
              # orientation (lambda_o=0.5) + location CLOSURE to mu_y (lambda_d=0.1) +
              # scale-pin + rank-pin. Moves LOCATION + ORIENTATION jointly, scale/rank held.
              "Clocorient")


def _embed_paths(condition: str, embeddings_dir: Path) -> tuple[Path, Path]:
    img = embeddings_dir / f"projected_{condition}_image_pooled.pt"
    txt = embeddings_dir / f"projected_{condition}_text_pooled.pt"
    return img, txt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=CONDITIONS)
    p.add_argument("--embeddings-dir", default="outputs/embeddings")
    p.add_argument("--out-dir", default="outputs/metrics")
    p.add_argument("--text-embeddings", default=None,
                   help="override the text cloud the gap is measured AGAINST (e.g. the CLIP "
                        "anchor outputs/embeddings_1300/clipanchor_text_1300.pt). G_mu and the "
                        "cross-cloud metrics are then computed vs THIS cloud -- the 'real' "
                        "location gap for a CLIP-anchor-trained model (the flag it walked to). "
                        "Default: the condition's own LLaMA text_pooled.")
    p.add_argument("--out-suffix", default="",
                   help="append to the output filename (e.g. _vsCLIP) so a vs-CLIP gap does "
                        "not overwrite the default gap_<cond>.json.")
    args = p.parse_args()

    img_path, txt_path = _embed_paths(args.condition, Path(args.embeddings_dir))
    X = torch.load(img_path)
    Y = torch.load(args.text_embeddings) if args.text_embeddings else torch.load(txt_path)
    metrics = compute_all_metrics(X, Y)

    out_path = Path(args.out_dir) / f"gap_{args.condition}{args.out_suffix}.json"
    save_json(metrics.to_dict(), out_path)
    spec = metrics.to_dict()["spec_metrics"]
    print(
        f"[ok] {args.condition}: "
        f"G_mu={spec['G_mu']:.4f}  alpha_img={spec['alpha_image']:.2f}  "
        f"JS={spec['js_divergence_angular']:.4f}  "
        f"knn={spec['knn_mixing_rate_k20']:.4f}"
    )
    if args.text_embeddings:
        print(f"[ok] gap measured vs text cloud: {args.text_embeddings}")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
