"""Generate diagnostic figures for one of the 5 measurement points,
plus the cross-condition trajectory plot when --condition=all is passed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.diagnostics.plots import make_all_figures, plot_gap_decomposition


CONDITIONS = ("C0_random", "C1_stage2", "C2_stage1", "C3_stage1", "C3_stage2")


def _per_condition_figures(condition: str, embeddings_dir: Path, out_dir: Path):
    img = embeddings_dir / f"projected_{condition}_image_pooled.pt"
    txt = embeddings_dir / f"projected_{condition}_text_pooled.pt"
    X = torch.load(img)
    Y = torch.load(txt)
    return make_all_figures(X, Y, out_dir / condition)


def _trajectory_figures(metrics_dir: Path, out_dir: Path) -> Path:
    metrics_by_condition: dict[str, dict] = {}
    for cond in CONDITIONS:
        p = metrics_dir / f"gap_{cond}.json"
        if not p.exists():
            continue
        with p.open() as f:
            metrics_by_condition[cond] = json.load(f)["spec_metrics"]
    if not metrics_by_condition:
        raise FileNotFoundError(f"no gap_*.json metrics under {metrics_dir}")
    return plot_gap_decomposition(metrics_by_condition, out_dir / "trajectory")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", default="all",
                   choices=("all", *CONDITIONS))
    p.add_argument("--embeddings-dir", default="outputs/embeddings")
    p.add_argument("--metrics-dir", default="outputs/metrics")
    p.add_argument("--out-dir", default="outputs/figures")
    args = p.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    out_dir = Path(args.out_dir)

    if args.condition == "all":
        for cond in CONDITIONS:
            img = embeddings_dir / f"projected_{cond}_image_pooled.pt"
            if not img.exists():
                print(f"[skip] {cond}: no embeddings at {img}")
                continue
            paths = _per_condition_figures(cond, embeddings_dir, out_dir)
            for k, (png, _pdf) in paths.items():
                print(f"[ok] {cond} {k}: {png}")
        # Trajectory plot across whichever conditions have metrics.
        traj = _trajectory_figures(Path(args.metrics_dir), out_dir)
        print(f"[ok] trajectory: {traj}")
    else:
        paths = _per_condition_figures(args.condition, embeddings_dir, out_dir)
        for k, (png, _pdf) in paths.items():
            print(f"[ok] {args.condition} {k}: {png}")


if __name__ == "__main__":
    main()
