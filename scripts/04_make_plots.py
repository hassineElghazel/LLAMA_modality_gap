"""Generate the four diagnostic figures (A, B, C, D) for a measurement point."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.diagnostics.plots import make_all_figures


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
    p.add_argument("--out-dir", default="outputs/figures")
    args = p.parse_args()

    img_name, txt_name = _FILE_TAGS[args.measurement_point]
    X = torch.load(Path(args.embeddings_dir) / img_name)
    Y = torch.load(Path(args.embeddings_dir) / txt_name)
    paths = make_all_figures(X, Y, Path(args.out_dir) / args.measurement_point)
    for k, (png, pdf) in paths.items():
        print(f"[ok] figure {k}: {png}")


if __name__ == "__main__":
    main()
