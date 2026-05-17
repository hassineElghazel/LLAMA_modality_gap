"""Score zero-shot captions against COCO Karpathy test references.

Computes CIDEr (primary), BLEU-4, METEOR, SPICE via pycocoevalcap.
Requires Java 8+ on PATH for METEOR and SPICE.
"""
from __future__ import annotations

import argparse

from src.captioning.evaluation import score_predictions
from src.utils.io import load_yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/captioning.yaml")
    p.add_argument("--data-config", default="configs/data.yaml")
    args = p.parse_args()

    cap_cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)

    scores = score_predictions(
        predictions_path=cap_cfg["output"]["predictions_path"],
        karpathy_json=data_cfg["coco"]["karpathy_split_json"],
        out_path=cap_cfg["output"]["scores_path"],
    )
    for k, v in scores.items():
        print(f"  {k:10s} {v:.4f}")


if __name__ == "__main__":
    main()
