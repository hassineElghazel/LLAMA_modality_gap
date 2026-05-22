"""Score captioning predictions against COCO val2017 references.

CIDEr / BLEU-4 / METEOR (+ SPICE if requested). Java required for METEOR/SPICE.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.captioning.evaluation import score_predictions
from src.data.coco_val2017_loader import CocoVal2017Dataset
from src.utils.io import load_yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/captioning.yaml")
    p.add_argument("--condition", required=True,
                   help="condition tag matching the captions_<condition>.json file")
    p.add_argument("--no-spice", action="store_true",
                   help="skip SPICE (faster smoke runs; avoids Java SPICE jar)")
    args = p.parse_args()

    cap_cfg = load_yaml(args.config)
    pred_path = Path(cap_cfg["output"]["predictions_dir"]) / f"captions_{args.condition}.json"

    ds = CocoVal2017Dataset(
        annotations_json=cap_cfg["eval_set"]["annotations_json"],
        image_root=cap_cfg["eval_set"]["image_root"],
    )
    references = ds.references()

    scores_path = Path(cap_cfg["output"]["scores_dir"]) / f"captioning_{args.condition}.json"
    scores = score_predictions(
        pred_path, references, scores_path,
        include_spice=not args.no_spice,
    )
    print(f"[ok] {args.condition} scores: " + " ".join(f"{k}={v:.4f}" for k, v in scores.items()))
    print(f"[ok] wrote {scores_path}")


if __name__ == "__main__":
    main()
