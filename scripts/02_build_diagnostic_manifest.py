"""Build the COCO val2017 diagnostic manifest (5000 pairs, fixed seed).

Reads ``configs/data.yaml`` for paths and sample size, then writes
``outputs/diagnostics_manifest.json`` — used by every condition's
gap-measurement step so the same (image, caption) pairs are scored
across C0/C1/C2/C3.
"""
from __future__ import annotations

import argparse

from src.data.coco_val2017_loader import (
    CocoVal2017Dataset,
    save_diagnostic_manifest,
)
from src.utils.io import load_yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-config", default="configs/data.yaml")
    args = p.parse_args()

    cfg = load_yaml(args.data_config)
    coco_cfg = cfg["coco_val2017"]
    diag_cfg = cfg["diagnostic_sample"]

    ds = CocoVal2017Dataset(
        image_root=coco_cfg["image_root"],
        annotations_json=coco_cfg["annotations_json"],
    )
    pairs = ds.diagnostic_sample(
        n=diag_cfg["num_pairs"],
        caption_per_image=diag_cfg["caption_per_image"],
        seed=diag_cfg["seed"],
    )
    out_path = diag_cfg["manifest_path"]
    save_diagnostic_manifest(pairs, out_path)
    print(f"[manifest] wrote {len(pairs)} pairs -> {out_path}")


if __name__ == "__main__":
    main()
