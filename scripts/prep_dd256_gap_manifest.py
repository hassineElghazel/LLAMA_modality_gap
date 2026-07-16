"""Build the gap-measurement manifest on the EXACT dd256 grounding set (n=1300).

The dd256 grounding harness (``08_run_captioning.py``) selects its images as
``list(CocoVal2017Dataset.items())[:num_images]`` — i.e. the FIRST ``num_images``
val2017 images by sorted ``image_id`` (deterministic head, NOT a random sample).
To measure the modality gap on the SAME images the grounding scores, this manifest
must reproduce that selection exactly; a ``diagnostic_sample(seed=...)`` random draw
would land on a different subset and silently de-align gap vs grounding.

For each of those images we attach ONE caption (the first by caption_id, matching
the deterministic "first caption per image" convention used elsewhere) so the text
cloud (LLaMA anchor) and Cloc's CLIP ``mu_y`` are both built over the same captions.

Reads the set spec (num_images / annotations / image_root) from
``configs/description_eval.yaml`` so it stays in lock-step with the grounding harness.

Output: a ``load_diagnostic_manifest``-compatible JSON of CocoPair rows.

Example:
    python scripts/prep_dd256_gap_manifest.py \
        --desc-config configs/description_eval.yaml \
        --out data/dd256/gap_manifest_1300.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.coco_val2017_loader import CocoPair, CocoVal2017Dataset, save_diagnostic_manifest
from src.utils.io import load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--desc-config", default="configs/description_eval.yaml",
                   help="grounding-eval config; its eval_set defines the 1300 images")
    p.add_argument("--out", default="data/dd256/gap_manifest_1300.json")
    args = p.parse_args()

    cfg = load_yaml(args.desc_config)
    es = cfg["eval_set"]
    n = int(es["num_images"])

    ds = CocoVal2017Dataset(
        annotations_json=es["annotations_json"],
        image_root=es["image_root"],
    )
    # EXACT dd256 selection: first n images by sorted image_id (ds.items() is sorted).
    items = list(ds.items())[:n]

    pairs: list[CocoPair] = []
    skipped = 0
    for it in items:
        if not it.captions:                     # no caption -> cannot anchor text; skip
            skipped += 1
            continue
        pairs.append(CocoPair(
            image_id=it.image_id,
            image_path=it.image_path,
            caption_id=it.caption_ids[0],       # first caption (deterministic)
            caption=it.captions[0],
        ))

    save_diagnostic_manifest(pairs, args.out)
    print(f"[dd256-manifest] {len(pairs)} pairs (skipped {skipped} caption-less) -> {args.out}")
    print(f"[dd256-manifest] image_id range: {pairs[0].image_id} .. {pairs[-1].image_id}")


if __name__ == "__main__":
    main()
