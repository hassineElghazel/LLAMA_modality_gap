"""Materialize DOCCI (google/docci) train/test splits for the LLaVA 3-arm study.

DOCCI standard partitions (verified against the HF dataset card):
    train 9,647  |  test 5,000  |  qual_dev 100  |  qual_test 100

We use the DEFAULT config (NOT ``DOCCI-AAR``, whose train is 4,932 -- a different,
augmented partition). Each example carries an embedded PIL ``image``, an
``example_id`` string, and a long human ``description`` (~136 tokens) = the gold
detail caption.

For each split we:
  * save every image as zero-padded ``{idx:012d}.jpg`` under
    ``<out_root>/images_<split>/`` so the COCO-style scorers (15_clipscore,
    15b_siglip) resolve them via ``image_root / f"{id:012d}.jpg"`` unchanged;
  * write ``<out_root>/<split>_manifest.json`` = list of
    ``{"image_id", "image_path", "example_id", "caption"}`` with
    ``image_id`` = the split-local enumeration index (0..N-1);
  * write ``<out_root>/<split>_references.json`` = ``{image_id: [caption]}``
    (single gold ref) for the reference-based scorer (28_score_docci_refs).

Contamination / partition guards (hard asserts, --strict on by default):
  * per-split row counts match the standard partition;
  * every ``example_id`` in a split starts with that split's name (rules out a
    silently-swapped config or a shuffled DatasetDict).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.io import save_json

# Standard DOCCI partition sizes (default config).
_EXPECTED_COUNTS = {"train": 9647, "test": 5000, "qual_dev": 100, "qual_test": 100}


def _prep_split(ds, split: str, out_root: Path, *, strict: bool) -> Path:
    n = len(ds)
    print(f"[{split}] {n} examples")
    if strict and split in _EXPECTED_COUNTS and n != _EXPECTED_COUNTS[split]:
        raise SystemExit(
            f"[{split}] count {n} != expected {_EXPECTED_COUNTS[split]}. "
            "Wrong config? The default google/docci train is 9,647; DOCCI-AAR is 4,932."
        )

    img_dir = out_root / f"images_{split}"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for idx, ex in enumerate(ds):
        example_id = str(ex["example_id"])
        if strict and not example_id.startswith(split):
            raise SystemExit(
                f"[{split}] row {idx} example_id={example_id!r} does not start with "
                f"{split!r}; the DatasetDict may be shuffled or the wrong config."
            )
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        rel = f"{idx:012d}.jpg"
        img.save(img_dir / rel, format="JPEG", quality=95)
        rows.append(
            {
                "image_id": idx,
                "image_path": str(img_dir / rel),
                "example_id": example_id,
                "caption": str(ex["description"]),
            }
        )
        if (idx + 1) % 500 == 0:
            print(f"[{split}] wrote {idx + 1}/{n} images")

    manifest_path = out_root / f"{split}_manifest.json"
    save_json(rows, manifest_path)
    refs = {r["image_id"]: [r["caption"]] for r in rows}
    save_json(refs, out_root / f"{split}_references.json")
    print(f"[{split}] manifest -> {manifest_path} ({len(rows)} rows)")
    return manifest_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-id", default="google/docci", help="HF dataset id (default config).")
    ap.add_argument("--out-root", default="data/docci")
    ap.add_argument("--splits", default="train,test", help="Comma list of splits to materialize.")
    ap.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Disable the partition-count / example_id-prefix guards.",
    )
    ap.set_defaults(strict=True)
    args = ap.parse_args()

    from datasets import load_dataset

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
        print(f"loading {args.hf_id} split={split} ...")
        ds = load_dataset(args.hf_id, split=split)
        _prep_split(ds, split, out_root, strict=args.strict)

    print("done.")


if __name__ == "__main__":
    main()
