"""Recover NoCaps reference captions for the OOD eval images and emit a
diagnostic manifest (image+caption pairs) for the modality-gap pipeline.

``scripts/prep_ood_eval.py`` built the 300-image OOD set for CLIPScore, which is
reference-free -- so it wrote ``annotations: []`` and kept only the images. The
gap measurement (``07_extract_projected.py`` -> ``03_compute_gap.py``) instead
needs a TEXT cloud: one caption per image, embedded through the frozen LLaMA
lookup. The connector is never applied to text, so the text embeddings are
identical across conditions; the caption set only anchors the text centroid, and
BOTH models must share it. Human NoCaps references are the principled anchor
(mirrors the COCO diagnostic manifest, which pairs each image with a human
caption), so we recover them here.

We reproduce the EXACT streaming shuffle of prep_ood_eval.py (same dataset,
split, seed, buffer_size) so the enumerate index == the ``source_row`` recorded
per image in ``ood_eval_<n>.json``. Matching on that index gives us each kept
image's captions with zero dependence on NoCaps' own ids.

Output (COCO-diagnostic format read by ``load_diagnostic_manifest``):
    [{"image_id": int, "image_path": "<image_root>/<id:012d>.jpg",
      "caption_id": int, "caption": str}, ...]

The same recovered references (as a captions_val2017-style annotations file) also
feed BLEU/CIDEr (``09_score_captions.py``) and CHAIR (``19_chair.py``); pass
``--also-coco-json`` to additionally write that.

Example:
    python scripts/prep_ood_refs.py --hf-dataset lmms-lab/NoCaps \
        --hf-split validation --out-root data/nocaps --n 300 --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pick_caption_column(rec: dict) -> str:
    """NoCaps parquet stores refs as a list-of-str column (``annotations_captions``).
    Detect it structurally, then fall back to a name heuristic."""
    for name, val in rec.items():
        if isinstance(val, (list, tuple)) and val and all(isinstance(x, str) for x in val):
            if "caption" in name.lower() or "annotation" in name.lower():
                return name
    # structural fallback: any list-of-str column
    for name, val in rec.items():
        if isinstance(val, (list, tuple)) and val and all(isinstance(x, str) for x in val):
            return name
    # last resort: a single-string caption column
    for name, val in rec.items():
        if isinstance(val, str) and "caption" in name.lower():
            return name
    raise SystemExit(f"[fatal] no caption column found; row keys = {list(rec)}")


def _captions_of(rec: dict, col: str) -> list[str]:
    v = rec[col]
    if isinstance(v, str):
        return [v]
    return [c for c in v if isinstance(c, str) and c.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dataset", default="lmms-lab/NoCaps",
                   help="HuggingFace dataset id used by prep_ood_eval.py")
    p.add_argument("--hf-split", default="validation")
    p.add_argument("--out-root", type=Path, default=Path("data/nocaps"))
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--also-coco-json", action="store_true",
                   help="also write a captions_val2017-style file (all refs) for BLEU/CHAIR")
    args = p.parse_args()

    eval_json = args.out_root / f"ood_eval_{args.n}.json"
    if not eval_json.exists():
        raise SystemExit(f"[fatal] {eval_json} missing -- run prep_ood_eval.py first")
    blob = json.loads(eval_json.read_text())
    img_root = args.out_root / "images"

    # source_row (enumerate index in the shuffled stream) -> our image id.
    row_to_id: dict[int, int] = {}
    id_to_file: dict[int, str] = {}
    for im in blob["images"]:
        if "source_row" not in im:
            raise SystemExit("[fatal] ood_eval json has no 'source_row' -- "
                             "rebuild it with the current prep_ood_eval.py")
        row_to_id[int(im["source_row"])] = int(im["id"])
        id_to_file[int(im["id"])] = im["file_name"]
    wanted = set(row_to_id)
    max_row = max(wanted)

    # Reproduce prep_ood_eval.py's stream EXACTLY so enumerate index lines up.
    from datasets import load_dataset
    ds = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=500)

    caps_by_id: dict[int, list[str]] = {}
    cap_col: str | None = None
    for idx, rec in enumerate(ds):
        if idx > max_row:
            break
        if idx not in wanted:
            continue
        if cap_col is None:
            cap_col = _pick_caption_column(rec)
            print(f"[refs] caption column = '{cap_col}'")
        caps = _captions_of(rec, cap_col)
        if caps:
            caps_by_id[row_to_id[idx]] = caps

    missing = sorted(set(id_to_file) - set(caps_by_id))
    if missing:
        raise SystemExit(f"[fatal] {len(missing)} images got no caption "
                         f"(ids {missing[:8]}...); stream replay may have drifted "
                         f"(datasets version / buffer_size / seed must match prep_ood_eval.py)")

    # Diagnostic manifest: ONE caption per image (first non-empty -> deterministic).
    manifest = []
    cap_id = 0
    for iid in sorted(id_to_file):
        manifest.append({
            "image_id": iid,
            "image_path": str(img_root / id_to_file[iid]),
            "caption_id": cap_id,
            "caption": caps_by_id[iid][0].strip(),
        })
        cap_id += 1
    out_manifest = args.out_root / f"ood_manifest_{args.n}.json"
    out_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"[ok] diagnostic manifest ({len(manifest)} pairs) -> {out_manifest}")

    if args.also_coco_json:
        # captions_val2017-style: images + ALL refs (BLEU/CIDEr/CHAIR want every ref).
        anns = []
        aid = 0
        for iid in sorted(id_to_file):
            for c in caps_by_id[iid]:
                anns.append({"id": aid, "image_id": iid, "caption": c.strip()})
                aid += 1
        coco = {"images": [{"id": iid, "file_name": id_to_file[iid]}
                           for iid in sorted(id_to_file)],
                "annotations": anns}
        out_coco = args.out_root / f"ood_eval_{args.n}_refs.json"
        out_coco.write_text(json.dumps(coco, indent=2))
        print(f"[ok] COCO-format refs ({len(anns)} captions) -> {out_coco}")


if __name__ == "__main__":
    main()
