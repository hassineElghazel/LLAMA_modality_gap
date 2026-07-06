"""Directly measure image overlap between each eval set and the images the model
ACTUALLY trained on (the seed-42 shuffled first N LLaVA items), not just the
train/val split in principle. Zero overlap => provably clean.

Run from the repo root:
    python scripts/check_eval_overlap.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from pathlib import Path


def coco_id(name: str):
    d = re.findall(r"\d+", str(name))
    return int(d[-1]) if d else None


def trained_image_ids(llava_dir: str, seed: int, n_items: int):
    p = os.path.join(llava_dir, "llava_instruct_150k.json")
    if not os.path.exists(p):
        cands = sorted(glob.glob(os.path.join(llava_dir, "*.json")))
        if not cands:
            return None, None
        p = cands[0]
    rows = json.load(open(p))
    full = {coco_id(r.get("image", "")) for r in rows if r.get("image")}
    full.discard(None)
    random.Random(seed).shuffle(rows)
    subset = rows[:n_items]
    trained = {coco_id(r.get("image", "")) for r in subset if r.get("image")}
    trained.discard(None)
    return trained, full


def val2017_ids(instances: str):
    if not os.path.exists(instances):
        return None
    return {im["id"] for im in json.load(open(instances))["images"]}


def gqa_coco_ids(questions: str, vg_image_data: str):
    if not (os.path.exists(questions) and os.path.exists(vg_image_data)):
        return None
    vg2coco = {}
    for e in json.load(open(vg_image_data)):
        if e.get("coco_id") is not None:
            vg2coco[str(e["image_id"])] = int(e["coco_id"])
    qs = json.load(open(questions))
    ids = set()
    for v in qs.values():
        c = vg2coco.get(str(v["imageId"]))
        if c is not None:
            ids.add(c)
    return ids  # only the COCO-derived GQA images; non-COCO VG images can't overlap


def report(name, eval_ids, trained, full):
    if eval_ids is None:
        print(f"  {name:28s} -- data not found, skipped")
        return
    ot = len(eval_ids & trained) if trained else None
    of = len(eval_ids & full) if full else None
    msg = f"  {name:28s} images={len(eval_ids):6d}"
    if trained is not None:
        msg += f"  overlap_with_TRAINED={ot}"
    if full is not None:
        msg += f"  overlap_with_FULL150K={of}"
    verdict = "CLEAN" if (trained is not None and ot == 0) else ("CONTAMINATED" if ot else "?")
    print(msg + f"   -> {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llava-dir", default="data/llava_instruct_150k")
    ap.add_argument("--val2017-instances", default="data/coco/annotations/instances_val2017.json")
    ap.add_argument("--gqa-questions", default="data/gqa/testdev_balanced_questions.json")
    ap.add_argument("--vg-image-data", default="data/gqa/image_data.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-items", type=int, default=14400)
    args = ap.parse_args()

    trained, full = trained_image_ids(args.llava_dir, args.seed, args.n_items)
    if trained is None:
        print("LLaVA json not found; cannot compute trained image set."); return
    print(f"trained items={args.n_items}  unique trained images={len(trained)}  "
          f"full-150K unique images={len(full)}\n")
    print("eval-set overlap with the images the model actually trained on:")
    report("captioning/POPE/VQAv2 (val2017)", val2017_ids(args.val2017_instances), trained, full)
    report("GQA (COCO-derived images)", gqa_coco_ids(args.gqa_questions, args.vg_image_data), trained, full)
    print("\n(val2017-based evals should show overlap 0 = CLEAN. GQA shows its real "
          "COCO-image overlap; non-COCO VG images cannot overlap by definition.)")


if __name__ == "__main__":
    main()
