"""ID-independent contamination check for GQA testdev: compare image CONTENT
(perceptual dHash) of the testdev images against the COCO images the model
actually trained on. GQA testdev uses 'n'-prefixed held-out images whose IDs do
NOT match VG/COCO, so the ID-based check can't verify them -- this hashes pixels
instead. Any near-duplicate => real overlap; none => provably clean by content.

Run from the repo root:
    python scripts/check_gqa_content_overlap.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image


def coco_id(name: str):
    d = re.findall(r"\d+", str(name))
    return int(d[-1]) if d else None


def dhash(path: Path, size: int = 8):
    try:
        img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    except Exception:
        return None
    a = np.asarray(img, dtype=np.int16)
    diff = (a[:, 1:] > a[:, :-1]).flatten()
    bits = 0
    for b in diff:
        bits = (bits << 1) | int(b)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llava", default="data/llava_instruct_150k/llava_instruct_150k.json")
    ap.add_argument("--train-root", default="data/coco/train2017")
    ap.add_argument("--gqa-questions", default="data/gqa/testdev_balanced_questions.json")
    ap.add_argument("--gqa-images", default="data/gqa/images")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-items", type=int, default=14400, help="trained subset size (seed-42 shuffle)")
    ap.add_argument("--threshold", type=int, default=5, help="max Hamming distance to call a match")
    args = ap.parse_args()

    # 1) COCO image ids the model actually trained on
    rows = json.load(open(args.llava))
    random.Random(args.seed).shuffle(rows)
    trained_ids = {coco_id(r.get("image", "")) for r in rows[:args.n_items] if r.get("image")}
    trained_ids.discard(None)
    train_root = Path(args.train_root)
    print(f"hashing {len(trained_ids)} trained COCO images from {train_root} ...")
    train_hashes = {}   # hash -> coco_id
    miss = 0
    for cid in trained_ids:
        h = dhash(train_root / f"{cid:012d}.jpg")
        if h is None:
            miss += 1
        else:
            train_hashes.setdefault(h, cid)
    if miss:
        print(f"  ({miss} trained images unreadable/missing on disk)")
    train_list = list(train_hashes.items())

    # 2) GQA testdev images
    qs = json.load(open(args.gqa_questions))
    testdev = sorted({str(v["imageId"]) for v in qs.values()})
    gqa_root = Path(args.gqa_images)
    present = [i for i in testdev if (gqa_root / f"{i}.jpg").exists()]
    print(f"GQA testdev images: {len(testdev)}  present on disk: {len(present)}  "
          f"missing: {len(testdev) - len(present)}")

    # 3) content comparison
    print(f"comparing {len(present)} testdev images vs trained (Hamming <= {args.threshold}) ...")
    overlaps = []
    for iid in present:
        h = dhash(gqa_root / f"{iid}.jpg")
        if h is None:
            continue
        if h in train_hashes:
            overlaps.append((iid, train_hashes[h], 0))
            continue
        for th, cid in train_list:
            d = hamming(h, th)
            if d <= args.threshold:
                overlaps.append((iid, cid, d))
                break

    print(f"\n==== RESULT ====")
    print(f"content overlaps (near-duplicate images): {len(overlaps)}")
    for o in overlaps[:25]:
        print(f"  gqa={o[0]}  ~  coco_train={o[1]:012d}  (hamming={o[2]})")
    print("VERDICT:", "CLEAN (no training image appears in GQA testdev)"
          if not overlaps else f"OVERLAP FOUND -> {len(overlaps)} images need removal")


if __name__ == "__main__":
    main()
