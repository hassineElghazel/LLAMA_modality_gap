"""Build demo/cache.json: real model captions + live-computed CLIPScore/CHAIR.

Captions are NEVER written by hand. They come from the thesis dd256 prediction
files, i.e. the same generations that produced every number in the thesis:

    outputs/predictions/captions_C3pinr_dd256.json    (all-pinned baseline)
    outputs/predictions/captions_Cloc_80k_dd256.json  (Cloc_80k, 2,500 steps)

CLIPScore and the CHAIR fields are recomputed here with the demo's own scorers,
so the cached panel and the live panel agree by construction.

    python demo/build_cache.py                       # uses demo/images
    python demo/build_cache.py --image-root data/coco/val2017   # on the cluster
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
from common import CURATED, MODELS, Chair, ClipScorer, image_path   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", default=None,
                    help="prefer COCO originals here; falls back to demo/images")
    ap.add_argument("--out", default=str(ROOT / "demo" / "cache.json"))
    args = ap.parse_args()

    preds = {}
    for label, fname in MODELS.items():
        p = ROOT / "outputs" / "predictions" / fname
        if not p.exists():
            print(f"!! missing {p}", file=sys.stderr); return 2
        preds[label] = {d["image_id"]: d["caption"] for d in json.loads(p.read_text())}

    chair, clip = Chair(), ClipScorer()
    root = Path(args.image_root) if args.image_root else None
    entries, t0 = {}, time.time()

    for iid, caption in CURATED:
        src = root / f"{iid:012d}.jpg" if root and (root / f"{iid:012d}.jpg").exists() \
              else image_path(iid)
        if not src.exists():
            print(f"!! no image for {iid}", file=sys.stderr); return 2
        img = Image.open(src).convert("RGB")
        rec = {"image_id": iid, "title": caption, "image_source": str(src),
               "models": {}}
        for label in MODELS:
            cap = preds[label].get(iid)
            if cap is None:
                print(f"!! {label} has no caption for {iid}", file=sys.stderr); return 2
            ch = chair.score(iid, cap)
            rec["models"][label] = {
                "caption": cap,
                "clipscore": round(clip.score(img, cap), 4),
                "words": ch["words"], "H": ch["H"],
                "recall": None if ch["recall"] is None else round(ch["recall"], 4),
                "n_gt": ch["n_gt"], "hallucinated": ch["hallucinated"],
                "named": ch["named"], "gt": ch["gt"],
            }
        entries[str(iid)] = rec
        print(f"  {iid:7d}  " + "  ".join(
            f"{l.split()[0]}: CLIP {rec['models'][l]['clipscore']:.3f} H {rec['models'][l]['H']:2d}"
            for l in MODELS))

    out = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": {
            "captions": {l: f"outputs/predictions/{f}" for l, f in MODELS.items()},
            "note": "captions are verbatim thesis dd256 generations; "
                    "CLIPScore and CHAIR recomputed by demo/build_cache.py",
            "clip_model": ClipScorer.MODEL, "clip_w": ClipScorer.W,
            "chair_gt_source": chair.gt_source,
            "image_root": str(root) if root else str(ROOT / "demo" / "images"),
        },
        "entries": entries,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}  ({len(entries)} images, {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
