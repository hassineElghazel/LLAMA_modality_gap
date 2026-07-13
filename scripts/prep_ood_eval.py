"""Build a 100-image OUT-OF-DISTRIBUTION grounding eval set from NoCaps.

NoCaps (Agrawal et al., 2019) images come from Open Images and are DISJOINT from
COCO -- so a model trained on LLaVA-Instruct (COCO train2017) has never seen them.
This is the held-out-domain stress test: does Cloc's location-closure CLIPScore
advantage survive off the COCO distribution, or was it a COCO-specific artefact?

We need ONLY images: CLIPScore is reference-free (generated caption vs image), so
the NoCaps ground-truth captions are ignored and the output JSON carries an empty
``annotations`` list.

Output is written in the official COCO format expected by
``src/data/coco_val2017_loader.CocoVal2017Dataset``:

    { "images": [{"id": <int>, "file_name": "<int:012d>.jpg"}, ...],
      "annotations": [] }

Images are materialised as ``<image_root>/<id:012d>.jpg`` so that BOTH the loader
(reads ``file_name``) and ``scripts/15_clipscore.py`` (reconstructs ``{id:012d}.jpg``)
resolve every image with ZERO changes to the scoring code.

Two input modes:
  --nocaps-json PATH   official nocaps_val_4500_captions.json; images are fetched
                       from each entry's ``coco_url`` (NoCaps S3 mirror).
  --image-dir  PATH    a folder of already-downloaded natural images (*.jpg/*.png);
                       100 are sampled and copied. Use this if the cluster already
                       has NoCaps/Open-Images pulled down (no network needed).

Deterministic: fixed seed picks the same 100 images every run.

Example:
    python scripts/prep_ood_eval.py \
        --nocaps-json data/nocaps/nocaps_val_4500_captions.json \
        --out-root data/nocaps --n 100 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import urllib.request
from pathlib import Path

from PIL import Image


def _load_candidates_from_nocaps(nocaps_json: Path) -> list[tuple[str, str]]:
    """Return [(file_name, url), ...] for every NoCaps val image."""
    with nocaps_json.open() as f:
        blob = json.load(f)
    out: list[tuple[str, str]] = []
    for img in blob["images"]:
        url = img.get("coco_url") or img.get("url")
        if not url:
            continue
        out.append((img["file_name"], url))
    return out


def _load_candidates_from_dir(image_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in exts)


def _save_rgb_jpg(src_or_bytes, dst: Path) -> bool:
    """Open an image (path or raw bytes), convert to RGB, save as JPEG."""
    try:
        if isinstance(src_or_bytes, (bytes, bytearray)):
            import io
            im = Image.open(io.BytesIO(src_or_bytes))
        else:
            im = Image.open(src_or_bytes)
        im.convert("RGB").save(dst, "JPEG", quality=95)
        return True
    except Exception as e:  # corrupt download / unreadable file
        print(f"  [skip] {dst.name}: {e}")
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nocaps-json", type=Path, default=None,
                   help="official nocaps_val_4500_captions.json (download images via coco_url)")
    p.add_argument("--image-dir", type=Path, default=None,
                   help="folder of already-downloaded non-COCO images (offline mode)")
    p.add_argument("--out-root", type=Path, default=Path("data/nocaps"),
                   help="output root; writes <out-root>/images/ + <out-root>/ood_eval_<n>.json")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if (args.nocaps_json is None) == (args.image_dir is None):
        p.error("pass exactly one of --nocaps-json or --image-dir")

    rng = random.Random(args.seed)
    img_root = args.out_root / "images"
    img_root.mkdir(parents=True, exist_ok=True)

    images_meta: list[dict] = []
    next_id = 0
    # oversample so download/decode failures still leave us with exactly n.
    target = args.n

    if args.nocaps_json is not None:
        cands = _load_candidates_from_nocaps(args.nocaps_json)
        cands.sort(key=lambda t: t[0])              # stable order before shuffle
        rng.shuffle(cands)
        for file_name, url in cands:
            if next_id >= target:
                break
            dst = img_root / f"{next_id:012d}.jpg"
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = r.read()
            except Exception as e:
                print(f"  [skip] {file_name}: download failed ({e})")
                continue
            if _save_rgb_jpg(data, dst):
                images_meta.append({"id": next_id, "file_name": dst.name,
                                    "source_file": file_name})
                next_id += 1
    else:
        cands = _load_candidates_from_dir(args.image_dir)
        if len(cands) < target:
            p.error(f"--image-dir has only {len(cands)} images, need {target}")
        rng.shuffle(cands)
        for src in cands:
            if next_id >= target:
                break
            dst = img_root / f"{next_id:012d}.jpg"
            if _save_rgb_jpg(src, dst):
                images_meta.append({"id": next_id, "file_name": dst.name,
                                    "source_file": str(src)})
                next_id += 1

    if len(images_meta) < target:
        raise SystemExit(
            f"[fatal] only secured {len(images_meta)}/{target} images; "
            f"rerun (network) or point --image-dir at more files")

    out_json = args.out_root / f"ood_eval_{target}.json"
    with out_json.open("w") as f:
        json.dump({"images": images_meta, "annotations": []}, f, indent=2)

    print(f"[ok] {len(images_meta)} OOD images -> {img_root}")
    print(f"[ok] eval manifest -> {out_json}")
    print(f"     set configs/ood_eval.yaml: annotations_json={out_json}, "
          f"image_root={img_root}, num_images={target}")


if __name__ == "__main__":
    main()
