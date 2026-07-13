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

Three input modes (pass exactly one):
  --hf-dataset ID      load a HuggingFace image-caption dataset (e.g.
                       ``HuggingFaceM4/NoCaps``) and take its decoded images
                       directly. Most robust: the hub resolves hosting, so no raw
                       S3/URL guessing and no separate annotation file. RECOMMENDED.
  --nocaps-json PATH   official nocaps_val_4500_captions.json; images are fetched
                       from each entry's ``coco_url`` (NoCaps S3 mirror -- often
                       AccessDenied now, prefer --hf-dataset).
  --image-dir  PATH    a folder of already-downloaded natural images (*.jpg/*.png);
                       n are sampled and copied. Offline -- no network needed.

Deterministic: fixed seed picks the same n images every run.

Example (recommended):
    python scripts/prep_ood_eval.py --hf-dataset HuggingFaceM4/NoCaps \
        --hf-split validation --out-root data/nocaps --n 300 --seed 42
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


def _pick_image_column(ds) -> str:
    """Find the PIL-Image feature column (NoCaps uses ``image``)."""
    from datasets import Image as HFImage
    feats = getattr(ds, "features", None)
    if feats:
        for name, feat in feats.items():
            if isinstance(feat, HFImage):
                return name
        for name in feats:                       # fallback: name looks like image
            if "image" in name.lower():
                return name
    # streaming with no declared features: peek the first row
    try:
        probe = next(iter(ds))
        for name, val in probe.items():
            if isinstance(val, Image.Image):
                return name
        for name in probe:
            if "image" in name.lower():
                return name
    except Exception:
        pass
    raise SystemExit("[fatal] could not locate an image column")


def _save_rgb_jpg(src, dst: Path) -> bool:
    """Save an image to dst as RGB JPEG. Accepts a PIL.Image, raw bytes, or a path."""
    try:
        if isinstance(src, Image.Image):
            im = src
        elif isinstance(src, (bytes, bytearray)):
            import io
            im = Image.open(io.BytesIO(src))
        else:
            im = Image.open(src)
        im.convert("RGB").save(dst, "JPEG", quality=95)
        return True
    except Exception as e:  # corrupt download / unreadable file
        print(f"  [skip] {dst.name}: {e}")
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dataset", default=None,
                   help="HuggingFace dataset id, e.g. HuggingFaceM4/NoCaps (recommended)")
    p.add_argument("--hf-split", default="validation",
                   help="split for --hf-dataset (NoCaps: validation)")
    p.add_argument("--nocaps-json", type=Path, default=None,
                   help="official nocaps_val_4500_captions.json (download images via coco_url)")
    p.add_argument("--image-dir", type=Path, default=None,
                   help="folder of already-downloaded non-COCO images (offline mode)")
    p.add_argument("--out-root", type=Path, default=Path("data/nocaps"),
                   help="output root; writes <out-root>/images/ + <out-root>/ood_eval_<n>.json")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    n_modes = sum(x is not None for x in (args.hf_dataset, args.nocaps_json, args.image_dir))
    if n_modes != 1:
        p.error("pass exactly one of --hf-dataset, --nocaps-json, or --image-dir")

    rng = random.Random(args.seed)
    img_root = args.out_root / "images"
    img_root.mkdir(parents=True, exist_ok=True)

    images_meta: list[dict] = []
    next_id = 0
    # oversample so download/decode failures still leave us with exactly n.
    target = args.n

    if args.hf_dataset is not None:
        # STREAM the split: pull bytes only as consumed, so we never download the
        # other splits' shards. shuffle() uses a bounded buffer, so the 300 picks
        # are seed-deterministic without random-accessing the whole set.
        from datasets import load_dataset
        ds = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)
        col = _pick_image_column(ds)
        print(f"[hf] {args.hf_dataset}:{args.hf_split} (streaming)  image_col='{col}'")
        ds = ds.shuffle(seed=args.seed, buffer_size=2000)
        for src_row, rec in enumerate(ds):
            if next_id >= target:
                break
            dst = img_root / f"{next_id:012d}.jpg"
            try:
                im = rec[col]                # decoded to PIL on access
            except Exception as e:
                print(f"  [skip] streamed row {src_row}: {e}")
                continue
            if _save_rgb_jpg(im, dst):
                images_meta.append({"id": next_id, "file_name": dst.name,
                                    "source_row": int(src_row)})
                next_id += 1
    elif args.nocaps_json is not None:
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
