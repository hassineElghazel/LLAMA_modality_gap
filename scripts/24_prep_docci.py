"""Materialize DOCCI train/test splits for the LLaVA 3-arm study.

Downloads DOCCI's official public files directly from Google Cloud Storage and
parses them ourselves, so this step depends on NO particular ``datasets``
version (``datasets>=3`` dropped script-based loaders, and ``google/docci`` is a
script dataset -- ``load_dataset("google/docci")`` fails with
"Dataset scripts are no longer supported").

Source (CC BY 4.0, https://google.github.io/docci/):
    https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines
    https://storage.googleapis.com/docci/data/docci_images.tar.gz   (all 15k images)

The descriptions JSONlines has one object per line with fields
``example_id`` / ``split`` / ``image_file`` / ``description``; inside the tar each
image lives at ``images/<image_file>``.

DOCCI standard partitions (default config, NOT DOCCI-AAR):
    train 9,647  |  test 5,000  |  qual_dev 100  |  qual_test 100

For each requested split we:
  * save every image as zero-padded ``{idx:012d}.jpg`` under
    ``<out_root>/images_<split>/`` so the COCO-style scorers (15_clipscore,
    15b_siglip) resolve them via ``image_root / f"{id:012d}.jpg"`` unchanged;
  * write ``<out_root>/<split>_manifest.json`` = list of
    ``{"image_id", "image_path", "example_id", "caption"}`` with
    ``image_id`` = the split-local enumeration index (0..N-1, in JSONlines order);
  * write ``<out_root>/<split>_references.json`` = ``{image_id: [caption]}``
    (single gold ref) for the reference-based scorer (28_score_docci_refs).

Contamination / partition guards (hard asserts, strict on by default):
  * per-split row counts match the standard partition;
  * every ``example_id`` in a split starts with that split's name;
  * every needed image is actually recovered from the tar.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image

from src.utils.io import save_json

_BASE_URL = "https://storage.googleapis.com/docci/data/"
_DESCR_FILE = "docci_descriptions.jsonlines"
_IMAGES_FILE = "docci_images.tar.gz"

# Standard DOCCI partition sizes (default config).
_EXPECTED_COUNTS = {"train": 9647, "test": 5000, "qual_dev": 100, "qual_test": 100}


def _download(url: str, dest: Path, *, force: bool = False) -> Path:
    """Stream ``url`` to ``dest`` (skips if a non-empty file already exists)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"[dl] {dest.name} already present ({dest.stat().st_size/1e6:.1f} MB) -- skip")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[dl] {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "docci-prep/1.0"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        next_mark = 0
        while True:
            chunk = resp.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if done >= next_mark:
                pct = f" ({100*done/total:.0f}%)" if total else ""
                print(f"[dl]   {done/1e6:8.1f} MB{pct}", flush=True)
                next_mark = done + (100 << 20)  # every ~100 MiB
    tmp.replace(dest)
    print(f"[dl] done {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


def _load_descriptions(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _build_rows(examples: list[dict], split: str, out_root: Path, *, strict: bool) -> list[dict]:
    """Split-local rows in JSONlines order; image_path is deterministic (filled by tar pass)."""
    img_dir = out_root / f"images_{split}"
    rows: list[dict] = []
    for ex in examples:
        if ex.get("split") != split:
            continue
        example_id = str(ex["example_id"])
        if strict and not example_id.startswith(split):
            raise SystemExit(
                f"[{split}] example_id={example_id!r} does not start with {split!r}."
            )
        idx = len(rows)
        rows.append(
            {
                "image_id": idx,
                "image_path": str(img_dir / f"{idx:012d}.jpg"),
                "example_id": example_id,
                "caption": str(ex["description"]),
                "image_file": str(ex["image_file"]),  # tar key; stripped before save
            }
        )
    n = len(rows)
    print(f"[{split}] {n} examples")
    if strict and split in _EXPECTED_COUNTS and n != _EXPECTED_COUNTS[split]:
        raise SystemExit(
            f"[{split}] count {n} != expected {_EXPECTED_COUNTS[split]}. "
            "Wrong file? default google/docci train is 9,647; DOCCI-AAR is 4,932."
        )
    return rows


def _save_image(fobj, dest: Path) -> None:
    img = Image.open(io.BytesIO(fobj.read()))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(dest, format="JPEG", quality=95)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="data/docci")
    ap.add_argument("--splits", default="train,test", help="Comma list of splits to materialize.")
    ap.add_argument("--base-url", default=_BASE_URL)
    ap.add_argument("--raw-dir", default=None, help="Where to cache the downloads (default <out_root>/_raw).")
    ap.add_argument("--rm-tar", action="store_true", help="Delete the images tarball after extraction.")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Disable the partition-count / example_id-prefix / image-recovery guards.",
    )
    ap.set_defaults(strict=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir) if args.raw_dir else out_root / "_raw"
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    # --- descriptions (small) ---
    descr_path = _download(args.base_url + _DESCR_FILE, raw_dir / _DESCR_FILE, force=args.force_download)
    examples = _load_descriptions(descr_path)
    print(f"[descr] {len(examples)} rows across all splits")

    # --- build manifests/references up front (image_path is deterministic) ---
    rows_by_split: dict[str, list[dict]] = {}
    for split in splits:
        rows = _build_rows(examples, split, out_root, strict=args.strict)
        (out_root / f"images_{split}").mkdir(parents=True, exist_ok=True)
        rows_by_split[split] = rows

    # image_file basename -> (split, dest_path); track which still need saving
    need: dict[str, Path] = {}
    for split, rows in rows_by_split.items():
        for r in rows:
            dest = Path(r["image_path"])
            if not (dest.exists() and dest.stat().st_size > 0):
                need[os.path.basename(r["image_file"])] = dest

    # --- images: only pull the tar if something is still missing ---
    if need:
        print(f"[img] {len(need)} images to extract from tarball")
        tar_path = _download(args.base_url + _IMAGES_FILE, raw_dir / _IMAGES_FILE, force=args.force_download)
        saved = 0
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                key = os.path.basename(member.name)
                dest = need.pop(key, None)
                if dest is None:
                    continue
                fobj = tar.extractfile(member)
                if fobj is None:
                    continue
                _save_image(fobj, dest)
                saved += 1
                if saved % 500 == 0:
                    print(f"[img] saved {saved} images (remaining {len(need)})", flush=True)
        print(f"[img] saved {saved} images; {len(need)} still missing")
        if args.strict and need:
            missing = list(need)[:5]
            raise SystemExit(
                f"[img] {len(need)} needed images not found in tarball (e.g. {missing}). "
                "The descriptions and images archive may be out of sync."
            )
        if args.rm_tar:
            tar_path.unlink(missing_ok=True)
            print(f"[img] removed {tar_path}")
    else:
        print("[img] all target images already present -- skipping tarball download")

    # --- write manifests + references (drop the internal image_file key) ---
    for split, rows in rows_by_split.items():
        clean = [{k: v for k, v in r.items() if k != "image_file"} for r in rows]
        manifest_path = out_root / f"{split}_manifest.json"
        save_json(clean, manifest_path)
        refs = {r["image_id"]: [r["caption"]] for r in clean}
        save_json(refs, out_root / f"{split}_references.json")
        print(f"[{split}] manifest -> {manifest_path} ({len(clean)} rows)")

    print("done.")


if __name__ == "__main__":
    main()
