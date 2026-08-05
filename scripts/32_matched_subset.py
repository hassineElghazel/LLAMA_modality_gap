"""Build a MATCHED evaluation subset: images where EVERY arm terminated its caption.

Why this exists. On DOCCI the fine-tuned arms run past the 256-token cap on
~half the images, and those run-on captions are degenerate (distinct_4 ~0.34,
100% looping). Crucially the arms run on at DIFFERENT rates -- location 45.8%,
vanilla 48.6%, pinned 57.2% -- so a whole-set comparison partly measures which
arm degenerated more often rather than how well each grounds its description.
Conditional on terminating, the arms write equally well (distinct_4 0.872 /
0.875 / 0.880), so restricting to images where ALL arms terminated compares
like with like.

The subset is selected on OUTCOME, so absolute scores on it are optimistic
(easy images are over-represented). It is FAIR BETWEEN ARMS -- identical images,
identical count -- which is what the geometry contrast needs. Report it as a
matched-subset analysis, never as the headline number.

Writes ``captions_<tag>_matched.json`` per tag, so the existing scorers run
unchanged:

    python scripts/15_clipscore.py  --config configs/docci_eval.yaml --condition <tag>_matched ...
    python scripts/15b_siglip_score.py --config configs/docci_eval.yaml --condition <tag>_matched ...
    python scripts/28_score_docci_refs.py --config configs/docci_eval.yaml --tag <tag>_matched --no-spice

Termination is decided by the SAME predicate as scripts/31 (imported, not
re-implemented, so the two can never disagree).

Usage:
    python scripts/32_matched_subset.py --tags arm0_docci vanilla_docci pinned_docci location_docci
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_caption_quality", Path(__file__).with_name("31_caption_quality.py"))
_cq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cq)          # caption_stats(), summarise()


def load_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    if isinstance(rows, dict):                       # {image_id: caption}
        rows = [{"image_id": int(k), "caption": v} for k, v in rows.items()]
    return [r for r in rows if isinstance(r, dict) and "image_id" in r]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--predictions-dir", default="outputs/predictions")
    p.add_argument("--suffix", default="_matched")
    p.add_argument("--include", nargs="*", default=None,
                   help="Tags that must terminate for an image to qualify "
                        "(default: all --tags). Use to exclude a reference arm "
                        "from the gating while still writing its subset file.")
    p.add_argument("--out-summary", default="outputs/metrics/matched_subset.json")
    p.add_argument("--dry-run", action="store_true", help="Report sizes, write nothing.")
    args = p.parse_args()

    pdir = Path(args.predictions_dir)
    by_tag: dict[str, dict[int, str]] = {}
    stopped: dict[str, set[int]] = {}

    for tag in args.tags:
        path = pdir / f"captions_{tag}.json"
        if not path.exists():
            print(f"[matched] SKIP {tag}: {path} not found")
            continue
        rows = load_rows(path)
        by_tag[tag] = {int(r["image_id"]): str(r["caption"]) for r in rows}
        stopped[tag] = {
            int(r["image_id"]) for r in rows
            if not _cq.caption_stats(str(r["caption"]))["truncated"]
        }
        print(f"[matched] {tag:22} {len(rows):5d} captions, "
              f"{len(stopped[tag]):5d} terminated ({100*len(stopped[tag])/max(len(rows),1):.1f}%)")

    if not by_tag:
        print("[matched] nothing to do")
        return

    gating = args.include if args.include else list(by_tag)
    missing = [t for t in gating if t not in stopped]
    if missing:
        print(f"[matched] FATAL: gating tags absent: {missing}")
        return

    keep = set.intersection(*(stopped[t] for t in gating))
    # every written arm must actually have a caption for each kept id
    for tag in by_tag:
        keep &= set(by_tag[tag])
    keep_sorted = sorted(keep)

    print(f"\n[matched] gating on: {gating}")
    print(f"[matched] MATCHED SUBSET n = {len(keep_sorted)} "
          f"({100*len(keep_sorted)/max(len(next(iter(by_tag.values()))),1):.1f}% of the eval set)")
    if len(keep_sorted) < 300:
        print("[matched] WARNING: subset is small; differences will be hard to resolve.")

    summary = {"gating_tags": gating, "n_matched": len(keep_sorted),
               "terminated_per_tag": {t: len(s) for t, s in stopped.items()},
               "image_ids": keep_sorted}

    hdr = f"\n{'condition':22} {'n':>6} {'words':>7} {'distinct4':>10} {'loop%':>7}"
    print(hdr)
    print("-" * (len(hdr) - 1))
    for tag, caps in by_tag.items():
        texts = [caps[i] for i in keep_sorted]
        s = _cq.summarise(texts)
        summary.setdefault("quality_on_subset", {})[tag] = {
            k: v for k, v in s.items() if k not in ("stopped", "ran_on")}
        print(f"{tag:22} {s['n']:6d} {s['mean_words']:7.1f} "
              f"{s['distinct_4']:10.3f} {100*s['looping_rate']:6.1f}%")
        if not args.dry_run:
            out = pdir / f"captions_{tag}{args.suffix}.json"
            out.write_text(json.dumps([{"image_id": i, "caption": caps[i]} for i in keep_sorted]))

    if args.dry_run:
        print("\n[matched] dry run - no files written")
        return

    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).write_text(json.dumps(summary, indent=2))
    print(f"\n[matched] wrote captions_<tag>{args.suffix}.json for {len(by_tag)} tags")
    print(f"[matched] summary -> {args.out_summary}")
    print("[matched] now re-score with --condition <tag>" + args.suffix)


if __name__ == "__main__":
    main()
