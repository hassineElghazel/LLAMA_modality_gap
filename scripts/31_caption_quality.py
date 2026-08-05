"""Measure degenerate repetition + truncation in generated captions.

Motivated by an observation on the DOCCI arms: on the SAME image, the pinned and
location arms both looped ("The line is painted in a zigzag pattern. The line is
painted in a red color. The line is painted on the sidewalk...") while vanilla
produced varied text. CLIPScore truncates at 77 tokens and SigLIP at 64, so
neither judge can see degeneration in captions that run to a 256-token cap --
a model can loop for three quarters of its output and lose nothing on either
reference-free metric. This script measures what those metrics are blind to.

Metrics per caption (word-level, lowercased, punctuation-stripped):
  distinct_4      unique 4-grams / total 4-grams. 1.0 = no repetition.
                  The standard diversity measure; low values mean looping.
  max_4gram_reps  how many times the most-repeated 4-gram occurs. >=3 is a
                  strong loop signal (a phrase recurring three times is rarely
                  natural description).
  looping         max_4gram_reps >= 3.
  truncated       does not end in terminal punctuation -> generation hit the
                  token cap mid-sentence rather than choosing to stop.
  n_words

Aggregates are means/rates over the file. Reference captions can be passed with
--refs to give a human-written floor for distinct_4 -- crucial context, since
detail captions legitimately repeat some phrasing and the absolute number means
little without it.

Pure stdlib: runs on a login node, no GPU, no torch.

Usage:
    python scripts/31_caption_quality.py --tags arm0_docci vanilla_docci pinned_docci location_docci
    python scripts/31_caption_quality.py --tags vanilla_docci --refs data/docci/test_manifest.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

WORD_RE = re.compile(r"[a-z0-9']+")
TERMINAL = '.!?"”'


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def caption_stats(text: str, n: int = 4) -> dict:
    w = words(text)
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    if grams:
        counts = Counter(grams)
        distinct = len(counts) / len(grams)
        max_reps = max(counts.values())
    else:
        distinct, max_reps = 1.0, 0
    stripped = text.rstrip()
    return {
        "n_words": len(w),
        "distinct_4": distinct,
        "max_4gram_reps": max_reps,
        "looping": max_reps >= 3,
        "truncated": bool(stripped) and stripped[-1] not in TERMINAL,
    }


def _agg(rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    return {
        "n": len(rows),
        "mean_words": sum(r["n_words"] for r in rows) / n,
        "distinct_4": sum(r["distinct_4"] for r in rows) / n,
        "looping_rate": sum(r["looping"] for r in rows) / n,
        "truncated_rate": sum(r["truncated"] for r in rows) / n,
        "mean_max_reps": sum(r["max_4gram_reps"] for r in rows) / n,
    }


def summarise(texts: list[str]) -> dict:
    """Overall stats, plus the STOPPED/RAN-ON split.

    The split is the diagnostic that matters: a model that produces clean text
    whenever it terminates has a STOPPING problem (it runs past its content and
    then loops); a model that loops even in captions it chose to end has a
    COHERENCE problem. Those call for different fixes, and the aggregate number
    cannot tell them apart.
    """
    rows = [caption_stats(t) for t in texts]
    out = _agg(rows)
    out["stopped"] = _agg([r for r in rows if not r["truncated"]])
    out["ran_on"] = _agg([r for r in rows if r["truncated"]])
    return out


def load_captions(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):                       # {image_id: caption}
        return [str(v) for v in data.values()]
    out = []
    for row in data:
        if isinstance(row, str):
            out.append(row)
        elif isinstance(row, dict):
            for key in ("caption", "description", "text", "pred"):
                if key in row:
                    out.append(str(row[key]))
                    break
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tags", nargs="+", required=True, help="Caption tags, e.g. vanilla_docci")
    p.add_argument("--predictions-dir", default="outputs/predictions")
    p.add_argument("--refs", default=None,
                   help="DOCCI manifest json -> human-written floor for distinct_4.")
    p.add_argument("--out", default="outputs/metrics/caption_quality.json")
    args = p.parse_args()

    results: dict[str, dict] = {}

    if args.refs:
        try:
            rows = json.loads(Path(args.refs).read_text())
            results["GOLD (human)"] = summarise([str(r["caption"]) for r in rows])
        except Exception as exc:
            print(f"[quality] could not read refs: {exc}")

    for tag in args.tags:
        path = Path(args.predictions_dir) / f"captions_{tag}.json"
        if not path.exists():
            print(f"[quality] SKIP {tag}: {path} not found")
            continue
        results[tag] = summarise(load_captions(path))

    if not results:
        print("[quality] nothing to report")
        return

    hdr = f"{'condition':22} {'n':>6} {'words':>7} {'distinct4':>10} {'loop%':>7} {'trunc%':>7} {'maxreps':>8}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for tag, s in results.items():
        print(f"{tag:22} {s['n']:6d} {s['mean_words']:7.1f} {s['distinct_4']:10.3f} "
              f"{100*s['looping_rate']:6.1f}% {100*s['truncated_rate']:6.1f}% {s['mean_max_reps']:8.2f}")

    print("\nRead: distinct_4 near the GOLD row = human-like variety; well below it = looping.")
    print("      loop% = captions with a 4-gram occurring 3+ times.")
    print("      trunc% = captions that hit the token cap instead of ending a sentence.")

    # --- the diagnostic split: captions that STOPPED vs captions that RAN ON ---
    print("\n" + "=" * len(hdr))
    print("SPLIT: captions the model CHOSE to end, vs captions that hit the cap")
    print("=" * len(hdr))
    sub = f"{'condition':22} {'group':>9} {'n':>6} {'words':>7} {'distinct4':>10} {'loop%':>7}"
    print(sub)
    print("-" * len(sub))
    for tag, s in results.items():
        for grp in ("stopped", "ran_on"):
            g = s.get(grp) or {}
            if not g.get("n"):
                continue
            print(f"{tag:22} {grp:>9} {g['n']:6d} {g['mean_words']:7.1f} "
                  f"{g['distinct_4']:10.3f} {100*g['looping_rate']:6.1f}%")
    print("\nRead the STOPPED rows. Clean there (distinct_4 near GOLD) => the model writes")
    print("well and simply fails to stop in time: a STOPPING problem, fixable by decoding")
    print("or more training. Still looping there => COHERENCE is damaged, a deeper fault.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[quality] wrote {out}")


if __name__ == "__main__":
    main()
