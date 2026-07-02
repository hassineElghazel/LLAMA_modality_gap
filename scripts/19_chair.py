"""CHAIR object-hallucination scoring for captioning conditions.

Measures how often the model *names objects that are not in the image* — the
standard object-hallucination metric (Rohrbach et al. 2018, "Object
Hallucination in Image Captioning", EMNLP). Two numbers per condition:

    CHAIR_i = hallucinated object mentions / all object mentions   (instance level)
    CHAIR_s = captions with >=1 hallucinated object / all captions (sentence level)

Ground truth per image is the UNION of (a) the 80 COCO-category segmentation
labels in ``instances_val2017.json`` and (b) the objects named across the 5
reference captions -- exactly Rohrbach's construction. If the instances file is
absent the score falls back to caption-only GT (flagged loudly in the output and
strictly more conservative -> higher, not lower, hallucination).

Because CHAIR alone is gamed by naming fewer objects, every condition also
reports object **recall** (of the objects actually present, how many were named),
objects mentioned per caption, and caption length -- the precision/recall/
productivity frame needed to read CHAIR honestly. All conditions are scored on
the identical image set in one pass, and every metric (plus its **paired**
difference vs. the baseline condition) carries a bootstrap 95% CI over images.

The synonym list (``scripts/chair_synonyms.txt``) and the double-word / toilet-seat
rules below are vendored verbatim from the original CHAIR release
(github.com/LisaAnne/Hallucination). This file reimplements ``caption_to_words``
self-contained (regex tokeniser + a COCO-tuned singulariser) so it needs neither
``nltk`` nor ``pattern`` on the cluster. Two documented deviations from upstream,
both applied symmetrically to GT and generated captions (so condition
comparisons are unaffected) and both strictly more correct: synonym tokens are
stripped/lower-cased (upstream leaks a leading space into ``' motor bike'`` and
keeps ``'iPhone'`` capitalised, so those never fire upstream).

Outputs: ``outputs/metrics/chair_summary.json`` (+ per-image records unless
``--no-save-records``).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from src.data.coco_val2017_loader import CocoVal2017Dataset
from src.utils.io import load_yaml, save_json


# --- COCO-tuned singulariser (substitutes pattern.en.singularize) -----------
# Irregulars + sibilant-stem plurals that a plain trailing-"s" rule mishandles,
# restricted to words that touch a COCO object or its synonyms.
_IRREGULAR = {
    "men": "man", "women": "woman", "children": "child", "people": "people",
    "geese": "goose", "mice": "mouse", "oxen": "ox",
    "knives": "knife", "pocketknives": "pocketknife",
    "buses": "bus", "glasses": "glass", "sandwiches": "sandwich",
    "benches": "bench", "toothbrushes": "toothbrush", "dishes": "dish",
    "scissors": "scissors", "skis": "ski", "sheep": "sheep",
}


def singularize(w: str) -> str:
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"          # puppies -> puppy
    if w.endswith("ss"):
        return w                     # glass / dress: already singular
    if w.endswith("s"):
        return w[:-1]                # horses -> horse, cats -> cat, planes -> plane
    return w


# --- Double-word rules (verbatim from CHAIR chair.py) -----------------------
_COCO_DOUBLE_WORDS = [
    "motor bike", "motor cycle", "air plane", "traffic light", "street light",
    "traffic signal", "stop light", "fire hydrant", "stop sign", "parking meter",
    "suit case", "sports ball", "baseball bat", "baseball glove", "tennis racket",
    "wine glass", "hot dog", "cell phone", "mobile phone", "teddy bear",
    "hair drier", "potted plant", "bow tie", "laptop computer", "stove top oven",
    "home plate", "train track",
]
_ANIMAL_WORDS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                 "bear", "zebra", "giraffe", "animal", "cub"]
_VEHICLE_WORDS = ["jet", "train"]


def _build_double_word_dict() -> dict[str, str]:
    d = {dw: dw for dw in _COCO_DOUBLE_WORDS}
    for a in _ANIMAL_WORDS:
        d[f"baby {a}"] = a
        d[f"adult {a}"] = a
    for v in _VEHICLE_WORDS:
        d[f"passenger {v}"] = v
    d["bow tie"] = "tie"
    d["toilet seat"] = "toilet"
    d["wine glas"] = "wine glass"
    return d


# 80 official COCO category names (used to assert synonym coverage of segments).
_COCO80 = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
}


def load_synonyms(path: Path) -> tuple[set[str], dict[str, str]]:
    """Return (mscoco_objects, inverse_synonym_dict). First token of each line is
    the canonical category; every token on the line maps to it. Tokens are
    stripped + lower-cased (fixes upstream leading-space / capitalisation leaks)."""
    objects: set[str] = set()
    inv: dict[str, str] = {}
    for line in path.read_text().splitlines():
        toks = [t.strip().lower() for t in line.split(",")]
        toks = [t for t in toks if t]
        if not toks:
            continue
        canonical = toks[0]
        for t in toks:
            objects.add(t)
            inv[t] = canonical
    missing = _COCO80 - set(inv)
    if missing:
        raise ValueError(f"synonym file missing COCO categories: {sorted(missing)}")
    return objects, inv


def caption_to_node_words(caption: str, objects: set[str], inv: dict[str, str],
                          dwd: dict[str, str]) -> tuple[list[str], int]:
    """Return (node_words, n_tokens): the canonical COCO objects named in the
    caption (with multiplicity) and the caption's word count. Mirrors CHAIR's
    caption_to_words: singularise -> merge double words -> toilet/seat rule ->
    map through the synonym dict."""
    tokens = re.findall(r"[a-z]+", caption.lower())
    words = [singularize(w) for w in tokens]

    merged: list[str] = []
    i = 0
    while i < len(words):
        double = " ".join(words[i:i + 2])
        if double in dwd:
            merged.append(dwd[double])
            i += 2
        else:
            merged.append(words[i])
            i += 1

    if ("toilet" in merged) and ("seat" in merged):
        merged = [w for w in merged if w != "seat"]

    node_words = [inv[w] for w in merged if w in objects]
    return node_words, len(tokens)


def build_gt(imids: list[int], references: dict[int, list[str]],
             objects: set[str], inv: dict[str, str], dwd: dict[str, str],
             instances_path: Path | None) -> tuple[dict[int, set[str]], str]:
    """Per-image GT object set = caption-derived objects (always) UNION
    segmentation categories (if instances_path exists). Returns (gt, source)."""
    gt: dict[int, set[str]] = {iid: set() for iid in imids}
    for iid in imids:
        for cap in references.get(iid, []):
            nodes, _ = caption_to_node_words(cap, objects, inv, dwd)
            gt[iid].update(nodes)

    source = "captions_only"
    if instances_path and instances_path.exists():
        inst = json.loads(instances_path.read_text())
        id_to_name = {c["id"]: c["name"].strip().lower() for c in inst["categories"]}
        want = set(imids)
        for ann in inst["annotations"]:
            iid = ann["image_id"]
            if iid in want:
                name = id_to_name[ann["category_id"]]
                gt[iid].add(inv[name])
        source = "segments+captions"
    return gt, source


def score_condition(preds: list[dict], gt: dict[int, set[str]],
                    objects: set[str], inv: dict[str, str], dwd: dict[str, str]):
    """Per-image record arrays for one condition, aligned to sorted imids."""
    by_imid = {p["image_id"]: p["caption"] for p in preds}
    imids = sorted(gt)
    recs = []
    for iid in imids:
        caption = by_imid[iid]
        nodes, n_tok = caption_to_node_words(caption, objects, inv, dwd)
        gt_set = gt[iid]
        node_set = set(nodes)
        n_hall = sum(1 for nw in nodes if nw not in gt_set)
        recs.append({
            "image_id": iid,
            "hallucinated": int(n_hall > 0),
            "n_hall_words": n_hall,
            "n_words": len(nodes),
            "n_gt": len(gt_set),
            "n_recalled": len(node_set & gt_set),
            "cap_len": n_tok,
            "gt_words": sorted(gt_set),
            "gen_words": nodes,
            "hall_words": [nw for nw in nodes if nw not in gt_set],
            "caption": caption,
        })
    return imids, recs


def _aggregate(hall, hw, nw, gt, rec, length) -> dict:
    nw_sum = float(nw.sum())
    gt_sum = float(gt.sum())
    return {
        "chair_s": float(hall.mean()),
        "chair_i": float(hw.sum() / nw_sum) if nw_sum else 0.0,
        "recall": float(rec.sum() / gt_sum) if gt_sum else 0.0,
        "objs_per_cap": float(nw.mean()),
        "avg_len": float(length.mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+", required=True,
                   help="condition tags (each needs predictions/captions_<tag>.json)")
    p.add_argument("--baseline", default=None,
                   help="condition for paired deltas (default: first in --conditions)")
    p.add_argument("--config", default="configs/captioning.yaml")
    p.add_argument("--predictions-dir", default="outputs/predictions")
    p.add_argument("--synonyms", default="scripts/chair_synonyms.txt")
    p.add_argument("--instances", default="data/coco/annotations/instances_val2017.json")
    p.add_argument("--out", default="outputs/metrics/chair_summary.json")
    p.add_argument("--records-dir", default="outputs/metrics")
    p.add_argument("--no-save-records", action="store_true")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    baseline = args.baseline or args.conditions[0]
    if baseline not in args.conditions:
        raise ValueError(f"baseline {baseline} not among --conditions {args.conditions}")

    objects, inv = load_synonyms(Path(args.synonyms))
    dwd = _build_double_word_dict()

    cap_cfg = load_yaml(args.config)
    ds = CocoVal2017Dataset(
        annotations_json=cap_cfg["eval_set"]["annotations_json"],
        image_root=cap_cfg["eval_set"]["image_root"],
    )
    references = ds.references()

    # Load predictions; enforce identical image sets across conditions.
    pred_dir = Path(args.predictions_dir)
    preds: dict[str, list[dict]] = {}
    imid_sets: dict[str, set[int]] = {}
    for cond in args.conditions:
        rows = json.loads((pred_dir / f"captions_{cond}.json").read_text())
        preds[cond] = rows
        imid_sets[cond] = {r["image_id"] for r in rows}
    shared = set.intersection(*imid_sets.values())
    for cond in args.conditions:
        extra = imid_sets[cond] - shared
        if extra:
            print(f"[warn] {cond}: {len(extra)} image_ids not shared by all "
                  f"conditions; scoring on the {len(shared)}-image intersection")
    imids = sorted(shared)

    gt, gt_source = build_gt(imids, references, objects, inv, dwd,
                             Path(args.instances))
    n_empty = sum(1 for iid in imids if not gt[iid])
    print(f"[chair] gt_source={gt_source}  images={len(imids)}  "
          f"empty_gt={n_empty} ({100*n_empty/max(len(imids),1):.1f}%)")
    if gt_source == "captions_only":
        print("[warn] instances_val2017.json not found -> caption-only GT "
              "(conservative: over-counts hallucination). Extract it from "
              "annotations_trainval2017.zip for the full Rohrbach metric.")

    # Restrict every condition to the shared imids and score.
    gt_shared = {iid: gt[iid] for iid in imids}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    records: dict[str, list[dict]] = {}
    for cond in args.conditions:
        c_imids, recs = score_condition(preds[cond], gt_shared, objects, inv, dwd)
        assert c_imids == imids
        records[cond] = recs
        arrays[cond] = {
            "hall": np.array([r["hallucinated"] for r in recs], dtype=np.float64),
            "hw": np.array([r["n_hall_words"] for r in recs], dtype=np.float64),
            "nw": np.array([r["n_words"] for r in recs], dtype=np.float64),
            "gt": np.array([r["n_gt"] for r in recs], dtype=np.float64),
            "rec": np.array([r["n_recalled"] for r in recs], dtype=np.float64),
            "len": np.array([r["cap_len"] for r in recs], dtype=np.float64),
        }

    # Point estimates.
    point = {c: _aggregate(a["hall"], a["hw"], a["nw"], a["gt"], a["rec"], a["len"])
             for c, a in arrays.items()}

    # Paired bootstrap over images: one shared resample per iteration.
    rng = np.random.default_rng(args.seed)
    N, B = len(imids), args.n_bootstrap
    idx = rng.integers(0, N, size=(B, N))
    metrics = ("chair_s", "chair_i", "recall", "objs_per_cap", "avg_len")
    boot: dict[str, dict[str, np.ndarray]] = {}
    for cond, a in arrays.items():
        hall, hw, nw, gt_a, rec, ln = (a["hall"][idx], a["hw"][idx], a["nw"][idx],
                                       a["gt"][idx], a["rec"][idx], a["len"][idx])
        nw_s, gt_s = nw.sum(1), gt_a.sum(1)
        boot[cond] = {
            "chair_s": hall.mean(1),
            "chair_i": np.divide(hw.sum(1), nw_s, out=np.zeros(B), where=nw_s > 0),
            "recall": np.divide(rec.sum(1), gt_s, out=np.zeros(B), where=gt_s > 0),
            "objs_per_cap": nw.mean(1),
            "avg_len": ln.mean(1),
        }

    def ci(x: np.ndarray) -> list[float]:
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    per_condition = {}
    for cond in args.conditions:
        entry = dict(point[cond])
        entry["ci95"] = {m: ci(boot[cond][m]) for m in metrics}
        if cond != baseline:
            entry["delta_vs_baseline"] = {
                m: {"delta": point[cond][m] - point[baseline][m],
                    "ci95": ci(boot[cond][m] - boot[baseline][m])}
                for m in metrics
            }
        per_condition[cond] = entry

    base_cs = point[baseline]["chair_s"]
    in_range = 0.05 <= base_cs <= 0.7
    examples = [
        {"image_id": r["image_id"], "caption": r["caption"],
         "gt_words": r["gt_words"], "gen_words": r["gen_words"],
         "hallucinated_words": r["hall_words"]}
        for r in records[baseline][:5]
    ]

    summary = {
        "config": {
            "conditions": args.conditions, "baseline": baseline,
            "gt_source": gt_source, "n_images": N,
            "n_bootstrap": B, "seed": args.seed,
            "synonyms": args.synonyms, "instances": args.instances,
        },
        "per_condition": per_condition,
        "sanity": {
            "n_images_empty_gt": n_empty,
            "baseline_chair_s": base_cs,
            "baseline_chair_s_in_expected_range": in_range,
            "note": ("baseline CHAIR_s outside the LLaVA-family ballpark "
                     "[0.05,0.7] -> check the parser") if not in_range else "ok",
            "examples_baseline": examples,
        },
    }
    save_json(summary, Path(args.out))

    if not args.no_save_records:
        for cond in args.conditions:
            slim = [{k: r[k] for k in ("image_id", "hallucinated", "n_hall_words",
                                       "n_words", "n_gt", "n_recalled", "cap_len")}
                    for r in records[cond]]
            save_json(slim, Path(args.records_dir) / f"chair_records_{cond}.json")

    # Console table.
    print(f"\n[chair] gt_source={gt_source}  N={N}  baseline={baseline}\n")
    hdr = f"{'condition':<14}{'CHAIR_s':>9}{'CHAIR_i':>9}{'recall':>9}{'obj/cap':>9}{'len':>7}"
    print(hdr); print("-" * len(hdr))
    for cond in args.conditions:
        e = point[cond]
        print(f"{cond:<14}{e['chair_s']:>9.3f}{e['chair_i']:>9.3f}"
              f"{e['recall']:>9.3f}{e['objs_per_cap']:>9.2f}{e['avg_len']:>7.1f}")
    print(f"\n[ok] wrote {args.out}")
    if not in_range:
        print(f"[warn] baseline CHAIR_s={base_cs:.3f} outside [0.05,0.7] -- verify parser")


if __name__ == "__main__":
    main()
