"""Per-image linkage: does connector geometry predict caption quality?

The condition-level comparison (C0/C1/C2/C3) only gives 4 data points and is
statistically inert — and worse, C2 and C3 are geometrically near-identical yet
behave oppositely, so no condition-level metric predicts captioning.

This script drops to the per-image level (n = up to 5000 within one condition)
and tests the hypothesis the literature points at (Decipher, 2025): downstream
quality is driven by *pairwise alignment* of each image with its own caption,
not by the global gap size.

For one condition it computes, per image:

    alignment_i = cos( image_pooled_i , text_pooled_i )      # geometry
    cider_i     = per-image CIDEr of the generated caption    # behaviour (graded)
    distinct1_i = unique-unigram ratio of the caption         # behaviour (ref-free)
    top1_frac_i = freq. of the most common token              # degeneracy signal

then reports Pearson + Spearman of alignment vs {CIDEr, degeneracy} and writes a
scatter plot + a per-image table.

IMPORTANT — the join is by ``image_id``, never by row position:
- embeddings are stored in the seed-42 diagnostic-manifest order;
- captions are written in sorted-image_id order.
Row i of the embedding tensor maps to ``manifest[i].image_id`` (extraction
iterates the manifest pairs in order and concatenates). We use that map to align
the two artifacts.

Usage:
    python scripts/11_link_gap_to_captions.py --condition C3_stage2
    python scripts/11_link_gap_to_captions.py --condition C2_stage1   # degeneracy only
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.data.coco_val2017_loader import CocoVal2017Dataset, load_diagnostic_manifest
from src.utils.io import load_yaml


# ---------------------------------------------------------------------------
# correlation helpers (numpy-only fallback so scipy is not a hard dependency)
# ---------------------------------------------------------------------------
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, ties handled (matches scipy.stats.rankdata 'average')."""
    order = a.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _corr_with_p(x: np.ndarray, y: np.ndarray) -> dict:
    """Pearson + Spearman, with a p-value when scipy is available."""
    out = {"pearson": _pearson(x, y), "spearman": _spearman(x, y), "n": int(len(x))}
    try:
        from scipy.stats import pearsonr, spearmanr
        if x.std() > 0 and y.std() > 0:
            out["pearson_p"] = float(pearsonr(x, y)[1])
            out["spearman_p"] = float(spearmanr(x, y)[1])
    except Exception:  # noqa: BLE001 — scipy optional
        pass
    return out


# ---------------------------------------------------------------------------
# per-image behaviour signals
# ---------------------------------------------------------------------------
def _degeneracy(caption: str) -> tuple[float, float, int]:
    """Reference-free signals. Returns (distinct1, top1_frac, n_tokens)."""
    toks = caption.split()
    n = len(toks)
    if n == 0:
        return 0.0, 1.0, 0
    distinct1 = len(set(toks)) / n
    top1_frac = Counter(toks).most_common(1)[0][1] / n
    return distinct1, top1_frac, n


def _per_image_cider(references: dict[int, list[str]], hyps: dict[int, str]) -> dict[int, float]:
    """Per-image CIDEr. Reuses pycocoevalcap's tokenizer; keeps the per-image
    array that ``score_predictions`` discards."""
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    ids = [iid for iid in hyps if iid in references]
    refs = {iid: [{"caption": c} for c in references[iid]] for iid in ids}
    hyp = {iid: [{"caption": hyps[iid]}] for iid in ids}
    tok = PTBTokenizer()
    refs_tok, hyps_tok = tok.tokenize(refs), tok.tokenize(hyp)

    # compute_score returns (avg, per_image_array); array order == dict key order
    _, per_image = Cider().compute_score(refs_tok, hyps_tok)
    keys = list(hyps_tok.keys())
    return {int(k): float(v) for k, v in zip(keys, per_image)}


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True,
                   help="tag shared by captions_<cond>.json and projected_<cond>_*.pt "
                        "(e.g. C3_stage2, C2_stage1, C1_stage2, C0_random)")
    p.add_argument("--emb-dir", default="outputs/embeddings")
    p.add_argument("--data-config", default="configs/data.yaml")
    p.add_argument("--captioning-config", default="configs/captioning.yaml")
    p.add_argument("--out-dir", default="outputs/metrics")
    p.add_argument("--fig-dir", default="outputs/figures/linkage")
    p.add_argument("--no-cider", action="store_true",
                   help="skip per-image CIDEr (use for C0/C2 where captions are noise; "
                        "degeneracy signals are the informative ones there)")
    args = p.parse_args()

    data_cfg = load_yaml(args.data_config)
    cap_cfg = load_yaml(args.captioning_config)

    # --- 1. row -> image_id map from the diagnostic manifest -----------------
    manifest_path = data_cfg["diagnostic_sample"]["manifest_path"]
    pairs = load_diagnostic_manifest(manifest_path)
    row_image_ids = [int(pr.image_id) for pr in pairs]

    # --- 2. load pooled embeddings (row order == manifest order) -------------
    emb_dir = Path(args.emb_dir)
    img = torch.load(emb_dir / f"projected_{args.condition}_image_pooled.pt", map_location="cpu")
    txt = torch.load(emb_dir / f"projected_{args.condition}_text_pooled.pt", map_location="cpu")
    img = img.to(torch.float64)
    txt = txt.to(torch.float64)
    if not (img.shape[0] == txt.shape[0] == len(row_image_ids)):
        raise ValueError(
            f"row mismatch: image={img.shape[0]} text={txt.shape[0]} "
            f"manifest={len(row_image_ids)} — embeddings and manifest disagree, "
            "cannot trust the row->image_id map.")

    # --- 3. per-image alignment cos(z_img, z_txt) ----------------------------
    img_n = img / img.norm(dim=1, keepdim=True).clamp(min=1e-12)
    txt_n = txt / txt.norm(dim=1, keepdim=True).clamp(min=1e-12)
    align = (img_n * txt_n).sum(dim=1).numpy()           # (N,)
    align_by_id = {iid: float(a) for iid, a in zip(row_image_ids, align)}

    # --- 4. load generated captions ------------------------------------------
    pred_path = Path(cap_cfg["output"]["predictions_dir"]) / f"captions_{args.condition}.json"
    with pred_path.open() as f:
        preds = json.load(f)
    hyps = {int(r["image_id"]): r["caption"] for r in preds}

    # --- 5. behaviour signals -------------------------------------------------
    cider_by_id: dict[int, float] = {}
    if not args.no_cider:
        ds = CocoVal2017Dataset(
            annotations_json=cap_cfg["eval_set"]["annotations_json"],
            image_root=cap_cfg["eval_set"]["image_root"],
        )
        cider_by_id = _per_image_cider(ds.references(), hyps)

    # --- 6. inner-join on image_id, assemble the table -----------------------
    rows = []
    for iid, cap in hyps.items():
        if iid not in align_by_id:
            continue
        d1, t1, ntok = _degeneracy(cap)
        rows.append({
            "image_id": iid,
            "alignment": align_by_id[iid],
            "cider": cider_by_id.get(iid),
            "distinct1": d1,
            "top1_frac": t1,
            "length": ntok,
        })
    if not rows:
        raise ValueError("no image_ids shared between embeddings and captions — "
                         "check the condition tag and that captioning is done.")

    a = np.array([r["alignment"] for r in rows])
    d1 = np.array([r["distinct1"] for r in rows])
    t1 = np.array([r["top1_frac"] for r in rows])

    summary = {
        "condition": args.condition,
        "n_joined": len(rows),
        "alignment_mean": float(a.mean()),
        "alignment_std": float(a.std()),
        "corr_alignment_vs_distinct1": _corr_with_p(a, d1),
        "corr_alignment_vs_top1frac": _corr_with_p(a, t1),
    }
    if not args.no_cider:
        c = np.array([r["cider"] for r in rows], dtype=np.float64)
        summary["cider_mean"] = float(c.mean())
        summary["corr_alignment_vs_cider"] = _corr_with_p(a, c)

    # --- 7. persist table + summary ------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / f"linkage_{args.condition}.json"
    with table_path.open("w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)

    # --- 8. scatter plot ------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ncol = 2 if not args.no_cider else 1
        fig, axes = plt.subplots(1, ncol, figsize=(6 * ncol, 5), squeeze=False)
        col = 0
        if not args.no_cider:
            axes[0][col].scatter(a, c, s=6, alpha=0.3)
            axes[0][col].set_xlabel("pairwise alignment  cos(z_img, z_txt)")
            axes[0][col].set_ylabel("per-image CIDEr")
            r = summary["corr_alignment_vs_cider"]
            axes[0][col].set_title(
                f"{args.condition}: alignment vs CIDEr\n"
                f"Pearson={r['pearson']:.3f}  Spearman={r['spearman']:.3f}")
            col += 1
        axes[0][col].scatter(a, d1, s=6, alpha=0.3)
        axes[0][col].set_xlabel("pairwise alignment  cos(z_img, z_txt)")
        axes[0][col].set_ylabel("distinct-1 (1.0 = no repetition)")
        rd = summary["corr_alignment_vs_distinct1"]
        axes[0][col].set_title(
            f"{args.condition}: alignment vs diversity\n"
            f"Pearson={rd['pearson']:.3f}  Spearman={rd['spearman']:.3f}")

        fig.tight_layout()
        fig_dir = Path(args.fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / f"linkage_{args.condition}.png"
        fig.savefig(fig_path, dpi=150)
        print(f"[ok] wrote {fig_path}")
    except Exception as e:  # noqa: BLE001 — plotting is best-effort
        print(f"[warn] plot skipped: {type(e).__name__}: {e}")

    # --- 9. report -----------------------------------------------------------
    print(f"[ok] wrote {table_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
