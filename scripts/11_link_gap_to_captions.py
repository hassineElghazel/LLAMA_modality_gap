"""Per-image linkage: does connector geometry predict caption quality?

The condition-level comparison (C0/C1/C2/C3) only gives 4 data points and is
statistically inert — and worse, C2 and C3 are geometrically near-identical yet
behave oppositely, so no condition-level metric predicts captioning.

This script drops to the per-image level (n = up to 5000 within one condition)
and tests the hypothesis the literature points at (Decipher, 2025): downstream
quality is driven by *pairwise alignment* of each image with its own caption,
not by the global gap size.

Three measurement lessons from the first pass are baked in here:

1. PREDICTOR — raw cos(z_img, z_txt) ~= 0.06 everywhere because both modalities
   sit in disjoint cones (the modality gap); the per-image signal is swamped by
   that global offset. We therefore also report a MEAN-CENTERED alignment
   cos(z_img - mu_img, z_txt - mu_txt), which removes the cone offset and is the
   "pairwise alignment" the theory actually refers to. Centered is the headline.

2. GROUNDING TARGET — CIDEr is floored by the verbose-vs-terse mismatch between
   LLaVA-style captions and COCO's short references (near-zero variance, nothing
   to predict). ``--clipscore`` adds a reference-free grounding target:
   cos(CLIP_image, CLIP_text(caption)) in CLIP's shared space, which measures
   whether the caption matches the IMAGE regardless of length/style.

3. DEGENERACY TARGET — whitespace distinct-1 is fooled by no-space gibberish
   ("stestestest..." -> one token -> scored as maximally diverse). We add a
   gzip compression ratio and a character-4-gram distinctness, both of which
   catch character-level loops.

Join is by ``image_id``, never by row position: embeddings are stored in the
seed-42 diagnostic-manifest order; captions are written in sorted-image_id
order. Row i of the embedding tensor maps to ``manifest[i].image_id``.

Usage:
    python scripts/11_link_gap_to_captions.py --condition C3_stage2 --clipscore
    python scripts/11_link_gap_to_captions.py --condition C2_stage1 --no-cider --clipscore
"""
from __future__ import annotations

import argparse
import json
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.data.coco_val2017_loader import CocoVal2017Dataset, load_diagnostic_manifest, load_image
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
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _corr(x: np.ndarray, y: np.ndarray) -> dict:
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
def _degeneracy(caption: str) -> dict:
    """Word- and char-level repetition signals.

    distinct1   : unique-unigram ratio (UNRELIABLE for no-space loops; kept for
                  continuity / comparison).
    comp_ratio  : gzip(caption) / len(caption); LOW = highly repetitive. Robust
                  to both word loops and no-space character loops.
    distinct_c4 : unique char-4-gram ratio; LOW = repetitive at char level.
    """
    toks = caption.split()
    n = len(toks)
    b = caption.encode("utf-8")
    distinct1 = (len(set(toks)) / n) if n else 0.0
    top1 = (Counter(toks).most_common(1)[0][1] / n) if n else 1.0
    comp = (len(zlib.compress(b, 9)) / len(b)) if len(b) else 0.0
    grams = [caption[i:i + 4] for i in range(len(caption) - 3)]
    distinct_c4 = (len(set(grams)) / len(grams)) if grams else 0.0
    return {"distinct1": distinct1, "top1_frac": top1,
            "comp_ratio": comp, "distinct_c4": distinct_c4, "length": n}


def _per_image_cider(references: dict[int, list[str]], hyps: dict[int, str]) -> dict[int, float]:
    """Per-image CIDEr. Keeps the per-image array ``score_predictions`` discards."""
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    ids = [iid for iid in hyps if iid in references]
    refs = {iid: [{"caption": c} for c in references[iid]] for iid in ids}
    hyp = {iid: [{"caption": hyps[iid]}] for iid in ids}
    tok = PTBTokenizer()
    refs_tok, hyps_tok = tok.tokenize(refs), tok.tokenize(hyp)
    _, per_image = Cider().compute_score(refs_tok, hyps_tok)
    return {int(k): float(v) for k, v in zip(list(hyps_tok.keys()), per_image)}


@torch.no_grad()
def _clipscores(ids: list[int], path_by_id: dict[int, object], cap_by_id: dict[int, str],
                clip_id: str, device: str, batch_size: int = 64) -> dict[int, float]:
    """Reference-free grounding: cos(CLIP_image, CLIP_text(caption)) in CLIP's
    shared space. Loads the FULL CLIPModel (the project's encoder is vision-only)."""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(clip_id).to(device).eval()
    proc = CLIPProcessor.from_pretrained(clip_id)
    out: dict[int, float] = {}
    from tqdm import tqdm
    for i in tqdm(range(0, len(ids), batch_size), desc="clipscore"):
        batch = ids[i:i + batch_size]
        imgs = [load_image(path_by_id[b]) for b in batch]
        caps = [(cap_by_id[b].strip() or " ") for b in batch]
        inp = proc(images=imgs, text=caps, return_tensors="pt",
                   padding=True, truncation=True, max_length=77).to(device)
        imf = model.get_image_features(pixel_values=inp["pixel_values"])
        txf = model.get_text_features(input_ids=inp["input_ids"],
                                      attention_mask=inp["attention_mask"])
        imf = imf / imf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        txf = txf / txf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        cos = (imf * txf).sum(-1)
        for b, c in zip(batch, cos.tolist()):
            out[b] = float(c)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _cos_rows(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    an = a / a.norm(dim=1, keepdim=True).clamp(min=1e-12)
    bn = b / b.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return (an * bn).sum(dim=1).numpy()


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True,
                   help="tag shared by captions_<cond>.json and projected_<cond>_*.pt")
    p.add_argument("--emb-dir", default="outputs/embeddings")
    p.add_argument("--data-config", default="configs/data.yaml")
    p.add_argument("--captioning-config", default="configs/captioning.yaml")
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--out-dir", default="outputs/metrics")
    p.add_argument("--fig-dir", default="outputs/figures/linkage")
    p.add_argument("--no-cider", action="store_true",
                   help="skip per-image CIDEr (use for C0/C2 noise; degeneracy is informative there)")
    p.add_argument("--clipscore", action="store_true",
                   help="compute reference-free CLIPScore grounding target (loads full CLIP, re-reads images)")
    args = p.parse_args()

    data_cfg = load_yaml(args.data_config)
    cap_cfg = load_yaml(args.captioning_config)
    enc_cfg = load_yaml(args.encoders_config)

    # --- 1. row -> image_id (+ image_path) from the diagnostic manifest ------
    pairs = load_diagnostic_manifest(data_cfg["diagnostic_sample"]["manifest_path"])
    row_image_ids = [int(pr.image_id) for pr in pairs]
    path_by_id = {int(pr.image_id): pr.image_path for pr in pairs}

    # --- 2. pooled embeddings (row order == manifest order) ------------------
    emb_dir = Path(args.emb_dir)
    img = torch.load(emb_dir / f"projected_{args.condition}_image_pooled.pt", map_location="cpu").to(torch.float64)
    txt = torch.load(emb_dir / f"projected_{args.condition}_text_pooled.pt", map_location="cpu").to(torch.float64)
    if not (img.shape[0] == txt.shape[0] == len(row_image_ids)):
        raise ValueError(
            f"row mismatch: image={img.shape[0]} text={txt.shape[0]} "
            f"manifest={len(row_image_ids)} — cannot trust the row->image_id map.")

    # --- 3. alignment: raw AND mean-centered ---------------------------------
    align_raw = _cos_rows(img, txt)
    img_c = img - img.mean(dim=0, keepdim=True)
    txt_c = txt - txt.mean(dim=0, keepdim=True)
    align_cen = _cos_rows(img_c, txt_c)
    raw_by_id = {iid: float(v) for iid, v in zip(row_image_ids, align_raw)}
    cen_by_id = {iid: float(v) for iid, v in zip(row_image_ids, align_cen)}

    # --- 4. generated captions -----------------------------------------------
    pred_path = Path(cap_cfg["output"]["predictions_dir"]) / f"captions_{args.condition}.json"
    with pred_path.open() as f:
        preds = json.load(f)
    hyps = {int(r["image_id"]): r["caption"] for r in preds}

    # --- 5. behaviour targets -------------------------------------------------
    cider_by_id: dict[int, float] = {}
    if not args.no_cider:
        ds = CocoVal2017Dataset(
            annotations_json=cap_cfg["eval_set"]["annotations_json"],
            image_root=cap_cfg["eval_set"]["image_root"],
        )
        cider_by_id = _per_image_cider(ds.references(), hyps)

    clip_by_id: dict[int, float] = {}
    if args.clipscore:
        joined = [iid for iid in hyps if iid in raw_by_id and iid in path_by_id]
        clip_by_id = _clipscores(
            joined, path_by_id, hyps,
            clip_id=enc_cfg["vision_model"]["hf_id"],
            device=enc_cfg["inference"]["device"],
        )

    # --- 6. inner-join on image_id -------------------------------------------
    rows = []
    for iid, cap in hyps.items():
        if iid not in raw_by_id:
            continue
        deg = _degeneracy(cap)
        rows.append({
            "image_id": iid,
            "alignment_raw": raw_by_id[iid],
            "alignment_centered": cen_by_id[iid],
            "cider": cider_by_id.get(iid),
            "clipscore": clip_by_id.get(iid),
            **deg,
        })
    if not rows:
        raise ValueError("no image_ids shared between embeddings and captions.")

    cen = np.array([r["alignment_centered"] for r in rows])
    raw = np.array([r["alignment_raw"] for r in rows])

    # correlate both alignment variants against every available target
    targets = {
        "comp_ratio": np.array([r["comp_ratio"] for r in rows]),
        "distinct_c4": np.array([r["distinct_c4"] for r in rows]),
        "distinct1": np.array([r["distinct1"] for r in rows]),
    }
    if not args.no_cider:
        targets["cider"] = np.array([r["cider"] for r in rows], dtype=np.float64)
    if args.clipscore:
        cs = np.array([r["clipscore"] if r["clipscore"] is not None else np.nan for r in rows])
        targets["clipscore"] = cs

    correlations = {}
    for tname, tvals in targets.items():
        m = ~np.isnan(tvals)
        correlations[f"centered_vs_{tname}"] = _corr(cen[m], tvals[m])
        correlations[f"raw_vs_{tname}"] = _corr(raw[m], tvals[m])

    summary = {
        "condition": args.condition,
        "n_joined": len(rows),
        "alignment_raw_mean": float(raw.mean()),
        "alignment_centered_mean": float(cen.mean()),
        "alignment_centered_std": float(cen.std()),
        "correlations": correlations,
    }
    if args.clipscore and "clipscore" in targets:
        summary["clipscore_mean"] = float(np.nanmean(targets["clipscore"]))
    if not args.no_cider:
        summary["cider_mean"] = float(targets["cider"].mean())

    # --- 7. persist -----------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / f"linkage_{args.condition}.json"
    with table_path.open("w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)

    # --- 8. scatter plots (centered alignment as the x-axis) -----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        panels = []
        if args.clipscore and "clipscore" in targets:
            panels.append(("clipscore", "CLIPScore  cos(CLIP_img, CLIP_txt)"))
        if not args.no_cider:
            panels.append(("cider", "per-image CIDEr"))
        panels.append(("comp_ratio", "gzip ratio (low = repetitive)"))

        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
        for ax, (tname, ylabel) in zip(axes[0], panels):
            tv = targets[tname]
            m = ~np.isnan(tv)
            ax.scatter(cen[m], tv[m], s=6, alpha=0.3)
            ax.set_xlabel("mean-centered alignment  cos(z_img-mu, z_txt-mu)")
            ax.set_ylabel(ylabel)
            r = correlations[f"centered_vs_{tname}"]
            ax.set_title(f"{args.condition}: centered align vs {tname}\n"
                         f"Pearson={r['pearson']:.3f}  Spearman={r['spearman']:.3f}")
        fig.tight_layout()
        fig_dir = Path(args.fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / f"linkage_{args.condition}.png"
        fig.savefig(fig_path, dpi=150)
        print(f"[ok] wrote {fig_path}")
    except Exception as e:  # noqa: BLE001 — plotting is best-effort
        print(f"[warn] plot skipped: {type(e).__name__}: {e}")

    print(f"[ok] wrote {table_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
