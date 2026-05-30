"""Per-image linkage: does connector geometry predict caption quality?

Condition-level (n=4) comparison is statistically inert, and the population gap
metrics (G_mu, subspace_overlap, alpha, eff_rank, JS) are single scalars per
condition with no per-image value. So we work per image (n up to 5000 within a
condition) and decompose the geometry into per-image features:

    align_raw            : cos(z_img, z_txt)                       (gap-confounded)
    align_centered       : cos(z_img-mu_img, z_txt-mu_txt)         (pairwise alignment)
    subspace_energy      : energy fraction of (z_img-mu_img) inside the TEXT cloud's
                           top-q principal subspace  (per-image analogue of the
                           population subspace_overlap -- the Stage-1 effect)
    dist_to_text_centroid: ||z_img - mu_txt||         (per-image analogue of G_mu)
    img_norm             : ||z_img||                  (per-image scale / anisotropy)

These are correlated against behavioural targets, and a multivariate OLS reports
the R^2 of the whole geometric feature set jointly predicting grounding.

Behavioural targets (measurement lessons from earlier passes baked in):
    clipscore   : reference-free grounding cos(CLIP_img, CLIP_txt(caption))
                  -- immune to the verbose-vs-terse style mismatch that floors CIDEr.
    comp_ratio  : gzip(caption)/len  -- LOW = repetitive; catches char-level loops
                  that whitespace distinct-1 misses ("stestest..." -> 1 token).
    distinct_c4 : unique char-4-gram ratio.
    cider       : per-image CIDEr (kept for documentation; floored for verbose text).

Join is by image_id (embeddings are in seed-42 manifest order, captions are in
sorted order; row i of the embedding tensor -> manifest[i].image_id).

Usage:
    python scripts/11_link_gap_to_captions.py --condition C3_stage2 --clipscore
    python scripts/11_link_gap_to_captions.py --condition C2_stage1 --no-cider --clipscore --subspace-q 64
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
    out = {"pearson": _pearson(x, y), "spearman": _spearman(x, y), "n": int(len(x))}
    try:
        from scipy.stats import pearsonr, spearmanr
        if x.std() > 0 and y.std() > 0:
            out["pearson_p"] = float(pearsonr(x, y)[1])
            out["spearman_p"] = float(spearmanr(x, y)[1])
    except Exception:  # noqa: BLE001 — scipy optional
        pass
    return out


def _ols_r2(features: dict[str, np.ndarray], names: list[str], y: np.ndarray) -> dict:
    """Multivariate OLS: standardized features -> y. Returns R^2 and standardized
    coefficients (comparable across features since each predictor is z-scored)."""
    cols = []
    for n in names:
        x = features[n].astype(np.float64)
        s = x.std()
        cols.append((x - x.mean()) / (s if s > 0 else 1.0))
    X = np.column_stack(cols + [np.ones(len(y))])   # + intercept
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "r2": r2,
        "coefficients_std": {n: float(b) for n, b in zip(names, beta[:-1])},
        "n": int(len(y)),
    }


# ---------------------------------------------------------------------------
# per-image geometry (decomposing the population gap metrics)
# ---------------------------------------------------------------------------
def _cos_rows(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    an = a / a.norm(dim=1, keepdim=True).clamp(min=1e-12)
    bn = b / b.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return (an * bn).sum(dim=1).numpy()


def _cos_to_vec(a: torch.Tensor, v: torch.Tensor) -> np.ndarray:
    an = a / a.norm(dim=1, keepdim=True).clamp(min=1e-12)
    vn = v / v.norm().clamp(min=1e-12)
    return (an @ vn).numpy()


def _per_image_geometry(img: torch.Tensor, txt: torch.Tensor,
                        q_primary: int, q_sweep: list[int]) -> tuple[dict, dict]:
    """Returns (features, subspace_energy_by_q).

    features:
        align_raw            : cos(z_img, z_txt)
        align_centered       : cos(z_img-mu_img, z_txt-mu_txt)
        subspace_energy      : energy fraction of (z_img-mu_img) in the text cloud's
                               top-q_primary principal subspace
        cos_to_text_centroid : cos(z_img, mu_txt)  -- per-image G_mu-direction analogue,
                               not dominated by norm (replaces the old norm-collinear feature)
    subspace_energy_by_q:  {q: energy-fraction array} for the q-sweep diagnostic.
    """
    mu_img = img.mean(dim=0, keepdim=True)
    mu_txt = txt.mean(dim=0, keepdim=True)
    img_c = img - mu_img
    txt_c = txt - mu_txt

    # one SVD of the centered text cloud; energy at any q is the first-q coeffs.
    _, _, Vh = torch.linalg.svd(txt_c, full_matrices=False)   # Vh: (D, D)
    proj_full = img_c @ Vh.T                                   # (N, D) coeffs on text PCs
    img_c_energy = (img_c ** 2).sum(dim=1).clamp(min=1e-12)

    def se(q: int) -> np.ndarray:
        q = min(q, proj_full.shape[1])
        return ((proj_full[:, :q] ** 2).sum(dim=1) / img_c_energy).numpy()

    features = {
        "align_raw": _cos_rows(img, txt),
        "align_centered": _cos_rows(img_c, txt_c),
        "subspace_energy": se(q_primary),
        "cos_to_text_centroid": _cos_to_vec(img, mu_txt.squeeze(0)),
    }
    sweep = {int(q): se(int(q)) for q in q_sweep}
    return features, sweep


# ---------------------------------------------------------------------------
# behaviour signals
# ---------------------------------------------------------------------------
def _degeneracy(caption: str) -> dict:
    toks = caption.split()
    n = len(toks)
    b = caption.encode("utf-8")
    distinct1 = (len(set(toks)) / n) if n else 0.0
    comp = (len(zlib.compress(b, 9)) / len(b)) if len(b) else 0.0
    grams = [caption[i:i + 4] for i in range(len(caption) - 3)]
    distinct_c4 = (len(set(grams)) / len(grams)) if grams else 0.0
    return {"distinct1": distinct1, "comp_ratio": comp, "distinct_c4": distinct_c4, "length": n}


def _per_image_cider(references: dict[int, list[str]], hyps: dict[int, str]) -> dict[int, float]:
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
def _clipscores(ids, path_by_id, cap_by_id, clip_id, device, batch_size=64) -> dict[int, float]:
    from transformers import CLIPModel, CLIPProcessor
    from tqdm import tqdm

    model = CLIPModel.from_pretrained(clip_id).to(device).eval()
    proc = CLIPProcessor.from_pretrained(clip_id)
    out: dict[int, float] = {}
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
        for b, c in zip(batch, (imf * txf).sum(-1).tolist()):
            out[b] = float(c)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
GEOM_FEATURES = ["align_raw", "align_centered", "subspace_energy", "cos_to_text_centroid"]
# features entered into the multivariate model (raw alignment excluded: it is the
# gap-confounded one whose sign flips under centering). img_norm dropped — it was
# collinear with the old dist_to_text_centroid; replaced by cos_to_text_centroid.
REGRESSION_FEATURES = ["align_centered", "subspace_energy", "cos_to_text_centroid"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True)
    p.add_argument("--emb-dir", default="outputs/embeddings")
    p.add_argument("--data-config", default="configs/data.yaml")
    p.add_argument("--captioning-config", default="configs/captioning.yaml")
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--out-dir", default="outputs/metrics")
    p.add_argument("--fig-dir", default="outputs/figures/linkage")
    p.add_argument("--subspace-q", type=int, default=64,
                   help="rank of the text principal subspace for the subspace_energy feature")
    p.add_argument("--subspace-q-sweep", type=int, nargs="*", default=[16, 32, 64, 128],
                   help="extra q values to report subspace_energy-vs-target correlations for")
    p.add_argument("--no-cider", action="store_true")
    p.add_argument("--clipscore", action="store_true")
    args = p.parse_args()

    data_cfg = load_yaml(args.data_config)
    cap_cfg = load_yaml(args.captioning_config)
    enc_cfg = load_yaml(args.encoders_config)

    # --- 1. row -> image_id (+ path) from the manifest -----------------------
    pairs = load_diagnostic_manifest(data_cfg["diagnostic_sample"]["manifest_path"])
    row_image_ids = [int(pr.image_id) for pr in pairs]
    path_by_id = {int(pr.image_id): pr.image_path for pr in pairs}

    # --- 2. pooled embeddings (row order == manifest order) ------------------
    emb_dir = Path(args.emb_dir)
    img = torch.load(emb_dir / f"projected_{args.condition}_image_pooled.pt", map_location="cpu").to(torch.float64)
    txt = torch.load(emb_dir / f"projected_{args.condition}_text_pooled.pt", map_location="cpu").to(torch.float64)
    if not (img.shape[0] == txt.shape[0] == len(row_image_ids)):
        raise ValueError(f"row mismatch: image={img.shape[0]} text={txt.shape[0]} "
                         f"manifest={len(row_image_ids)}")

    # --- 3. per-image geometry -----------------------------------------------
    geom, sweep = _per_image_geometry(img, txt, q_primary=args.subspace_q,
                                      q_sweep=args.subspace_q_sweep)
    geom_by_id = {feat: {iid: float(v) for iid, v in zip(row_image_ids, vals)}
                  for feat, vals in geom.items()}
    sweep_by_id = {q: {iid: float(v) for iid, v in zip(row_image_ids, vals)}
                   for q, vals in sweep.items()}

    # --- 4. captions ----------------------------------------------------------
    pred_path = Path(cap_cfg["output"]["predictions_dir"]) / f"captions_{args.condition}.json"
    with pred_path.open() as f:
        preds = json.load(f)
    hyps = {int(r["image_id"]): r["caption"] for r in preds}

    # --- 5. behaviour targets -------------------------------------------------
    cider_by_id: dict[int, float] = {}
    if not args.no_cider:
        ds = CocoVal2017Dataset(annotations_json=cap_cfg["eval_set"]["annotations_json"],
                                image_root=cap_cfg["eval_set"]["image_root"])
        cider_by_id = _per_image_cider(ds.references(), hyps)

    clip_by_id: dict[int, float] = {}
    if args.clipscore:
        joined = [iid for iid in hyps if iid in geom_by_id["align_raw"] and iid in path_by_id]
        clip_by_id = _clipscores(joined, path_by_id, hyps,
                                 clip_id=enc_cfg["vision_model"]["hf_id"],
                                 device=enc_cfg["inference"]["device"])

    # --- 6. inner-join on image_id -------------------------------------------
    rows = []
    for iid, cap in hyps.items():
        if iid not in geom_by_id["align_raw"]:
            continue
        deg = _degeneracy(cap)
        row = {"image_id": iid, "cider": cider_by_id.get(iid), "clipscore": clip_by_id.get(iid), **deg}
        for feat in GEOM_FEATURES:
            row[feat] = geom_by_id[feat][iid]
        rows.append(row)
    if not rows:
        raise ValueError("no image_ids shared between embeddings and captions.")

    feats = {f: np.array([r[f] for r in rows], dtype=np.float64) for f in GEOM_FEATURES}
    targets = {
        "comp_ratio": np.array([r["comp_ratio"] for r in rows]),
        "distinct_c4": np.array([r["distinct_c4"] for r in rows]),
    }
    if not args.no_cider:
        targets["cider"] = np.array([r["cider"] for r in rows], dtype=np.float64)
    if args.clipscore:
        targets["clipscore"] = np.array([r["clipscore"] if r["clipscore"] is not None else np.nan
                                         for r in rows])

    # --- 7. univariate correlations (every feature x every target) -----------
    univariate = {f: {} for f in GEOM_FEATURES}
    for f in GEOM_FEATURES:
        for tname, tv in targets.items():
            m = ~np.isnan(tv)
            univariate[f][tname] = _corr(feats[f][m], tv[m])

    # --- 8. multivariate R^2 (geometry feature set -> each target) -----------
    multivariate = {}
    for tname, tv in targets.items():
        m = ~np.isnan(tv)
        fm = {f: feats[f][m] for f in REGRESSION_FEATURES}
        multivariate[tname] = _ols_r2(fm, REGRESSION_FEATURES, tv[m])

    # --- 8b. q-sweep: subspace_energy(q) vs each target ----------------------
    q_sweep = {}
    for q in sorted(sweep_by_id):
        se_q = np.array([sweep_by_id[q][r["image_id"]] for r in rows], dtype=np.float64)
        q_sweep[str(q)] = {}
        for tname, tv in targets.items():
            m = ~np.isnan(tv)
            q_sweep[str(q)][tname] = _corr(se_q[m], tv[m])

    summary = {
        "condition": args.condition,
        "n_joined": len(rows),
        "subspace_q": args.subspace_q,
        "feature_means": {f: float(feats[f].mean()) for f in GEOM_FEATURES},
        "univariate": univariate,
        "multivariate": multivariate,
        "subspace_q_sweep": q_sweep,
    }
    if args.clipscore:
        summary["clipscore_mean"] = float(np.nanmean(targets["clipscore"]))
    if not args.no_cider:
        summary["cider_mean"] = float(targets["cider"].mean())

    # --- 9. persist -----------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / f"linkage_{args.condition}.json"
    with table_path.open("w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)

    # --- 10. scatter: each geometric feature vs the primary grounding target --
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        target_name = "clipscore" if args.clipscore else "comp_ratio"
        tv = targets[target_name]
        m = ~np.isnan(tv)
        plot_feats = ["align_centered", "subspace_energy", "cos_to_text_centroid"]
        fig, axes = plt.subplots(1, len(plot_feats), figsize=(6 * len(plot_feats), 5), squeeze=False)
        for ax, f in zip(axes[0], plot_feats):
            ax.scatter(feats[f][m], tv[m], s=6, alpha=0.3)
            ax.set_xlabel(f)
            ax.set_ylabel(target_name)
            r = univariate[f][target_name]
            ax.set_title(f"{args.condition}: {f} vs {target_name}\n"
                         f"Pearson={r['pearson']:.3f}  Spearman={r['spearman']:.3f}")
        r2 = multivariate[target_name]["r2"]
        fig.suptitle(f"{args.condition}: multivariate geometry -> {target_name}  R^2={r2:.3f}", y=1.02)
        fig.tight_layout()
        fig_dir = Path(args.fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / f"linkage_{args.condition}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"[ok] wrote {fig_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] plot skipped: {type(e).__name__}: {e}")

    print(f"[ok] wrote {table_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
