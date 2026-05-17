"""Diagnostic figures matching the paper style.

Four figure types per measurement point (see §7.6 of the plan):
  A: Compatible dominant geometry (eigenvalue spectra + subspace overlap)
  B: Anisotropic residual (centroid removal bar + residual spectrum + E(K))
  C: PCA cluster scatter
  D: Angular topology (KDE of pairwise cosines)

Locked color palette per the plan; serif font, top/right spines off, 300 DPI.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from .metrics import (
    _covariance,
    _eigh_desc,
    _to_f64,
    centroid_gap,
    knn_mixing_rate,
    residual_covariance,
    residual_energy_curve,
    spectral_correlation,
    subspace_overlap,
)


COLORS = {
    "image": "#3B8BD4",
    "text": "#D4537E",
    "before": "#993C1D",
    "after": "#185FA5",
    "baseline": "#888780",
    "observed": "#26215C",
}


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
    })


def _save(fig: Figure, out_dir: Path, name: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


# ---------- Figure A ----------

def figure_A_dominant_geometry(X, Y, out_dir: Path, q_ladder=(1, 4, 16, 64, 128, 256, 512)) -> tuple[Path, Path]:
    _style()
    X = _to_f64(X); Y = _to_f64(Y)
    d = X.shape[1]
    wx, _ = _eigh_desc(_covariance(X))
    wy, _ = _eigh_desc(_covariance(Y))
    cl = spectral_correlation(X, Y)
    qs = [q for q in q_ladder if q <= d]
    overlaps = [subspace_overlap(X, Y, q) for q in qs]
    baselines = [q / d for q in qs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.loglog(np.arange(1, d + 1), np.clip(wx, 1e-12, None), color=COLORS["image"], label="image")
    ax1.loglog(np.arange(1, d + 1), np.clip(wy, 1e-12, None), color=COLORS["text"], label="text")
    ax1.set_xlabel("eigenvalue index")
    ax1.set_ylabel("eigenvalue")
    ax1.set_title(f"Eigenspectra  (C_lambda = {cl:.3f})")
    ax1.legend()

    ax2.plot(qs, overlaps, "o-", color=COLORS["observed"], label="observed O_q")
    ax2.plot(qs, baselines, "--", color=COLORS["baseline"], label="random baseline q/d")
    ax2.set_xscale("log")
    ax2.set_xlabel("q (subspace size)")
    ax2.set_ylabel("subspace overlap")
    ax2.set_title("Subspace overlap")
    ax2.legend()

    return _save(fig, out_dir, "figureA_dominant_geometry")


# ---------- Figure B ----------

def figure_B_anisotropic_residual(X, Y, out_dir: Path) -> tuple[Path, Path]:
    _style()
    X = _to_f64(X); Y = _to_f64(Y)
    d = X.shape[1]
    g2 = centroid_gap(X, Y) ** 2
    Sr = residual_covariance(X, Y)
    w, _ = _eigh_desc(Sr)
    tr = float(np.sum(w))
    Ar = float(w[0] / (tr / d)) if tr > 0 else float("nan")
    deff = float((tr ** 2) / float(np.sum(w * w))) if np.sum(w * w) > 0 else float("nan")
    rratio = tr / (g2 + tr) if (g2 + tr) > 0 else float("nan")
    Ek = residual_energy_curve(X, Y)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(["centroid (G_mu^2)", "residual (tr Sigma_r)"], [g2, tr],
                color=[COLORS["before"], COLORS["after"]])
    axes[0].set_title(f"Gap decomposition  (residual ratio = {rratio:.3f})")

    axes[1].loglog(np.arange(1, d + 1), np.clip(w, 1e-12, None), color=COLORS["after"], label="residual eigvals")
    axes[1].axhline(tr / d, color=COLORS["baseline"], linestyle="--", label=f"isotropic baseline 1/d (tr/d = {tr/d:.2e})")
    axes[1].set_xlabel("eigenvalue index")
    axes[1].set_ylabel("eigenvalue")
    axes[1].set_title("Residual spectrum")
    axes[1].legend()

    axes[2].plot(np.arange(1, d + 1), Ek, color=COLORS["observed"])
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("cumulative residual energy E(K)")
    axes[2].set_title(f"Residual energy  (A_r = {Ar:.2f}, d_eff = {deff:.1f})")

    return _save(fig, out_dir, "figureB_anisotropic_residual")


# ---------- Figure C ----------

def figure_C_pca_scatter(X, Y, out_dir: Path) -> tuple[Path, Path]:
    _style()
    X = _to_f64(X); Y = _to_f64(Y)
    pca = PCA(n_components=2, random_state=0)
    joint = np.vstack([X, Y])
    proj = pca.fit_transform(joint)
    n = X.shape[0]
    Xp, Yp = proj[:n], proj[n:]
    mix = knn_mixing_rate(X, Y, k=20)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(Xp[:, 0], Xp[:, 1], s=6, alpha=0.5, c=COLORS["image"], label="image")
    ax.scatter(Yp[:, 0], Yp[:, 1], s=6, alpha=0.5, c=COLORS["text"], label="text")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA projection of joint embedding pool")
    ax.text(0.98, 0.02, f"k-NN mixing (k=20) = {mix:.4f}",
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    ax.legend()

    return _save(fig, out_dir, "figureC_pca_scatter")


# ---------- Figure D ----------

def _pairwise_cos(A: np.ndarray, B: np.ndarray | None = None, max_pairs: int = 200_000, seed: int = 0):
    """Return a sample of pairwise cosines. If B is None, samples within A."""
    rng = np.random.default_rng(seed)
    A_n = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    if B is None:
        n = A.shape[0]
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        keep = i != j
        return (A_n[i[keep]] * A_n[j[keep]]).sum(axis=1)
    B_n = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
    n, m = A.shape[0], B.shape[0]
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, m, size=max_pairs)
    return (A_n[i] * B_n[j]).sum(axis=1)


def figure_D_angular_topology(X, Y, out_dir: Path) -> tuple[Path, Path]:
    _style()
    X = _to_f64(X); Y = _to_f64(Y)
    cos_xx = _pairwise_cos(X)
    cos_yy = _pairwise_cos(Y, seed=1)
    cos_xy = _pairwise_cos(X, Y, seed=2)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.kdeplot(cos_xx, ax=ax, color=COLORS["image"], label="image-image")
    sns.kdeplot(cos_yy, ax=ax, color=COLORS["text"], label="text-text")
    sns.kdeplot(cos_xy, ax=ax, color=COLORS["observed"], label="image-text")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title("Pairwise cosine similarity distributions")
    ax.legend()

    return _save(fig, out_dir, "figureD_angular_topology")


# ---------- driver ----------

def make_all_figures(X, Y, out_dir: str | Path) -> dict[str, tuple[Path, Path]]:
    out_dir = Path(out_dir)
    return {
        "A": figure_A_dominant_geometry(X, Y, out_dir),
        "B": figure_B_anisotropic_residual(X, Y, out_dir),
        "C": figure_C_pca_scatter(X, Y, out_dir),
        "D": figure_D_angular_topology(X, Y, out_dir),
    }
