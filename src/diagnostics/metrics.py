"""Modality-gap diagnostic metrics.

Pure functions over paired embedding tensors (X for modality A, Y for modality B,
both shaped (n, d)). All eigendecompositions run on CPU in Float64 via
``scipy.linalg.eigh`` for exactness — the matrices are at most 4096x4096 so speed
is not the constraint.

All metrics are parameterized by the ambient dimension ``d`` so the same
implementation works at the encoder measurement point (d=768) and the
projected-token measurement point (d=4096). Subspace-overlap baselines (q/d),
effective-dimension ratios (d_eff/d), and isotropic baselines (1/d) all depend
on d — never hardcode 768.

Float64 throughout: ReAlign Appendix E.2 documents that Float32 accumulation
introduces a ~1e-8 error floor that contaminates centroid/mean metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from sklearn.neighbors import NearestNeighbors


# ---------- helpers ----------

def _to_f64(arr) -> np.ndarray:
    """Coerce a tensor-like to a 2-D Float64 numpy array."""
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {a.shape}")
    return a


def _check_paired(x: np.ndarray, y: np.ndarray) -> tuple[int, int]:
    if x.shape != y.shape:
        raise ValueError(f"X and Y must share shape; got {x.shape} vs {y.shape}")
    return x.shape


def _covariance(x: np.ndarray) -> np.ndarray:
    """Sample covariance with 1/n normalization, Float64."""
    mu = x.mean(axis=0, keepdims=True)
    centered = x - mu
    return (centered.T @ centered) / x.shape[0]


def _eigh_desc(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric eigendecomposition, eigenvalues descending."""
    M_sym = 0.5 * (M + M.T)
    w, V = scipy.linalg.eigh(M_sym)
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]


# ---------- centroid / covariance gap ----------

def centroid_gap(X, Y) -> float:
    """G_mu := || mu_x - mu_y ||_2."""
    X = _to_f64(X); Y = _to_f64(Y)
    return float(np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0)))


def covariance_shape_discrepancy(X, Y, eps: float = 1e-12) -> float:
    """G_Sigma := || Sigma_x - Sigma_y ||_F / (|| Sigma_x ||_F + eps)."""
    X = _to_f64(X); Y = _to_f64(Y)
    Sx = _covariance(X); Sy = _covariance(Y)
    return float(np.linalg.norm(Sx - Sy, "fro") / (np.linalg.norm(Sx, "fro") + eps))


def spectral_correlation(X, Y, eps: float = 1e-12) -> float:
    """C_lambda := corr(log eigvals(Sigma_x), log eigvals(Sigma_y)).

    Eigenvalues clipped at ``eps`` before log to keep things finite when a
    near-zero eigenvalue appears.
    """
    X = _to_f64(X); Y = _to_f64(Y)
    wx, _ = _eigh_desc(_covariance(X))
    wy, _ = _eigh_desc(_covariance(Y))
    lx = np.log(np.clip(wx, eps, None))
    ly = np.log(np.clip(wy, eps, None))
    if np.std(lx) < eps or np.std(ly) < eps:
        return 1.0
    return float(np.corrcoef(lx, ly)[0, 1])


def subspace_overlap(X, Y, q: int) -> float:
    """O_q := (1/q) || U_x^q.T U_y^q ||_F^2 for top-q principal subspaces.

    Random q-dim subspace baseline is q/d.
    """
    X = _to_f64(X); Y = _to_f64(Y)
    _, d = _check_paired(X, Y)
    if not 1 <= q <= d:
        raise ValueError(f"q must be in [1, d={d}], got {q}")
    _, Ux = _eigh_desc(_covariance(X))
    _, Uy = _eigh_desc(_covariance(Y))
    M = Ux[:, :q].T @ Uy[:, :q]
    return float(np.linalg.norm(M, "fro") ** 2 / q)


# ---------- residual covariance (centroid-corrected) ----------

def residual_vectors(X, Y) -> np.ndarray:
    """r_i := (x_i - mu_x) - (y_i - mu_y)."""
    X = _to_f64(X); Y = _to_f64(Y)
    _check_paired(X, Y)
    return (X - X.mean(axis=0, keepdims=True)) - (Y - Y.mean(axis=0, keepdims=True))


def residual_covariance(X, Y) -> np.ndarray:
    R = residual_vectors(X, Y)
    return (R.T @ R) / R.shape[0]


def anisotropy_ratio(X, Y) -> float:
    """A_r := lambda_max(Sigma_r) / (tr(Sigma_r) / d). Isotropic baseline = 1."""
    Sr = residual_covariance(X, Y)
    d = Sr.shape[0]
    w, _ = _eigh_desc(Sr)
    tr = float(np.sum(w))
    if tr <= 0:
        return float("nan")
    return float(w[0] / (tr / d))


def effective_dimension(X, Y) -> float:
    """d_eff := tr(Sigma_r)^2 / tr(Sigma_r^2). Isotropic baseline = d."""
    Sr = residual_covariance(X, Y)
    w, _ = _eigh_desc(Sr)
    num = float(np.sum(w)) ** 2
    den = float(np.sum(w * w))
    if den <= 0:
        return float("nan")
    return num / den


def residual_ratio(X, Y) -> float:
    """tr(Sigma_r) / (|| mu_x - mu_y ||^2 + tr(Sigma_r)).

    Fraction of total gap-second-moment NOT explained by the centroid offset.
    Close to 1 means the gap is dominated by per-sample residual structure
    rather than a clean rigid translation.
    """
    X = _to_f64(X); Y = _to_f64(Y)
    Sr = residual_covariance(X, Y)
    tr = float(np.trace(Sr))
    g2 = float(np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0)) ** 2)
    denom = g2 + tr
    if denom <= 0:
        return float("nan")
    return tr / denom


def residual_energy_curve(X, Y) -> np.ndarray:
    """E(K) := cumsum(sorted_eigvals(Sigma_r)) / tr(Sigma_r), descending order.

    Returns an array of length d. Plot E(K) vs K for the residual-energy figure.
    """
    Sr = residual_covariance(X, Y)
    w, _ = _eigh_desc(Sr)
    tr = float(np.sum(w))
    if tr <= 0:
        return np.zeros_like(w)
    return np.cumsum(w) / tr


# ---------- topological / mixing ----------

def knn_mixing_rate(X, Y, k: int = 20) -> float:
    """Fraction of k-NN that come from the OPPOSITE modality, averaged over all
    2n points. Computed in the joint pool (X stacked with Y), excluding self.

    Per ReAlign Appendix D.3. Range [0, 1]; near 0 means the two modalities
    occupy disjoint regions of the embedding space.
    """
    X = _to_f64(X); Y = _to_f64(Y)
    n, _ = _check_paired(X, Y)
    pool = np.vstack([X, Y])
    labels = np.concatenate([np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)])
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(pool)
    _, idx = nn.kneighbors(pool)
    neighbor_idx = idx[:, 1:]   # drop self
    neighbor_labels = labels[neighbor_idx]
    opposite = (neighbor_labels != labels[:, None]).sum()
    return float(opposite) / (2 * n * k)


# ---------- summary container ----------

@dataclass
class GapMetrics:
    """All scalar diagnostic metrics for one paired embedding set."""

    n: int
    d: int
    G_mu: float
    G_Sigma: float
    C_lambda: float
    A_r: float
    d_eff: float
    d_eff_over_d: float
    residual_ratio: float
    knn_mixing_rate_k20: float
    subspace_overlap_q: dict[int, float]
    subspace_overlap_random_baseline_q: dict[int, float]

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "d": self.d,
            "G_mu": self.G_mu,
            "G_Sigma": self.G_Sigma,
            "C_lambda": self.C_lambda,
            "A_r": self.A_r,
            "d_eff": self.d_eff,
            "d_eff_over_d": self.d_eff_over_d,
            "residual_ratio": self.residual_ratio,
            "knn_mixing_rate_k20": self.knn_mixing_rate_k20,
            "subspace_overlap_q": {str(k): v for k, v in self.subspace_overlap_q.items()},
            "subspace_overlap_random_baseline_q": {
                str(k): v for k, v in self.subspace_overlap_random_baseline_q.items()
            },
        }


DEFAULT_Q_LADDER = (1, 4, 16, 64, 128, 256, 512)


def compute_all_metrics(
    X,
    Y,
    q_ladder: tuple[int, ...] = DEFAULT_Q_LADDER,
    knn_k: int = 20,
) -> GapMetrics:
    """Run every metric and pack into a ``GapMetrics`` container.

    Subspace-overlap q values that exceed d are silently skipped (so the same
    default ladder works for d=768 and d=4096).
    """
    X = _to_f64(X); Y = _to_f64(Y)
    n, d = _check_paired(X, Y)
    qs = tuple(q for q in q_ladder if q <= d)
    overlaps = {q: subspace_overlap(X, Y, q) for q in qs}
    baselines = {q: q / d for q in qs}
    deff = effective_dimension(X, Y)
    return GapMetrics(
        n=n,
        d=d,
        G_mu=centroid_gap(X, Y),
        G_Sigma=covariance_shape_discrepancy(X, Y),
        C_lambda=spectral_correlation(X, Y),
        A_r=anisotropy_ratio(X, Y),
        d_eff=deff,
        d_eff_over_d=(deff / d) if not np.isnan(deff) else float("nan"),
        residual_ratio=residual_ratio(X, Y),
        knn_mixing_rate_k20=knn_mixing_rate(X, Y, k=knn_k),
        subspace_overlap_q=overlaps,
        subspace_overlap_random_baseline_q=baselines,
    )
