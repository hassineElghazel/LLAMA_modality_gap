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
        # Both numerator and denominator vanish (X == Y up to centering):
        # there is no gap, hence no residual fraction.
        return 0.0
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


# ---------- Overleaf-spec metrics (per-modality + bias decomposition) ----------

def trace_cov(X) -> float:
    """tr(Sigma_x) — global variance scale (spec §"Trace")."""
    X = _to_f64(X)
    return float(np.trace(_covariance(X)))


def effective_rank(X) -> float:
    """tr(Sigma)^2 / tr(Sigma^2) on a single modality's covariance.

    Per spec §"Effective rank — representation compactness". Distinct from
    ``effective_dimension(X, Y)`` which is evaluated on the residual covariance
    Sigma_r; this version operates on Sigma_x or Sigma_y in isolation.
    """
    X = _to_f64(X)
    S = _covariance(X)
    w, _ = _eigh_desc(S)
    num = float(np.sum(w)) ** 2
    den = float(np.sum(w * w))
    if den <= 0:
        return float("nan")
    return num / den


def cov_condition_number(X, eps: float = 1e-12) -> float:
    """kappa(Sigma_x) := lambda_max(Sigma_x) / max(lambda_min(Sigma_x), eps).

    Per spec §"Residual anisotropy". Large kappa indicates a needle-like
    covariance; isotropic baseline ~ 1.
    """
    X = _to_f64(X)
    w, _ = _eigh_desc(_covariance(X))
    w_max = float(w[0])
    w_min = float(w[-1])
    return w_max / max(w_min, eps)


def power_law_exponent(
    X,
    fit_start: int = 1,
    fit_end: int | None = None,
    eps: float = 1e-12,
) -> float:
    """alpha such that lambda_i ~ i^{-alpha} for sorted eigenvalues of Sigma_x.

    Per spec §"Power-law exponent — semantic hierarchy preservation". Fits a
    line through ``log(rank)`` vs ``log(eigenvalue)`` over the rank range
    [fit_start, fit_end]. By default uses ranks 1..d/2 to avoid the noisy tail.
    Returns the positive exponent ``alpha = -slope``.
    """
    X = _to_f64(X)
    w, _ = _eigh_desc(_covariance(X))
    d = len(w)
    if fit_end is None:
        fit_end = max(2, d // 2)
    fit_start = max(1, fit_start)
    fit_end = min(d, fit_end)
    if fit_end <= fit_start:
        return float("nan")
    ranks = np.arange(fit_start, fit_end + 1, dtype=np.float64)
    eigs = w[fit_start - 1 : fit_end]
    log_ranks = np.log(ranks)
    log_eigs = np.log(np.clip(eigs, eps, None))
    slope, _ = np.polyfit(log_ranks, log_eigs, 1)
    return float(-slope)


def pmb_cob_decomposition(X, Y) -> dict[str, float]:
    """Decompose the paired bias `b_i = x_i - y_i` into:

    - **COB** (Centroid Of Bias): `gamma = mean(b_i) = mu_x - mu_y`. Its norm
      `||gamma||` equals the centroid gap `G_mu`.
    - **PMB** (Per-pair Mean Bias magnitude): `||beta|| = sqrt(mean(||b_i||^2))`,
      the root-mean-square per-pair bias magnitude.

    The two satisfy `||beta||^2 = ||gamma||^2 + (1/n) sum_i ||b_i - gamma||^2`,
    so `||beta||^2 - ||gamma||^2` is the per-pair residual energy that the
    centroid offset does NOT explain.

    Returns ``{"beta_norm": ..., "gamma_norm": ..., "residual_energy": ...}``.
    """
    X = _to_f64(X); Y = _to_f64(Y)
    _check_paired(X, Y)
    B = X - Y                              # (n, d) per-pair bias vectors
    gamma = B.mean(axis=0)
    gamma_norm = float(np.linalg.norm(gamma))
    beta_norm_sq = float(np.mean(np.sum(B * B, axis=1)))
    beta_norm = float(np.sqrt(beta_norm_sq))
    residual_energy = float(beta_norm_sq - gamma_norm * gamma_norm)
    return {
        "beta_norm": beta_norm,
        "gamma_norm": gamma_norm,
        "residual_energy": max(residual_energy, 0.0),   # numerical guard
    }


def js_divergence_angular(
    X,
    Y,
    n_bins: int = 50,
    n_sample_pairs: int = 20000,
    seed: int = 0,
    eps: float = 1e-12,
) -> float:
    """Jensen-Shannon divergence between within-modality pairwise angle
    distributions of X and Y.

    Per spec §"JS divergence — angular topology mismatch". Each modality is
    mean-centered, then ``n_sample_pairs`` random index pairs (i != j) are
    sampled to estimate the distribution of pairwise angles
    ``theta_ij = arccos(cos_sim(x_i, x_j))`` on the support ``[0, pi]``.
    Histograms with ``n_bins`` uniform bins are compared via the
    symmetric JS divergence (in nats; range ``[0, ln 2]``).
    """
    rng = np.random.default_rng(seed)
    X = _to_f64(X); Y = _to_f64(Y)
    n, _ = _check_paired(X, Y)

    def _angle_hist(Z: np.ndarray) -> np.ndarray:
        Z = Z - Z.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(Z, axis=1)
        norms = np.maximum(norms, eps)
        Zn = Z / norms[:, None]
        idx_i = rng.integers(0, n, size=n_sample_pairs)
        idx_j = rng.integers(0, n, size=n_sample_pairs)
        same = idx_i == idx_j
        if same.any():
            idx_j[same] = (idx_j[same] + 1) % n
        dots = np.einsum("nd,nd->n", Zn[idx_i], Zn[idx_j])
        dots = np.clip(dots, -1.0 + eps, 1.0 - eps)
        angles = np.arccos(dots)
        h, _ = np.histogram(angles, bins=n_bins, range=(0.0, np.pi), density=False)
        p = h.astype(np.float64) + eps
        return p / p.sum()

    P = _angle_hist(X)
    Q = _angle_hist(Y)
    M = 0.5 * (P + Q)
    kl_pm = float(np.sum(P * np.log(P / M)))
    kl_qm = float(np.sum(Q * np.log(Q / M)))
    return 0.5 * (kl_pm + kl_qm)


# ---------- summary container ----------

@dataclass
class GapMetrics:
    """All scalar diagnostic metrics for one paired embedding set.

    Spec metrics (Overleaf §"Metrics → Geometric Metrics") are stored as
    top-level fields. Legacy metrics from earlier paper-aligned analyses
    (G_Sigma, C_lambda, A_r, residual_ratio, subspace_overlap) remain on the
    dataclass so existing analyses do not break.
    """

    n: int
    d: int
    # ---- Overleaf spec metrics ----
    G_mu: float                    # centroid distance (= ||gamma||)
    alpha_image: float             # power-law exponent of Sigma_x
    alpha_text: float              # power-law exponent of Sigma_y
    js_divergence_angular: float
    knn_mixing_rate_k20: float
    beta_norm: float               # PMB magnitude
    gamma_norm: float              # COB magnitude (= G_mu)
    pair_residual_energy: float
    kappa_image: float             # kappa(Sigma_x)
    kappa_text: float              # kappa(Sigma_y)
    eff_rank_image: float
    eff_rank_text: float
    trace_image: float
    trace_text: float
    # ---- Legacy / supplementary ----
    G_Sigma: float
    C_lambda: float
    A_r: float
    d_eff: float                   # effective dim of Sigma_r (residual)
    d_eff_over_d: float
    residual_ratio: float
    subspace_overlap_q: dict[int, float]
    subspace_overlap_random_baseline_q: dict[int, float]

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "d": self.d,
            "spec_metrics": {
                "G_mu": self.G_mu,
                "alpha_image": self.alpha_image,
                "alpha_text": self.alpha_text,
                "js_divergence_angular": self.js_divergence_angular,
                "knn_mixing_rate_k20": self.knn_mixing_rate_k20,
                "beta_norm": self.beta_norm,
                "gamma_norm": self.gamma_norm,
                "pair_residual_energy": self.pair_residual_energy,
                "kappa_image": self.kappa_image,
                "kappa_text": self.kappa_text,
                "eff_rank_image": self.eff_rank_image,
                "eff_rank_text": self.eff_rank_text,
                "trace_image": self.trace_image,
                "trace_text": self.trace_text,
            },
            "extras": {
                "G_Sigma": self.G_Sigma,
                "C_lambda": self.C_lambda,
                "A_r": self.A_r,
                "d_eff": self.d_eff,
                "d_eff_over_d": self.d_eff_over_d,
                "residual_ratio": self.residual_ratio,
                "subspace_overlap_q": {str(k): v for k, v in self.subspace_overlap_q.items()},
                "subspace_overlap_random_baseline_q": {
                    str(k): v for k, v in self.subspace_overlap_random_baseline_q.items()
                },
            },
        }


DEFAULT_Q_LADDER = (1, 4, 16, 64, 128, 256, 512)


def compute_all_metrics(
    X,
    Y,
    q_ladder: tuple[int, ...] = DEFAULT_Q_LADDER,
    knn_k: int = 20,
    js_n_bins: int = 50,
    js_n_sample_pairs: int = 20000,
    js_seed: int = 0,
    power_law_fit_end: int | None = None,
) -> GapMetrics:
    """Run every spec metric (+ legacy extras) and pack into ``GapMetrics``.

    Subspace-overlap q values that exceed d are silently skipped, so the same
    ladder works at any ambient dimension.
    """
    X = _to_f64(X); Y = _to_f64(Y)
    n, d = _check_paired(X, Y)
    qs = tuple(q for q in q_ladder if q <= d)
    overlaps = {q: subspace_overlap(X, Y, q) for q in qs}
    baselines = {q: q / d for q in qs}
    deff = effective_dimension(X, Y)
    pmb = pmb_cob_decomposition(X, Y)
    return GapMetrics(
        n=n,
        d=d,
        # spec metrics
        G_mu=centroid_gap(X, Y),
        alpha_image=power_law_exponent(X, fit_end=power_law_fit_end),
        alpha_text=power_law_exponent(Y, fit_end=power_law_fit_end),
        js_divergence_angular=js_divergence_angular(
            X, Y, n_bins=js_n_bins, n_sample_pairs=js_n_sample_pairs, seed=js_seed
        ),
        knn_mixing_rate_k20=knn_mixing_rate(X, Y, k=knn_k),
        beta_norm=pmb["beta_norm"],
        gamma_norm=pmb["gamma_norm"],
        pair_residual_energy=pmb["residual_energy"],
        kappa_image=cov_condition_number(X),
        kappa_text=cov_condition_number(Y),
        eff_rank_image=effective_rank(X),
        eff_rank_text=effective_rank(Y),
        trace_image=trace_cov(X),
        trace_text=trace_cov(Y),
        # legacy
        G_Sigma=covariance_shape_discrepancy(X, Y),
        C_lambda=spectral_correlation(X, Y),
        A_r=anisotropy_ratio(X, Y),
        d_eff=deff,
        d_eff_over_d=(deff / d) if not np.isnan(deff) else float("nan"),
        residual_ratio=residual_ratio(X, Y),
        subspace_overlap_q=overlaps,
        subspace_overlap_random_baseline_q=baselines,
    )
