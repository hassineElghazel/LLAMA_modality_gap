"""Known-answer tests for the diagnostic metrics (§7.4 of the plan)."""
from __future__ import annotations

import numpy as np
import pytest

from src.diagnostics.metrics import (
    anisotropy_ratio,
    centroid_gap,
    compute_all_metrics,
    covariance_shape_discrepancy,
    effective_dimension,
    knn_mixing_rate,
    residual_ratio,
    spectral_correlation,
    subspace_overlap,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ---------- identical distributions: gap should vanish ----------

def test_identical_distributions_have_zero_gap(rng):
    X = rng.standard_normal((2000, 32))
    metrics = compute_all_metrics(X, X.copy())
    assert metrics.G_mu < 1e-10
    assert metrics.G_Sigma < 1e-10
    assert metrics.C_lambda == pytest.approx(1.0, abs=1e-6)
    assert metrics.residual_ratio < 1e-10  # both numerator and denom are 0
    assert metrics.knn_mixing_rate_k20 == pytest.approx(0.5, abs=0.05)


# ---------- centroid offset only ----------

def test_pure_centroid_gap(rng):
    X = rng.standard_normal((1000, 16))
    Y = X + np.array([3.0, 0.0, 0.0] + [0.0] * 13)
    g = centroid_gap(X, Y)
    assert g == pytest.approx(3.0, rel=1e-6)
    # Residual is exactly zero -> residual_ratio == 0.
    assert residual_ratio(X, Y) == pytest.approx(0.0, abs=1e-10)


# ---------- isotropic Gaussian residual: A_r ~ 1, d_eff ~ d ----------

def test_isotropic_residual(rng):
    n, d = 5000, 64
    X = rng.standard_normal((n, d))
    # Y = X plus independent isotropic noise -> residual is isotropic.
    Y = X + rng.standard_normal((n, d))
    assert anisotropy_ratio(X, Y) == pytest.approx(1.0, rel=0.15)
    assert effective_dimension(X, Y) == pytest.approx(d, rel=0.15)


# ---------- anisotropic residual: A_r large, d_eff small ----------

def test_anisotropic_residual(rng):
    n, d = 5000, 64
    X = rng.standard_normal((n, d))
    direction = np.zeros(d); direction[0] = 1.0
    # Y = X plus residual concentrated in a single direction.
    noise = rng.standard_normal(n)[:, None] * direction[None, :] * 5.0
    Y = X + noise
    assert anisotropy_ratio(X, Y) > 10.0
    assert effective_dimension(X, Y) < 5.0


# ---------- subspace overlap: random subspaces give ~q/d ----------

def test_subspace_overlap_random_baseline(rng):
    n, d = 4000, 128
    # Two unrelated Gaussians -> top-q principal subspaces are random in d dims.
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal((n, d))
    for q in (1, 4, 16, 64):
        baseline = q / d
        observed = subspace_overlap(X, Y, q)
        # Tolerant — finite-sample variance is real.
        assert abs(observed - baseline) < 0.2, f"q={q} obs={observed} baseline={baseline}"


def test_subspace_overlap_identical_subspaces(rng):
    n, d, q = 1000, 32, 8
    # Construct X, Y sharing top-q principal directions exactly.
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eig = np.zeros(d); eig[:q] = 10.0; eig[q:] = 0.01
    Sigma = Q @ np.diag(eig) @ Q.T
    L = np.linalg.cholesky(Sigma + 1e-9 * np.eye(d))
    X = rng.standard_normal((n, d)) @ L.T
    Y = rng.standard_normal((n, d)) @ L.T
    # Both share the same dominant subspace -> O_q ~ 1.
    assert subspace_overlap(X, Y, q) == pytest.approx(1.0, abs=0.15)


# ---------- spectral correlation ----------

def test_spectral_correlation_matched(rng):
    n, d = 2000, 32
    L = np.diag(np.linspace(0.1, 10.0, d))
    X = rng.standard_normal((n, d)) @ L
    Y = rng.standard_normal((n, d)) @ L
    assert spectral_correlation(X, Y) > 0.95


# ---------- knn mixing ----------

def test_knn_mixing_disjoint_clusters_low(rng):
    n, d = 500, 8
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal((n, d)) + 100.0    # far apart
    assert knn_mixing_rate(X, Y, k=10) < 0.05


def test_knn_mixing_overlapping_high(rng):
    n, d = 500, 8
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal((n, d))
    assert knn_mixing_rate(X, Y, k=10) == pytest.approx(0.5, abs=0.1)


# ---------- ambient-dim parameterization ----------

def test_metrics_use_ambient_dim_from_data(rng):
    """Same metric implementation must work for d=768 and d=4096 — the
    encoder-space and projected-token-space measurement points (§2)."""
    for d in (768, 4096):
        n = 200
        X = rng.standard_normal((n, d))
        Y = rng.standard_normal((n, d))
        metrics = compute_all_metrics(X, Y)
        assert metrics.d == d
        # Subspace ladder must be capped at d (e.g. q=512 must be excluded for d=128).
        assert all(int(q) <= d for q in metrics.subspace_overlap_q)


# ---------- Float64 enforcement ----------

def test_metrics_run_in_float64(rng):
    """Pass Float32 input — internals must up-cast to Float64."""
    n, d = 500, 16
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = X + rng.standard_normal((n, d)).astype(np.float32) * 0.01
    g = centroid_gap(X, Y)
    # Just ensures no crash and value is finite.
    assert np.isfinite(g)


# ---------- covariance shape discrepancy ----------

def test_covariance_shape_discrepancy_zero_for_same_cov(rng):
    n, d = 2000, 16
    L = np.diag(np.linspace(0.5, 5.0, d))
    X = rng.standard_normal((n, d)) @ L
    Y = rng.standard_normal((n, d)) @ L
    # Sample cov differs slightly across draws but should be small relative.
    assert covariance_shape_discrepancy(X, Y) < 0.2
