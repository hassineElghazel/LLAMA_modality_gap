"""Known-answer tests for the diagnostic metrics (§7.4 of the plan)."""
from __future__ import annotations

import numpy as np
import pytest

from src.diagnostics.metrics import (
    anisotropy_ratio,
    centroid_gap,
    compute_all_metrics,
    cov_condition_number,
    covariance_shape_discrepancy,
    effective_dimension,
    effective_rank,
    js_divergence_angular,
    knn_mixing_rate,
    pmb_cob_decomposition,
    power_law_exponent,
    residual_ratio,
    spectral_correlation,
    subspace_overlap,
    trace_cov,
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
    # A_r is biased upward by Marchenko-Pastur edge fluctuations:
    # lambda_max / mean(lambda) -> (1+sqrt(d/n))^2 in finite samples, ~1.24
    # at d/n=64/5000. Tolerance reflects that.
    assert anisotropy_ratio(X, Y) == pytest.approx(1.0, rel=0.3)
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


# ---------- Overleaf-spec metrics ----------

def test_trace_equals_sum_of_per_dim_variance(rng):
    n, d = 5000, 8
    sigmas = np.linspace(0.5, 4.0, d)
    X = rng.standard_normal((n, d)) * sigmas
    expected = float(np.sum(sigmas ** 2))
    assert trace_cov(X) == pytest.approx(expected, rel=0.05)


def test_effective_rank_isotropic(rng):
    n, d = 5000, 32
    X = rng.standard_normal((n, d))
    # Isotropic Gaussian -> effective rank ~ d.
    assert effective_rank(X) == pytest.approx(d, rel=0.1)


def test_effective_rank_collapsed():
    # Rank-1 input: all variance in one direction -> effective rank ~ 1.
    n, d = 1000, 16
    e = np.zeros(d); e[0] = 1.0
    z = np.linspace(-3.0, 3.0, n)[:, None]
    X = z * e
    assert effective_rank(X) == pytest.approx(1.0, abs=0.1)


def test_condition_number_isotropic_close_to_one(rng):
    n, d = 4000, 16
    X = rng.standard_normal((n, d))
    assert 0.3 < 1.0 / cov_condition_number(X) <= 1.0
    # Anisotropic case -> kappa large.
    sigmas = np.array([10.0] + [0.1] * (d - 1))
    Y = rng.standard_normal((n, d)) * sigmas
    assert cov_condition_number(Y) > 100.0


def test_power_law_exponent_known_slope():
    # Construct eigenvalues following exact i^{-alpha} decay; covariance has
    # those eigenvalues; fitted exponent must recover alpha.
    n, d = 2000, 64
    alpha_true = 1.5
    target_eigs = (np.arange(1, d + 1, dtype=np.float64)) ** (-alpha_true)
    # Build random orthonormal basis Q and form Sigma = Q diag(target_eigs) Q^T.
    rng = np.random.default_rng(123)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    L = Q @ np.diag(np.sqrt(target_eigs)) @ Q.T
    X = rng.standard_normal((n, d)) @ L.T
    alpha_hat = power_law_exponent(X, fit_start=1, fit_end=d // 2)
    assert alpha_hat == pytest.approx(alpha_true, rel=0.2)


def test_pmb_cob_decomposition_pure_centroid_offset(rng):
    n, d = 1000, 8
    X = rng.standard_normal((n, d))
    offset = np.zeros(d); offset[0] = 4.0
    Y = X - offset           # bias is constant: x_i - y_i = offset
    dec = pmb_cob_decomposition(X, Y)
    assert dec["gamma_norm"] == pytest.approx(4.0, rel=1e-6)
    assert dec["beta_norm"] == pytest.approx(4.0, rel=1e-6)
    assert dec["residual_energy"] == pytest.approx(0.0, abs=1e-8)


def test_pmb_cob_decomposition_zero_centroid_random_pairs(rng):
    n, d = 5000, 16
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal((n, d))     # independent -> mu_x - mu_y ~ 0
    dec = pmb_cob_decomposition(X, Y)
    assert dec["gamma_norm"] < 0.3
    # beta is RMS magnitude of independent random vector difference.
    # E[||x_i - y_i||^2] = 2d, so beta_norm ~ sqrt(2d).
    assert dec["beta_norm"] == pytest.approx(np.sqrt(2 * d), rel=0.1)


def test_js_divergence_self_is_small(rng):
    n, d = 1000, 32
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal((n, d))
    # Two samples from the same distribution -> JS ~ 0 (small finite-sample noise).
    assert js_divergence_angular(X, Y, n_sample_pairs=10000, seed=0) < 0.02


def test_js_divergence_anisotropic_vs_isotropic_is_larger(rng):
    n, d = 2000, 16
    X = rng.standard_normal((n, d))
    # Anisotropic: variance concentrated in first axis -> angles concentrated near 0/pi.
    sigmas = np.array([10.0] + [0.1] * (d - 1))
    Y = rng.standard_normal((n, d)) * sigmas
    js_iso = js_divergence_angular(X, rng.standard_normal((n, d)),
                                   n_sample_pairs=10000, seed=1)
    js_aniso = js_divergence_angular(X, Y, n_sample_pairs=10000, seed=1)
    assert js_aniso > 10 * js_iso


def test_compute_all_metrics_populates_spec_fields(rng):
    n, d = 500, 16
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal((n, d))
    m = compute_all_metrics(X, Y)
    # All new spec fields present and finite.
    for field in (
        "G_mu", "alpha_image", "alpha_text", "js_divergence_angular",
        "knn_mixing_rate_k20", "beta_norm", "gamma_norm", "pair_residual_energy",
        "kappa_image", "kappa_text", "eff_rank_image", "eff_rank_text",
        "trace_image", "trace_text",
    ):
        v = getattr(m, field)
        assert np.isfinite(v), f"{field}={v}"
    # to_dict() exposes them under "spec_metrics" and legacy under "extras".
    blob = m.to_dict()
    assert "spec_metrics" in blob and "extras" in blob
    assert "G_mu" in blob["spec_metrics"]
    assert "A_r" in blob["extras"]
