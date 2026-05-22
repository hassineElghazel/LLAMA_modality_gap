"""Smoke tests for the diagnostic plotting functions.

Verifies that ``make_all_figures`` and ``plot_gap_decomposition`` produce
non-empty PNG files from synthetic data, without requiring a GPU or real
checkpoint. We use matplotlib's non-interactive Agg backend to avoid
display dependencies in CI.
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")   # must be set before any other matplotlib import

import numpy as np
import pytest


RNG = np.random.default_rng(0)
N, D = 200, 32   # small enough to be fast; large enough to avoid degenerate covariances


def _synth_pair(rng=RNG, n=N, d=D):
    """Return (X, Y) with a small centroid gap and moderate shared covariance."""
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = X + rng.standard_normal((n, d)).astype(np.float32) * 0.5 + 0.5
    return X, Y.astype(np.float32)


# ---------------------------------------------------------------------------
# make_all_figures
# ---------------------------------------------------------------------------

def test_make_all_figures_writes_nonempty_pngs(tmp_path):
    from src.diagnostics.plots import make_all_figures

    X, Y = _synth_pair()
    result = make_all_figures(X, Y, tmp_path)

    assert set(result.keys()) == {"A", "B", "C", "D"}
    for key, (png, pdf) in result.items():
        assert png.exists(), f"PNG missing for figure {key}"
        assert png.stat().st_size > 0, f"PNG empty for figure {key}"
        assert pdf.exists(), f"PDF missing for figure {key}"
        assert pdf.stat().st_size > 0, f"PDF empty for figure {key}"


def test_make_all_figures_creates_output_dir(tmp_path):
    from src.diagnostics.plots import make_all_figures

    X, Y = _synth_pair()
    out = tmp_path / "deeply" / "nested" / "out"
    assert not out.exists()
    make_all_figures(X, Y, out)
    assert out.exists()
    assert any(out.iterdir())


# ---------------------------------------------------------------------------
# individual figure functions
# ---------------------------------------------------------------------------

def test_figure_A_produces_png(tmp_path):
    from src.diagnostics.plots import figure_A_dominant_geometry

    X, Y = _synth_pair()
    png, _ = figure_A_dominant_geometry(X, Y, tmp_path)
    assert png.exists() and png.stat().st_size > 0


def test_figure_B_produces_png(tmp_path):
    from src.diagnostics.plots import figure_B_anisotropic_residual

    X, Y = _synth_pair()
    png, _ = figure_B_anisotropic_residual(X, Y, tmp_path)
    assert png.exists() and png.stat().st_size > 0


def test_figure_C_produces_png(tmp_path):
    from src.diagnostics.plots import figure_C_pca_scatter

    X, Y = _synth_pair()
    png, _ = figure_C_pca_scatter(X, Y, tmp_path)
    assert png.exists() and png.stat().st_size > 0


def test_figure_D_produces_png(tmp_path):
    from src.diagnostics.plots import figure_D_angular_topology

    X, Y = _synth_pair()
    png, _ = figure_D_angular_topology(X, Y, tmp_path)
    assert png.exists() and png.stat().st_size > 0


# ---------------------------------------------------------------------------
# plot_gap_decomposition
# ---------------------------------------------------------------------------

def test_plot_gap_decomposition_all_conditions(tmp_path):
    from src.diagnostics.plots import plot_gap_decomposition, TRAJECTORY_ORDER

    rng = np.random.default_rng(1)
    metrics = {
        cond: {
            "beta_norm": float(rng.uniform(0.1, 2.0)),
            "gamma_norm": float(rng.uniform(0.1, 2.0)),
            "kappa_image": float(rng.uniform(1, 50)),
            "kappa_text": float(rng.uniform(1, 50)),
            "G_mu": float(rng.uniform(0.1, 2.0)),
            "js_divergence_angular": float(rng.uniform(0.0, 0.5)),
        }
        for cond in TRAJECTORY_ORDER
    }

    png = plot_gap_decomposition(metrics, tmp_path)
    assert png.exists()
    assert png.stat().st_size > 0


def test_plot_gap_decomposition_partial_conditions(tmp_path):
    """Missing conditions are silently skipped; plot still produced."""
    from src.diagnostics.plots import plot_gap_decomposition

    metrics = {
        "C0_random": {"beta_norm": 1.0, "gamma_norm": 1.5, "kappa_image": 10.0,
                      "kappa_text": 8.0, "G_mu": 1.2, "js_divergence_angular": 0.3},
        "C3_stage2": {"beta_norm": 0.4, "gamma_norm": 0.6, "kappa_image": 3.0,
                      "kappa_text": 2.5, "G_mu": 0.5, "js_divergence_angular": 0.1},
    }

    png = plot_gap_decomposition(metrics, tmp_path)
    assert png.exists() and png.stat().st_size > 0


def test_plot_gap_decomposition_no_conditions_raises():
    from src.diagnostics.plots import plot_gap_decomposition
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError, match="no recognised conditions"):
            plot_gap_decomposition({"UNKNOWN_COND": {}}, td)
