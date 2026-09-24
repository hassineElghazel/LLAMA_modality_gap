#!/usr/bin/env python3
"""Four training-time figures covering all six factorial conditions.

Mirrors the four figures of the scaling subsection, but one figure per quantity
with six lines instead of one figure per condition.  Parsed from the pooled
SLURM logs; every condition ran 450 steps at an effective batch of 32.

Note: the per-step location gap is deliberately absent.  17_train_c5_distance
logs `gap` (distance to the text centroid) while the orientation-pinned trainer
logs `drift` (displacement from that run's own frozen centroid); they are
different quantities and cannot share an axis.  The evaluated G_mu for all six
is in tab:factorial_geometry.

Usage:  PYTHONPATH=. python3 scripts/36_fig_factorial_training.py
"""
from __future__ import annotations
import re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS, OUT = ROOT / "outputs" / "logs", ROOT / "outputs" / "figures"

JOBS = [("all-pinned", "pooled_86398.out", "#999999", "-"),
        ("Cscale1500", "pooled_86397.out", "#E69F00", "-"),
        ("Crank15",    "pooled_86422.out", "#CC79A7", "-"),
        ("Corient",    "pooled_86701.out", "#009E73", "-"),
        ("Cloc",       "pooled_86463.out", "#0072B2", "-"),
        ("Clocorient", "pooled_86991.out", "#D55E00", (0, (4, 2)))]
FIELDS = ("L_ar", "L_scale", "L_rank", "effrank", "gap", "Gmu", "drift", "L_nce")

def parse(p: Path) -> dict[str, np.ndarray]:
    recs, cur = {}, None
    for line in p.read_text(errors="replace").splitlines():
        h = re.search(r"epoch=\d+\s+step=(\d+)/", line)
        if h:
            cur = {"step": int(h.group(1))}; recs[cur["step"]] = cur
        if cur is None: continue
        for f in FIELDS:
            m = re.search(rf"\b{f}=([0-9.eE+-]+)", line)
            if m and f not in cur:
                try: cur[f] = float(m.group(1))
                except ValueError: pass
    steps = sorted(recs)
    out = {"step": np.array(steps, float)}
    for f in FIELDS:
        out[f] = np.array([recs[s].get(f, np.nan) for s in steps], float)
    return out

def roll(y, w=15):
    ok = ~np.isnan(y)
    if ok.sum() < w: return y
    pad = np.pad(y[ok], w // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, w), -1)[:ok.sum()]

def panel(name, field, ylabel, title="", log=False, ylim=None):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for label, f, colour, ls in JOBS:
        d = parse(LOGS / f)
        y, x = d[field], d["step"]
        ok = ~np.isnan(y)
        if ok.sum() == 0: continue
        ax.plot(x[ok], y[ok], color=colour, lw=0.5, alpha=0.18, zorder=2)
        ax.plot(x[ok][:len(roll(y[ok]))], roll(y[ok]), color=colour, lw=1.7,
                ls=ls, zorder=3, label=label)
    if log: ax.set_yscale("log")
    if ylim: ax.set_ylim(*ylim)
    ax.set_xlim(0, 455); ax.set_xlabel("adapter step"); ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=False, ncol=2, fontsize=8.5)
    fig.tight_layout(pad=0.5)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png"); plt.close(fig)
    print(f"[fig] wrote {OUT / (name + '.png')}")

BASE_GMU = 177.86     # all-pinned G_mu, tab:factorial_geometry

BTRACE0, EFFRANK0 = 4767.3, 23.30      # common baselines (auto-measured at step 1)
SCALE_TARGET = {"Cscale1500": 1500.0}   # everyone else pins at BTRACE0
CHANCE_NCE = float(np.log(64))          # InfoNCE at chance for a 64-image batch


def _axis_fig(ylabel, ylim, hline=None, band=None):
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    if band: ax.axhspan(*band, color="#999999", alpha=0.13, lw=0, zorder=0)
    if hline is not None: ax.axhline(hline, color="#666666", lw=1.0, zorder=1)
    ax.set_xlim(0, 455); ax.set_ylim(*ylim)
    ax.set_xlabel("adapter step"); ax.set_ylabel(ylabel)
    return fig, ax


def scale():
    """Total variance against the common baseline: only Cscale1500 leaves it."""
    fig, ax = _axis_fig(r"$\mathrm{tr}\Sigma_B\,/\,\mathrm{tr}\Sigma_0$",
                        (0.15, 1.45), hline=1.0, band=(0.9, 1.1))
    for label, f, colour, ls in JOBS:
        d = parse(LOGS / f); y, x = d["L_scale"], d["step"]
        ok = ~np.isnan(y)
        if not ok.sum(): continue
        tgt = SCALE_TARGET.get(label, BTRACE0)
        dev = np.sqrt(np.clip(y[ok], 0, None))
        lo, hi = roll(tgt * (1 - dev) / BTRACE0, 21), roll(tgt * (1 + dev) / BTRACE0, 21)
        xx = x[ok][:len(lo)]
        ax.fill_between(xx, lo, hi, color=colour, alpha=0.28, lw=0, zorder=2)
        ax.plot(xx, (lo + hi) / 2, color=colour, lw=1.5, ls=ls, zorder=3, label=label)
    # the pinned band sits at 1.0 and Cscale1500 at 0.32, so the gap between
    # them is the only clear space on this panel
    ax.legend(loc="center right", frameon=False, ncol=2, fontsize=8,
              bbox_to_anchor=(0.985, 0.42), borderaxespad=0)
    fig.tight_layout(pad=0.5); fig.savefig(OUT / "fact_scale.png"); plt.close(fig)
    print(f"[fig] wrote {OUT / 'fact_scale.png'}")


def shape():
    """Participation ratio against the common baseline: only Crank15 leaves it."""
    fig, ax = _axis_fig(r"$r_B\,/\,r_0$", (0.4, 1.45), hline=1.0, band=(0.9, 1.1))
    for label, f, colour, ls in JOBS:
        d = parse(LOGS / f); y, x = d["effrank"], d["step"]
        ok = ~np.isnan(y)
        if not ok.sum(): continue
        r = roll(y[ok] / EFFRANK0, 21)
        ax.plot(x[ok][:len(r)], r, color=colour, lw=1.5, ls=ls, zorder=3, label=label)
    ax.legend(loc="lower right", frameon=False, ncol=2, fontsize=8)
    fig.tight_layout(pad=0.5); fig.savefig(OUT / "fact_shape.png"); plt.close(fig)
    print(f"[fig] wrote {OUT / 'fact_shape.png'}")


def orientation():
    """The contrastive term, which is what the orientation drive optimises.

    Cloc is absent: the distance trainer has no contrastive leg and never
    evaluates it.  Its evaluated overlap is at baseline (0.0519 against 0.0515),
    so orientation is untouched there.
    """
    fig, ax = _axis_fig(r"$\mathcal{L}_{NCE}$", (1.4, 4.45), hline=CHANCE_NCE)
    ax.text(448, CHANCE_NCE, " chance", ha="right", va="bottom", fontsize=8,
            color="#666666")
    for label, f, colour, ls in JOBS:
        d = parse(LOGS / f); y, x = d["L_nce"], d["step"]
        ok = ~np.isnan(y)
        if not ok.sum(): continue
        r = roll(y[ok], 21)
        ax.plot(x[ok][:len(r)], r, color=colour, lw=1.5, ls=ls, zorder=3, label=label)
    ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=8)
    fig.tight_layout(pad=0.5); fig.savefig(OUT / "fact_orientation.png"); plt.close(fig)
    print(f"[fig] wrote {OUT / 'fact_orientation.png'}")


def location():
    """The location axis for all six.

    The two conditions that drive location log the gap to the text centroid
    directly.  The four that pin it log only displacement from their own frozen
    centroid, so their gap is not measured but is bounded by the triangle
    inequality to BASE_GMU +/- that displacement; this is drawn as a band.
    """
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    lo_all, hi_all, xs = None, None, None
    for label, f, colour, ls in JOBS:
        d = parse(LOGS / f)
        drift = d["drift"]
        if not np.isnan(drift).all():
            ok = ~np.isnan(drift)
            lo = np.abs(BASE_GMU - drift[ok]); hi = BASE_GMU + drift[ok]
            xs = d["step"][ok]
            lo_all = lo if lo_all is None else np.minimum(lo_all, lo)
            hi_all = hi if hi_all is None else np.maximum(hi_all, hi)
    ax.fill_between(xs, lo_all, hi_all, color="#999999", alpha=0.30, lw=0, zorder=2)
    ax.plot(xs, np.full_like(xs, BASE_GMU), color="#666666", lw=1.2, zorder=3,
            label="location pinned (4 conditions)")
    for label, f, colour, ls in [j for j in JOBS if j[0] in ("Cloc", "Clocorient")]:
        d = parse(LOGS / f)
        y = d["gap"] if not np.isnan(d["gap"]).all() else d["Gmu"]
        ok = ~np.isnan(y); r = roll(y[ok])
        ax.plot(d["step"][ok][:len(r)], r, color=colour, lw=1.8, ls=ls,
                zorder=4, label=label)
    ax.set_yscale("log"); ax.set_xlim(0, 455); ax.set_ylim(9, 320)
    ax.set_xlabel("adapter step"); ax.set_ylabel(r"location gap $G_\mu$")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5,
              borderaxespad=0.4, labelspacing=0.3)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "fact_location.png"); plt.close(fig)
    print(f"[fig] wrote {OUT / 'fact_location.png'}")


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "grid.linewidth": 0.6, "figure.dpi": 200, "savefig.dpi": 400,
        "savefig.bbox": "tight", "xtick.labelsize": 9, "ytick.labelsize": 9})
    panel("fact_loss", "L_ar", r"$\mathcal{L}_{AR}$",
          "the language objective holds in every condition", log=True, ylim=(0.6, 14))
    location()
    scale()
    shape()
    orientation()

if __name__ == "__main__":
    main()
