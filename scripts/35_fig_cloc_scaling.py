#!/usr/bin/env python3
"""Cloc scaling figure: convergence of the location drive and what it buys.

Four panels, all built from artefacts in this repository:

  (a) training-batch location gap per step, with the three evaluated
      checkpoints overlaid as full-set G_mu,
  (b) the autoregressive loss per step (collapse check),
  (c) the two pinned statistics as ratios to their own targets (isolation check),
  (d) the fraction of the baseline-to-reference distance closed at each
      checkpoint on the dd256 set.

Panels (a)-(c) are parsed from the SLURM stdout of the three training legs.
Step 1-100 come from job 91319 (killed by an NFS stall), 101-1529 from its
resume 91363, and 1530-2500 from the continuation 91561.  The legs overlap on
101-146; the later file wins, so every step appears exactly once.

Usage:
    PYTHONPATH=. python3 scripts/35_fig_cloc_scaling.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "logs"
METRICS = ROOT / "outputs" / "metrics"
OUT = ROOT / "outputs" / "figures"

# Training legs, in the order they ran.  Later files win on overlapping steps.
LEGS = ["cloc_long_91319.out", "cloc_long_91363.out", "cloc_cont_91561.out"]

# Pin targets, from scripts/submit_cloc_continue.sbatch (BTRACE0 / EFFRANK0).
BTRACE0 = 4767.3
EFFRANK0 = 23.31

RESTART_STEP = 1529          # end of leg 1; the warm restart begins at 1530
CKPTS = {"Cloc": 450, "Cloc_long": 1529, "Cloc_80k": 2500}

# Okabe-Ito, chosen for colour-vision deficiency separation.
BLUE, ORANGE, GREEN, VERM, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
GREY = "#999999"

FIELDS = ("L_ar", "L_dist", "L_scale", "L_rank", "effrank", "gap", "btrace", "lr")


def parse_leg(path: Path) -> dict[int, dict]:
    """One record per step.  A step opens with 'epoch=E step=S/T' and its
    remaining fields arrive on continuation lines before the next step."""
    records: dict[int, dict] = {}
    current: dict | None = None
    for line in path.read_text(errors="replace").splitlines():
        head = re.search(r"epoch=(\d+)\s+step=(\d+)/(\d+)", line)
        if head:
            if current is not None:
                records[current["step"]] = current
            current = {"step": int(head.group(2)), "total": int(head.group(3))}
        if current is None:
            continue
        for field in FIELDS:
            hit = re.search(rf"\b{field}=([0-9.eE+-]+)", line)
            if hit and field not in current:
                try:
                    current[field] = float(hit.group(1))
                except ValueError:
                    pass
    if current is not None:
        records[current["step"]] = current
    return records


def load_trace() -> dict[str, np.ndarray]:
    merged: dict[int, dict] = {}
    for name in LEGS:
        merged.update({s: r for s, r in parse_leg(LOGS / name).items() if "gap" in r})
    steps = sorted(merged)
    gaps = [s for s in range(1, steps[-1] + 1) if s not in merged]
    if gaps:
        raise SystemExit(f"[fig] {len(gaps)} steps missing from the logs, e.g. {gaps[:8]}")
    out = {"step": np.array(steps, dtype=float)}
    for field in FIELDS:
        out[field] = np.array([merged[s].get(field, np.nan) for s in steps], dtype=float)
    return out


def rolling_median(y: np.ndarray, window: int = 25) -> np.ndarray:
    pad = window // 2
    padded = np.pad(y, pad, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(view, axis=-1)[: len(y)]


def dd256_table() -> dict[str, dict[str, float]]:
    """Baseline, the three Cloc checkpoints and the reference, on dd256."""
    chair = json.loads((METRICS / "chair_summary_dd256.json").read_text())["per_condition"]

    def clip(tag: str) -> float:
        return json.loads((METRICS / f"clipscore_{tag}_dd256.json").read_text())["CLIPScore"]

    rows = {}
    for tag in ("C3pinr", "Cloc", "Cloc_long", "Cloc_80k"):
        c = chair[f"{tag}_dd256"]
        rows[tag] = {"CLIPScore": clip(tag), "chair_s": c["chair_s"],
                     "chair_i": c["chair_i"], "recall": c["recall"]}
    # The reference is scored by the same pipeline but is not in the summary
    # that ships with the repo; its CHAIR numbers are recomputed by
    # scripts/19_chair.py with --conditions llava15_dd256.
    rows["llava15"] = {"CLIPScore": clip("llava15"), "chair_s": 0.5069,
                       "chair_i": 0.1518, "recall": 0.7738}
    return rows


def eval_gap(tag: str) -> float:
    return json.loads((METRICS / f"gap_{tag}.json").read_text())["spec_metrics"]["G_mu"]



FIGSIZE = (6.4, 3.4)
RC = {
    "font.size": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
}
XMAX = 2560


def _new():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(0, XMAX)
    ax.set_xlabel("adapter step")
    return fig, ax


def _restart(ax, label_y=None):
    ax.axvline(RESTART_STEP, color=GREY, lw=0.9, ls=(0, (4, 3)), zorder=1)
    if label_y is not None:
        ax.text(RESTART_STEP - 45, label_y, "warm restart", ha="right", va="top",
                fontsize=8.5, color=GREY, rotation=90)


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] wrote {path}")


def fig_gap(tr):
    """The targeted axis closing, batch trace plus the evaluated checkpoints.

    Each checkpoint gets its own colour and is named in the legend, so no text
    sits next to the markers to collide with the curve or with its neighbour.
    """
    step = tr["step"]
    fig, ax = _new()
    ax.plot(step, tr["gap"], color=BLUE, lw=0.5, alpha=0.22, zorder=2)
    ax.plot(step, rolling_median(tr["gap"]), color=BLUE, lw=1.8, zorder=3,
            label="training batch")
    colours = {"Cloc": VERM, "Cloc_long": PURPLE, "Cloc_80k": GREEN}
    for tag, s in CKPTS.items():
        g = eval_gap(tag)
        ax.plot([s], [g], marker="o", ls="none", ms=7.5, color=colours[tag],
                mec="white", mew=1.3, zorder=5, clip_on=False,
                label=f"{tag}  {g:.2f}")
    _restart(ax, label_y=62)      # clear of the legend block above it
    ax.set_yscale("log")
    ax.set_ylim(9, 320)
    ax.set_ylabel("location gap")
    ax.legend(loc="upper right", frameon=False, handlelength=1.0,
              borderaxespad=0.3, labelspacing=0.25, fontsize=8,
              handletextpad=0.45)
    _save(fig, "cloc_gap")


def fig_loss(tr):
    """Collapse check. A log axis fits the opening value and the plateau in one
    frame, so nothing has to be annotated as running off the scale."""
    step = tr["step"]
    fig, ax = _new()
    ax.plot(step, tr["L_ar"], color=ORANGE, lw=0.5, alpha=0.22, zorder=2)
    ax.plot(step, rolling_median(tr["L_ar"]), color=ORANGE, lw=1.8, zorder=3)
    _restart(ax, label_y=13.5)
    ax.set_yscale("log")
    ax.set_ylim(0.65, 15)
    ax.set_yticks([1, 2, 5, 10])
    ax.set_yticklabels(["1", "2", "5", "10"])
    ax.set_ylabel("$\\mathcal{L}_{AR}$")
    _save(fig, "cloc_loss")


def fig_pins(tr):
    """Isolation check: both pinned statistics as ratios to their targets."""
    step = tr["step"]
    fig, ax = _new()
    ax.axhspan(0.9, 1.1, color=GREY, alpha=0.13, lw=0, zorder=1)
    ax.axhline(1.0, color=GREY, lw=0.9, zorder=2)
    ax.plot(step, rolling_median(tr["btrace"] / BTRACE0), color=GREEN, lw=1.6,
            zorder=3, label="scale  $\\operatorname{tr}\\Sigma_B/\\operatorname{tr}\\Sigma_0$")
    ax.plot(step, rolling_median(tr["effrank"] / EFFRANK0), color=PURPLE, lw=1.6,
            zorder=3, label="shape  $r_B/r_0$")
    _restart(ax, label_y=1.245)
    ax.set_ylim(0.75, 1.25)
    ax.set_ylabel("ratio to pin target")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, "cloc_pins")


ITEMS = {"Cloc": 14400, "Cloc_long": 48928, "Cloc_80k": 80000}
REF_ITEMS = 1_223_000        # LLaVA-1.5: 558K pre-training + 665K instruction


def fig_quality(rows):
    """All four metrics on one axis against training items, with a broken x-axis.

    Every metric here is a dimensionless rate living in the same part of [0, 1],
    so a shared y-axis is honest.  The x-axis is broken because the reference is
    trained on 1.22M items against the ladder's 14.4k to 80k: on one continuous
    log axis a single point would claim more width than the whole ladder, and on
    a linear one the ladder would collapse into the left margin.  The break is
    drawn explicitly so the spacing is never read as continuous.
    """
    from matplotlib.patches import ConnectionPatch

    tags = ("Cloc", "Cloc_long", "Cloc_80k")
    xs = [ITEMS[t] for t in tags]
    series = [("CLIPScore", "CLIPScore $\\uparrow$", BLUE, "o"),
              ("recall", "object recall $\\uparrow$", GREEN, "s"),
              ("chair_s", "$\\mathrm{CHAIR}_s$ $\\downarrow$", VERM, "v"),
              ("chair_i", "$\\mathrm{CHAIR}_i$ $\\downarrow$", PURPLE, "^")]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(7.0, 4.2), sharey=True,
        gridspec_kw={"width_ratios": [3.4, 1.0], "wspace": 0.07})

    for key, label, colour, marker in series:
        ref = rows["llava15"][key]
        ys = [rows[t][key] for t in tags]
        axL.plot(xs, ys, color=colour, lw=1.8, marker=marker, ms=6,
                 mec="white", mew=0.9, zorder=4, label=label)
        axR.plot([0.5], [ref], marker="*", ls="none", ms=16, color=colour,
                 mec="white", mew=0.9, zorder=5)
        # one straight dashed line drawn across both panels
        fig.add_artist(ConnectionPatch(
            xyA=(xs[-1], ys[-1]), coordsA=axL.transData,
            xyB=(0.5, ref), coordsB=axR.transData,
            color=colour, lw=1.3, ls=(0, (5, 4)), alpha=0.85, zorder=1))

    axL.set_xscale("log")
    axL.set_xlim(12_500, 95_000)
    axL.set_xticks(xs)
    axL.set_xticklabels(["14.4k", "48.9k", "80k"])
    axL.minorticks_off()
    axL.set_ylim(0.08, 0.98)
    axL.set_ylabel("metric value")
    axL.legend(loc="lower left", frameon=False, ncol=2, fontsize=9,
               columnspacing=1.3, handlelength=1.8, borderaxespad=0.5)

    axR.set_xlim(0, 1)
    axR.set_xticks([0.5])
    axR.set_xticklabels(["1.22M"])
    axR.tick_params(axis="y", length=0)
    axR.grid(axis="x", alpha=0.25, linewidth=0.6)
    axR.annotate("$\\bigstar$ LLaVA-1.5-7B", xy=(0.5, 0.955), ha="center",
                 va="top", fontsize=8.5, color=GREY)

    # the break itself: facing spines dropped, slashes drawn in their place
    axL.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=7, linestyle="none",
              color="k", mec="k", mew=1, clip_on=False)
    axL.plot([1], [0], transform=axL.transAxes, **kw)   # on the broken axis only
    axR.plot([0], [0], transform=axR.transAxes, **kw)

    fig.supxlabel("training items", fontsize=10, y=0.035)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.97, bottom=0.17)
    _save(fig, "cloc_quality")


def main() -> None:
    tr = load_trace()
    plt.rcParams.update(RC)
    fig_gap(tr)
    fig_loss(tr)
    fig_pins(tr)
    fig_quality(dd256_table())


if __name__ == "__main__":
    main()
