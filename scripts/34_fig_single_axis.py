"""Figure for Section 'The proposed objective': only the driven axis moves.

Reads the measured geometry straight from outputs/metrics so the figure cannot
drift from the numbers in the text. Values are shown relative to Cloc, the
450-step run, because all three conditions share the same pin setpoints
(btrace0 = 4767.3, effrank0 = 23.31) and are therefore directly comparable.
The all-pinned baseline C3pinr is NOT on the ladder: it carries different pin
anchors, so a ratio against it would mix a change in setpoint with a change in
outcome. Its value is quoted in the caption instead.

    python scripts/34_fig_single_axis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = Path("outputs/metrics")
OUT = [Path("outputs/figures/fig_single_axis.pdf"),
       Path("thesis draft/thesis_1/The provisional one/images/fig_single_axis.pdf")]

LADDER = [("Cloc", 14_400), ("Cloc_long", 48_928), ("Cloc_80k", 80_000)]

# validated categorical slots 1-4, light mode
SERIES = [
    ("location $G_\\mu$",       "#2a78d6", lambda s, e: s["G_mu"]),
    ("scale $\\mathrm{tr}\\Sigma_X$", "#eb6834", lambda s, e: s["trace_image"]),
    ("shape $r$",               "#1baf7a", lambda s, e: s["eff_rank_image"]),
    ("orientation $O_{16}$",    "#eda100", lambda s, e: e["subspace_overlap_q"]["16"]),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"


def load(tag):
    d = json.loads((M / f"gap_{tag}.json").read_text())
    return d["spec_metrics"], d.get("extras", {})


def main():
    raw = {t: load(t) for t, _ in LADDER}
    x = [n for _, n in LADDER]

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    ax.axhline(1.0, color=GRID, lw=1.2, zorder=1)

    rel = {}
    for label, colour, pick in SERIES:
        vals = [pick(*raw[t]) for t, _ in LADDER]
        rel[label] = [v / vals[0] for v in vals]
        ax.plot(x, rel[label], color=colour, lw=2.0, marker="o", ms=6,
                mec="#fcfcfb", mew=2.0, zorder=3, label=label, clip_on=False)

    # Only location has room on the main axes; the three held series are
    # direct-labelled inside the inset, which is where they are legible.
    lab, col, _ = SERIES[0]
    ax.annotate(f"{lab}\n$-{100*(1-rel[lab][-1]):.0f}\\%$",
                (x[-1], rel[lab][-1]), xytext=(9, 0), textcoords="offset points",
                color=col, fontsize=9, va="center", ha="left")

    # Inset: the three pinned or untouched axes at their own resolution, so the
    # reader can see they are flat rather than infer it from an overlap.
    ins = ax.inset_axes([0.46, 0.30, 0.50, 0.30])
    ins.set_facecolor("#fcfcfb")
    ins.axhline(1.0, color=GRID, lw=1.0, zorder=1)
    for label, colour, _ in SERIES[1:]:
        ins.plot(x, rel[label], color=colour, lw=1.8, marker="o", ms=4.5,
                 mec="#fcfcfb", mew=1.5, zorder=3)
        ins.annotate(label, (x[-1], rel[label][-1]), xytext=(6, 0),
                     textcoords="offset points", color=colour, fontsize=7.5,
                     va="center", ha="left", annotation_clip=False)
    ins.set_ylim(0.985, 1.020)
    ins.set_xlim(x[0] - 2500, x[-1] + 2500)
    ins.set_yticks([0.99, 1.00, 1.01])
    ins.set_yticklabels(["0.99", "1.00", "1.01"], fontsize=7)
    ins.set_xticks([])
    ins.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "bottom"):
        ins.spines[side].set_visible(False)
    ins.spines["left"].set_color(GRID)
    ins.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ins.set_axisbelow(True)
    ins.set_title("held and untouched axes, magnified", fontsize=7.5,
                  color=MUTED, pad=3, loc="left")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n//1000}k" for n in x])
    ax.set_xlabel("instruction items seen", color=MUTED, fontsize=9)
    ax.set_ylabel("value relative to the 450-step run", color=MUTED, fontsize=9)
    ax.set_ylim(0.40, 1.08)
    ax.set_xlim(x[0] - 2500, x[-1] + 26_000)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left",
              labelcolor=MUTED, handlelength=1.6, ncol=2,
              bbox_to_anchor=(0.0, 0.0))

    fig.tight_layout()
    for p in OUT:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[ok] {p}")

    print("\nvalues plotted (relative to the 450-step run):")
    for label, _, pick in SERIES:
        v = [pick(*raw[t]) for t, _ in LADDER]
        print(f"  {label:26} " + "  ".join(f"{q/v[0]:.4f}" for q in v)
              + "     raw " + "  ".join(f"{q:.4g}" for q in v))


if __name__ == "__main__":
    main()
