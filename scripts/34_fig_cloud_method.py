"""Figure for Section 'The proposed objective': the measured cloud under the recipe.

Shows the real pooled-257 embeddings, not a schematic. The two panels are the
all-pinned baseline and Cloc, drawn in ONE shared 2D frame so the comparison is
geometric rather than two unrelated projections:

    axis 1  the baseline gap direction, u = (xbar_base - ybar) / ||.||
    axis 2  the leading direction of the baseline image covariance, orthogonalised
            against u

Both panels share limits, so a cloud that changed size would look like it changed
size. Nothing here uses the scaled runs; this section is about the objective, not
about how far it can be pushed.

    python scripts/34_fig_cloud_method.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

E = Path("outputs/embeddings")
M = Path("outputs/metrics")
OUT = [Path("outputs/figures/fig_cloud_method.pdf"),
       Path("thesis draft/thesis_1/The provisional one/images/fig_cloud_method.pdf")]

PANELS = [("C3pinr", "all-pinned baseline"), ("Cloc", "Cloc: location driven")]
IMG, TXT = "#2a78d6", "#eb6834"          # validated categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
N_SHOW, SEED = 1200, 0


def load(tag):
    x = torch.load(E / f"projected_{tag}_image_pooled.pt").double().numpy()
    y = torch.load(E / f"projected_{tag}_text_pooled.pt").double().numpy()
    g = json.loads((M / f"gap_{tag}.json").read_text())
    return x, y, g["spec_metrics"]


def main():
    data = {t: load(t) for t, _ in PANELS}

    # Shared frame, built from the BASELINE only.
    xb, yb, _ = data["C3pinr"]
    u = xb.mean(0) - yb.mean(0)
    u /= np.linalg.norm(u)
    xc = xb - xb.mean(0)
    xc -= np.outer(xc @ u, u)                     # remove the gap direction
    v = np.linalg.svd(xc, full_matrices=False)[2][0]
    v -= (v @ u) * u
    v /= np.linalg.norm(v)
    origin = yb.mean(0)                           # text centroid is the origin

    rng = np.random.default_rng(SEED)
    sns.set_theme(style="ticks", font_scale=0.9)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.25), sharex=True, sharey=True)

    for ax, (tag, title) in zip(axes, PANELS):
        x, y, s = data[tag]
        px = np.c_[(x - origin) @ u, (x - origin) @ v]
        py = np.c_[(y - origin) @ u, (y - origin) @ v]
        ix = rng.choice(len(px), min(N_SHOW, len(px)), replace=False)
        iy = rng.choice(len(py), min(N_SHOW, len(py)), replace=False)

        sns.scatterplot(x=px[ix, 0], y=px[ix, 1], ax=ax, s=7, color=IMG,
                        alpha=0.30, linewidth=0, rasterized=True,
                        label="image $X$" if tag == "C3pinr" else None)
        sns.scatterplot(x=py[iy, 0], y=py[iy, 1], ax=ax, s=7, color=TXT,
                        alpha=0.55, linewidth=0, rasterized=True,
                        label="text $Y$" if tag == "C3pinr" else None)
        ax.scatter(*px.mean(0), s=46, color=IMG, ec="white", lw=1.4, zorder=5)
        ax.scatter(*py.mean(0), s=46, color=TXT, ec="white", lw=1.4, zorder=5)

        # the centroid distance, drawn
        ax.annotate("", xy=tuple(py.mean(0)), xytext=tuple(px.mean(0)),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1,
                                    shrinkA=5, shrinkB=5, alpha=0.75), zorder=6)
        ax.text(*(0.5 * (px.mean(0) + py.mean(0)) + np.array([0, 9])),
                f"$G_\\mu={s['G_mu']:.1f}$", fontsize=8.5, color=INK,
                ha="center", zorder=7)

        ax.set_title(title, fontsize=9.5, color=INK, pad=7)
        ax.text(0.5, -0.30,
                f"$\\mathrm{{tr}}\\Sigma_X={s['trace_image']:.0f}$"
                f"   $r={s['eff_rank_image']:.1f}$",
                transform=ax.transAxes, ha="center", fontsize=8.5, color=MUTED)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(colors=MUTED, labelsize=8)

    axes[0].set_ylabel("orthogonal direction", color=MUTED, fontsize=9)
    fig.supxlabel("baseline gap direction $\\hat u$  (text centroid at $0$)",
                  color=MUTED, fontsize=9, y=0.10)
    h, l = axes[0].get_legend_handles_labels()
    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()
    leg = fig.legend(h, l, frameon=False, fontsize=8.5, ncol=2,
                     loc="lower center", bbox_to_anchor=(0.5, -0.02),
                     markerscale=2.6, labelcolor=MUTED)
    sns.despine(fig=fig)
    fig.tight_layout(rect=(0, 0.13, 1, 1))

    for p in OUT:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"[ok] {p}")

    for tag, _ in PANELS:
        s = data[tag][2]
        print(f"  {tag:10} G_mu={s['G_mu']:8.2f}  trace={s['trace_image']:8.1f}  "
              f"r={s['eff_rank_image']:.2f}")


if __name__ == "__main__":
    main()
