"""Method figure: what the objective does to the image cloud.

One panel. The frozen text cloud sits at the origin, the image cloud is drawn
where it starts and where the location drive takes it, and the two pins are
named against the property they hold. There are deliberately no axis numbers and
no measured values: this belongs to the methodology, so it states the mechanism,
not the result.

Shapes come from the real pooled-257 embeddings, projected into the plane spanned
by the gap direction and the leading orthogonal direction of the image covariance,
so the clouds are honest even though nothing is quantified.

    python scripts/34_fig_cloud_method.py            # real embeddings
    python scripts/34_fig_cloud_method.py --demo     # synthetic, for layout checks
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

E = Path("outputs/embeddings")
OUT = [Path("outputs/figures/fig_cloud_method.png"),
       Path("thesis draft/thesis_1/The provisional one/images/fig_cloud_method.png")]

BEFORE, AFTER, TEXT = "#9db8dd", "#2a78d6", "#eb6834"
INK, MUTED = "#1b1b1a", "#6b6a64"


def real():
    import torch, json
    def L(tag, side):
        return torch.load(E / f"projected_{tag}_{side}_pooled.pt").double().numpy()
    xb, y, xa = L("C3pinr", "image"), L("C3pinr", "text"), L("Cloc", "image")
    u = xb.mean(0) - y.mean(0); u /= np.linalg.norm(u)
    c = xb - xb.mean(0); c -= np.outer(c @ u, u)
    v = np.linalg.svd(c, full_matrices=False)[2][0]
    v -= (v @ u) * u; v /= np.linalg.norm(v)
    o = y.mean(0)
    # negated so the panel reads left to right: before, after, text
    P = lambda a: np.c_[-((a - o) @ u), (a - o) @ v]
    return P(xb), P(xa), P(y)


def demo():
    """Synthetic clouds with the same gross geometry, for checking layout."""
    g = np.random.default_rng(3)
    cov = np.array([[70.0, 18.0], [18.0, 300.0]])
    base = g.multivariate_normal([-178, 0], cov, 1400)
    after = g.multivariate_normal([-31, 0], cov, 1400)
    txt = g.multivariate_normal([0, 0], [[3.0, 0], [0, 3.0]], 1400)
    return base, after, txt


def blob(ax, pts, color, *, fill, lw, ls="-", alpha=0.85, z=2):
    sns.kdeplot(x=pts[:, 0], y=pts[:, 1], ax=ax, levels=[0.15, 0.4, 0.75],
                fill=fill, color=color, linewidths=lw, linestyles=ls,
                alpha=alpha, zorder=z, bw_adjust=1.15,
                **({"thresh": 0.12} if fill else {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    before, after, txt = demo() if a.demo else real()

    sns.set_theme(style="white", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    blob(ax, before, BEFORE, fill=False, lw=1.5, ls="--", alpha=0.95, z=2)
    blob(ax, after,  AFTER,  fill=True,  lw=0,   alpha=0.55, z=3)
    blob(ax, after,  AFTER,  fill=False, lw=1.6, alpha=0.95, z=4)
    blob(ax, txt,    TEXT,   fill=True,  lw=0,   alpha=0.70, z=3)
    blob(ax, txt,    TEXT,   fill=False, lw=1.6, alpha=0.95, z=4)

    cb, ca, ct = before.mean(0), after.mean(0), txt.mean(0)
    for c, col in ((cb, BEFORE), (ca, AFTER), (ct, TEXT)):
        ax.scatter(*c, s=34, color=col, ec="white", lw=1.6, zorder=6)

    ax.annotate("", xy=(ca[0], ca[1]), xytext=(cb[0], cb[1]), zorder=5,
                arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.55",
                                color=INK, lw=1.6, shrinkA=9, shrinkB=9))

    top = max(before[:, 1].max(), after[:, 1].max())
    mid = 0.5 * (cb + ca)

    ax.text(mid[0], mid[1] + 0.30 * top, r"$\mathcal{L}_{\mathrm{dist}}$",
            ha="center", va="bottom", fontsize=13, color=INK, zorder=7)
    ax.text(mid[0], mid[1] + 0.13 * top, "drives location", ha="center",
            va="bottom", fontsize=9.5, color=MUTED, zorder=7)

    ax.annotate(r"$\mathcal{L}_{\mathrm{scale}}$, $\mathcal{L}_{\mathrm{rank}}$"
                "\nhold extent and shape",
                xy=(ca[0] + 20, ca[1] + 0.62 * top),
                xytext=(ca[0] + 105, top * 1.02),
                fontsize=9.5, color=MUTED, ha="left", va="center", linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color="#c4c3bc", lw=1.1,
                                connectionstyle="arc3,rad=0.22"))

    ax.text(cb[0], cb[1] - 0.80 * top, "before", ha="center", va="top",
            fontsize=9.5, color=BEFORE)
    ax.text(ca[0], ca[1] - 0.80 * top, "after", ha="center", va="top",
            fontsize=9.5, color=AFTER)
    ax.text(ct[0] + 26, ct[1], "text $Y$\nfrozen", ha="left", va="center",
            fontsize=9.5, color=TEXT, linespacing=1.35)
    ax.text(mid[0], -top * 1.12, "orientation carries no term",
            ha="center", va="center", fontsize=9.5, color=MUTED)

    ax.set_xlim(cb[0] - 90, ct[0] + 130)
    ax.set_ylim(-top * 1.32, top * 1.36)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    sns.despine(ax=ax, left=True, bottom=True)
    fig.tight_layout()

    for p in OUT:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[ok] {p}")


if __name__ == "__main__":
    main()
