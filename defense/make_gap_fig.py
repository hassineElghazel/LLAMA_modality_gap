"""Intuitive modality-gap illustration: real COCO images from the thesis on one
side, their captions on the other, matched pairs joined across the gap."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import os

for f in ("Quicksand-Regular.ttf", "Quicksand-Bold.ttf", "Quicksand-Medium.ttf"):
    p = os.path.expanduser("~/Library/Fonts/" + f)
    if os.path.exists(p): font_manager.fontManager.addfont(p)
plt.rcParams["font.family"] = "Quicksand"

BODY, MUTED, ACC = "#262626", "#707070", "#C06A3E"
IMG_C, TXT_C = "#3B6EA5", "#B03A48"

PAIRS = [("gapimg_qq_85823.jpg",  "a group of zebras standing\nin a grassy field",  1.05,  3.05),
         ("gapimg_qq_113403.jpg", "two stuffed animals\nlying on a bed",           1.62,  1.85),
         ("gapimg_qq_34760.jpg",  "a bathroom with a white\nsink and a toilet",     1.00,  0.70),
         ("gapimg_qq_134882.jpg", "a black and white cat\nsitting on a bed",        1.55, -0.40)]
HALF_W  = 0.66          # every thumbnail is cropped to 4:3, so one half-width
CAP_X   = 7.15

def crop43(im):
    w, h = im.size
    if w / h > 4 / 3: nw = int(h * 4 / 3); im = im.crop(((w - nw)//2, 0, (w + nw)//2, h))
    else:             nh = int(w * 3 / 4); im = im.crop((0, (h - nh)//2, w, (h + nh)//2))
    return im

fig, ax = plt.subplots(figsize=(9.6, 4.35), dpi=210)
ax.set_xlim(-0.35, 11.05); ax.set_ylim(-1.75, 4.45); ax.axis("off")
fig.patch.set_facecolor("white")

ax.add_patch(Ellipse((1.24, 1.27), 3.30, 5.15, angle=-5, facecolor=IMG_C,
                     alpha=0.055, edgecolor=IMG_C, lw=1.1, ls=(0, (5, 4))))
ax.add_patch(Ellipse((8.62, 1.27), 4.45, 5.15, angle=5, facecolor=TXT_C,
                     alpha=0.05, edgecolor=TXT_C, lw=1.1, ls=(0, (5, 4))))

for fn, cap, ix, iy in PAIRS:                     # matched-pair threads first
    ax.plot([ix + HALF_W, CAP_X - 0.12], [iy, iy], color="#9A9A9A", lw=0.9,
            alpha=0.85, zorder=1)

for fn, cap, ix, iy in PAIRS:
    im = crop43(Image.open("fig/" + fn)); im.thumbnail((300, 300))
    ab = AnnotationBbox(OffsetImage(im, zoom=0.24),
                        (ix, iy), frameon=True, pad=0.0, zorder=3,
                        bboxprops=dict(edgecolor=IMG_C, lw=1.3,
                                       boxstyle="round,pad=0.012"))
    ax.add_artist(ab)
    ax.text(CAP_X, iy, cap, fontsize=10.5, color=BODY, va="center", ha="left",
            zorder=3, linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor=TXT_C, lw=1.1))

ax.add_patch(FancyArrowPatch((3.05, 3.92), (6.85, 3.92), arrowstyle="<->",
                             mutation_scale=15, color=ACC, lw=1.7))
ax.text(4.95, 4.10, "the modality gap", fontsize=13, color=ACC,
        ha="center", va="bottom", fontweight="bold")
ax.text(1.24, -1.52, "image embeddings", fontsize=12, color=IMG_C,
        ha="center", fontweight="bold")
ax.text(8.62, -1.52, "caption embeddings", fontsize=12, color=TXT_C,
        ha="center", fontweight="bold")
ax.text(4.95, -1.52, "one shared space", fontsize=11, color=MUTED, ha="center",
        style="italic")

fig.tight_layout(pad=0.15)
fig.savefig("fig/fig_gap_intuition.png", dpi=210, facecolor="white",
            bbox_inches="tight", pad_inches=0.06)
print("wrote fig/fig_gap_intuition.png",
      Image.open("fig/fig_gap_intuition.png").size)
