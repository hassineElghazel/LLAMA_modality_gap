"""Regression test for the demo's SCORING path (no GPU needed).

For every qualitative figure in the thesis it re-computes CLIPScore and H from
the stored predictions and compares against the value printed in the figure.
This validates the CLIPScore backbone/weight and the CHAIR synonym+GT pipeline.
It does NOT validate the model itself -- see demo/regression_live.py for that.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
from common import Chair, ClipScorer, image_path                  # noqa: E402

TEX = ROOT / "thesis draft/Thesis_official_Hassine/The provisional one/content/chapters/chap_experiments.tex"
COND = {                                   # figure row label -> predictions file
    "all-pinned": "C3pinr", "Cscale1500": "Cscale1500", "Crank15": "Crank15",
    "Corient": "Corient", "Cloc": "Cloc", "Clocorient": "Clocorient",
    "Cloc (450)": "Cloc", "Cloc_long (1529)": "Cloc_long",
    "Cloc_80k (2500)": "Cloc_80k", "LLaVA-1.5-7B": "llava15",
}

def rows():
    src = TEX.read_text(encoding="utf-8")
    for m in re.finditer(r"\\label\{(fig:qual:[a-z0-9:_]+)\}", src):
        blk = src[src.rindex("\\begin{figure}", 0, m.start()):m.start()]
        mi = re.search(r"includegraphics\[[^\]]*\]\{images/qq?_(\d+)\.jpg\}", blk)
        if not mi or "\\midrule" not in blk: continue
        iid = int(mi.group(1))
        body = blk[blk.index("\\midrule") + 8:blk.index("\\bottomrule")].replace("\\midrule", "")
        for line in body.split("\\\\"):
            c = [x.strip() for x in line.strip().split("&")]
            if len(c) < 5: continue
            name = re.sub(r"\\textbf\{|\}|\\,|\$\\star\$|\\", "", c[0]).strip()
            try: yield m.group(1), iid, name, float(c[2]), int(c[3])
            except ValueError: continue

def main():
    if not TEX.exists():
        print("thesis source not found:\n  " + str(TEX) +
              "\nThis test parses the thesis figures, so it only runs where the\n"
              "LaTeX lives (your laptop). Nothing is wrong with the demo.")
        return 0
    preds, cache = {}, {}
    chair, clip = Chair(), ClipScorer()
    ok = bad = skip = 0
    print(f"{'figure':24s} {'img':>7s} {'condition':18s} "
          f"{'CLIP thesis/demo':>20s}  {'H thesis/demo':>14s}")
    for fig, iid, name, clip_ref, h_ref in rows():
        tag = COND.get(name)
        p = ROOT / "outputs/predictions" / f"captions_{tag}_dd256.json" if tag else None
        if not p or not p.exists() or not image_path(iid).exists():
            skip += 1; continue
        if tag not in preds:
            preds[tag] = {d["image_id"]: d["caption"] for d in json.loads(p.read_text())}
        cap = preds[tag].get(iid)
        if cap is None: skip += 1; continue
        key = (tag, iid)
        if key not in cache:
            cache[key] = clip.score(Image.open(image_path(iid)).convert("RGB"), cap)
        cs, h = cache[key], chair.score(iid, cap)["H"]
        good = abs(cs - clip_ref) <= 0.0015 and h == h_ref
        ok, bad = ok + good, bad + (not good)
        print(f"{fig:24s} {iid:7d} {name:18s} "
              f"{clip_ref:8.3f} /{cs:8.3f}   {h_ref:5d} /{h:5d}   {'ok' if good else 'MISMATCH'}")
    print(f"\n{ok} match, {bad} mismatch, {skip} skipped")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
