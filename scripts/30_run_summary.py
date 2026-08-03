"""Collect everything known about one arm into a compact summary block.

Reads whatever exists for a tag and prints a few lines of Markdown. Used for the
Telegram notifications (so a finished job reports NUMBERS, not just "done") and as
a quick terminal check. Every input is optional -- missing files are reported as
such rather than raising, because a job should still be able to tell you what it
DID produce when one scorer failed.

Inputs (all under outputs/):
    checkpoints/llava_docci_<arm>/train_sidecar.json   geometry the training reached
    metrics/gap_<tag>.json                             full gap suite on the eval set
    metrics/clipscore_<tag>.json                       reference-free alignment
    metrics/siglip_<tag>.json                          independent non-CLIP judge
    metrics/docci_refs_<tag>.json                      single-gold BLEU/CIDEr/METEOR

Usage:
    python scripts/30_run_summary.py --arm location --tag location_docci
    python scripts/30_run_summary.py --arm location --tag location_docci --compare pinned_docci
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "--"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True)
    p.add_argument("--tag", default=None, help="Metric tag (default: <arm>_docci).")
    p.add_argument("--compare", default=None,
                   help="Another tag to diff CLIPScore against (e.g. the causal control).")
    p.add_argument("--metrics-dir", default="outputs/metrics")
    p.add_argument("--ckpt-root", default="outputs/checkpoints")
    args = p.parse_args()

    tag = args.tag or f"{args.arm}_docci"
    md = Path(args.metrics_dir)
    lines: list[str] = [f"*{args.arm}*  (tag `{tag}`)"]

    # ---- training geometry ----
    sc = load(Path(args.ckpt_root) / f"llava_docci_{args.arm}" / "train_sidecar.json")
    if sc:
        lam = (f"l_d={sc.get('lambda_d')} l_p={sc.get('lambda_p')} "
               f"l_s={sc.get('lambda_s')} l_r={sc.get('lambda_r')}")
        lines.append(f"train: {lam}, {sc.get('total_steps')} steps, AR={fmt(sc.get('final_AR'))}")
        pre = sc.get("pretrained_G_mu_norm")
        fin = sc.get("final_G_mu_norm")
        if isinstance(fin, (int, float)):
            arrow = f"{fmt(pre,3)} -> {fmt(fin,3)}" if isinstance(pre, (int, float)) else fmt(fin, 3)
            lines.append(f"location gap (train, normalised): {arrow}")
        if isinstance(sc.get("final_pin_drift_norm"), (int, float)):
            lines.append(f"pin drift from pretrained centroid: {fmt(sc['final_pin_drift_norm'],3)}"
                         + ("  <- should be ~0 for the pinned arm" if sc.get("lambda_p") else ""))
        if sc.get("anchors_pins_frozen") is False:
            lines.append("WARNING: scale/rank setpoints were captured live, not frozen")

    # ---- gap on the eval set ----
    gap = load(md / f"gap_{tag}.json")
    if gap:
        sm = gap.get("spec_metrics", {})
        src = gap.get("source", {})
        tr = sm.get("trace_image")
        gmu = sm.get("G_mu")
        ng = src.get("G_mu_norm")
        if ng is None and isinstance(gmu, (int, float)) and isinstance(tr, (int, float)) and tr > 0:
            ng = gmu / (tr ** 0.5)
        lines.append(f"GAP (eval n={src.get('n_pairs', gap.get('n','?'))}): "
                     f"G_mu={fmt(gmu,2)} normalised={fmt(ng,3)} "
                     f"trace={fmt(tr,1)} effR={fmt(sm.get('eff_rank_image'),2)}")
    else:
        lines.append("GAP: not measured yet (run scripts/23 with --connector-path)")

    # ---- grounding ----
    cs = load(md / f"clipscore_{tag}.json")
    if cs:
        n = cs.get("n_evaluated")
        std = cs.get("CLIPScore_std")
        se = (std / (n ** 0.5)) if isinstance(std, (int, float)) and isinstance(n, int) and n else None
        lines.append(f"CLIPScore={fmt(cs.get('CLIPScore'))} (sd {fmt(std,3)}, "
                     f"SE {fmt(se,4)}, n={n})")
    sg = load(md / f"siglip_{tag}.json")
    if sg:
        lines.append(f"SigLIP={fmt(sg.get('SigLIPScore'))}")
    rf = load(md / f"docci_refs_{tag}.json")
    if rf:
        s = rf.get("scores", rf)
        lines.append("refs: " + " ".join(f"{k}={fmt(s[k])}" for k in
                                         ("BLEU-4", "CIDEr", "METEOR") if k in s))

    # ---- causal contrast ----
    if args.compare:
        other = load(md / f"clipscore_{args.compare}.json")
        if cs and other:
            a, b = cs.get("CLIPScore"), other.get("CLIPScore")
            na, nb = cs.get("n_evaluated"), other.get("n_evaluated")
            sa, sb = cs.get("CLIPScore_std"), other.get("CLIPScore_std")
            if all(isinstance(x, (int, float)) for x in (a, b, sa, sb)) and na and nb:
                se = ((sa ** 2) / na + (sb ** 2) / nb) ** 0.5
                z = (a - b) / se if se > 0 else float("nan")
                verdict = "SIGNIFICANT" if abs(z) >= 2 else "null"
                lines.append(f"vs `{args.compare}`: dCLIP={a-b:+.4f} "
                             f"(diff-SE {fmt(se,4)}, z={z:+.1f}) -> {verdict}")
        gother = load(md / f"gap_{args.compare}.json")
        if gap and gother:
            def ng_of(g):
                s = g.get("source", {})
                if isinstance(s.get("G_mu_norm"), (int, float)):
                    return s["G_mu_norm"]
                sm2 = g.get("spec_metrics", {})
                t = sm2.get("trace_image")
                return sm2.get("G_mu") / (t ** 0.5) if t else None
            lines.append(f"vs `{args.compare}`: normalised gap "
                         f"{fmt(ng_of(gother),3)} -> {fmt(ng_of(gap),3)}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
