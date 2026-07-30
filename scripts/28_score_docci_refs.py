"""Reference-based scoring of DOCCI-test captions against the single gold ref.

Uses the DOCCI ``description`` as the reference (one per image) via
``docci_references``. Reports SPICE (scene-graph semantic F1, meaningful with a
single reference) and METEOR (stem/synonym-aware) as the headline metrics; BLEU-4
and CIDEr are also emitted by ``score_predictions`` but CIDEr is TF-IDF over the
reference set and is noisy with a single reference -- read SPICE/METEOR here.

FAIRNESS: circular against Arm0 (the zero-FT reference never saw DOCCI's style),
so Arm0's reference-based numbers will look low BY CONSTRUCTION -- that contrast is
only meaningful reference-FREE (CLIPScore/SigLIP). BUT this metric is FAIR BETWEEN
the geometry arms (vanilla/pinned/location), which share identical DOCCI training
data and differ only in the geometry term. Read it as a geometry-arm contrast.

Usage (needs Java for METEOR/SPICE):
    python scripts/28_score_docci_refs.py --tag location_docci \
        --manifest data/docci/test_manifest.json --config configs/docci_eval.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.captioning.evaluation import score_predictions
from src.data.docci_loader import docci_references
from src.utils.io import load_yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/docci_eval.yaml")
    p.add_argument("--tag", required=True, help="tag matching captions_<tag>.json")
    p.add_argument("--manifest", default=None,
                   help="DOCCI manifest for refs (default: eval_set.manifest_path)")
    p.add_argument("--no-spice", action="store_true", help="skip SPICE (avoids SPICE jar)")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    manifest = args.manifest or cfg["eval_set"]["manifest_path"]
    references = docci_references(manifest)

    pred_path = Path(cfg["output"]["predictions_dir"]) / f"captions_{args.tag}.json"
    scores_path = Path(cfg["output"]["scores_dir"]) / f"docci_refs_{args.tag}.json"
    scores = score_predictions(
        pred_path, references, scores_path,
        include_meteor=True, include_spice=not args.no_spice,
    )
    print(f"[ok] {args.tag} (DOCCI single-ref): "
          + " ".join(f"{k}={v:.4f}" for k, v in scores.items()))
    print(f"[ok] wrote {scores_path}  (read SPICE/METEOR; CIDEr noisy @1 ref)")


if __name__ == "__main__":
    main()
