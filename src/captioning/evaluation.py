"""COCO captioning scoring via pycocoevalcap.

Computes CIDEr (primary), BLEU-4, METEOR, SPICE. Java 8+ required for METEOR
and SPICE.
"""
from __future__ import annotations

import json
from pathlib import Path

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def _to_eval_format(refs_per_image: dict[int, list[str]], hyps_per_image: dict[int, str]):
    refs = {iid: [{"caption": c} for c in refs_per_image[iid]] for iid in hyps_per_image}
    hyps = {iid: [{"caption": hyps_per_image[iid]}] for iid in hyps_per_image}
    tok = PTBTokenizer()
    return tok.tokenize(refs), tok.tokenize(hyps)


def score_predictions(
    predictions_path: str | Path,
    karpathy_json: str | Path,
    out_path: str | Path,
) -> dict:
    with Path(predictions_path).open() as f:
        preds = json.load(f)
    with Path(karpathy_json).open() as f:
        kp = json.load(f)

    refs_per_image: dict[int, list[str]] = {}
    test_ids: set[int] = set()
    for entry in kp["images"]:
        if entry["split"] != "test":
            continue
        iid = entry["cocoid"]
        refs_per_image[iid] = [s["raw"] for s in entry["sentences"]]
        test_ids.add(iid)

    hyps_per_image = {row["image_id"]: row["caption"] for row in preds if row["image_id"] in test_ids}
    refs_tok, hyps_tok = _to_eval_format(refs_per_image, hyps_per_image)

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE"),
    ]

    scores: dict[str, float] = {}
    for scorer, name in scorers:
        s, _ = scorer.compute_score(refs_tok, hyps_tok)
        if isinstance(name, list):
            for n, v in zip(name, s):
                scores[n] = float(v)
        else:
            scores[name] = float(s)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"n_evaluated": len(hyps_per_image), "scores": scores}, f, indent=2)
    return scores
