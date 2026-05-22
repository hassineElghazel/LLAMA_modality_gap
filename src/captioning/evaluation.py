"""COCO val2017 captioning scoring via pycocoevalcap.

Computes CIDEr (primary), BLEU-4, METEOR (per Overleaf spec
§"Captioning Quality"). Java 8+ required for METEOR and SPICE.

The scoring entry point ``score_predictions`` takes a generic
``references: dict[image_id, list[str]]`` so it works with any
annotation source (COCO val2017 here; other COCO splits trivially).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def _to_eval_format(refs_per_image: dict, hyps_per_image: dict):
    refs = {iid: [{"caption": c} for c in refs_per_image[iid]] for iid in hyps_per_image}
    hyps = {iid: [{"caption": hyps_per_image[iid]}] for iid in hyps_per_image}
    tok = PTBTokenizer()
    return tok.tokenize(refs), tok.tokenize(hyps)


def score_predictions(
    predictions_path: str | Path,
    references: dict[int, list[str]],
    out_path: str | Path,
    *,
    include_spice: bool = True,
) -> dict:
    """Score predictions against a references dict.

    Args:
        predictions_path: JSON list of ``{"image_id": int, "caption": str}``.
        references: ``image_id -> list[reference_caption]``.
        out_path: where to write the JSON scores.
        include_spice: SPICE adds Java overhead; disable for fast smoke runs.
    """
    with Path(predictions_path).open() as f:
        preds = json.load(f)

    eval_ids = set(references)
    hyps = {row["image_id"]: row["caption"] for row in preds if row["image_id"] in eval_ids}
    refs_tok, hyps_tok = _to_eval_format(references, hyps)

    scorers: list[tuple[object, object]] = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Cider(), "CIDEr"),
    ]
    if include_spice:
        scorers.append((Spice(), "SPICE"))

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
        json.dump({"n_evaluated": len(hyps), "scores": scores}, f, indent=2)
    return scores
