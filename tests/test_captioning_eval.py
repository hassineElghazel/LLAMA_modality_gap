"""Smoke test for the captioning scorer (``score_predictions``).

Uses a tiny synthetic predictions JSON + references dict; asserts the scorer
returns numeric scores for CIDEr / BLEU-4 and writes a JSON file at the
requested path.

METEOR is excluded from these tests: meteor-1.5.jar resolves its data files via
a URL derived from the JAR's location; when the project path contains spaces
(``/Data science and engineering/...``) the URL becomes double-percent-encoded
and the paraphrase file lookup fails with a RuntimeException inside the JVM.
METEOR can be tested in production by running from a space-free path.

SPICE is also excluded (heavyweight Java SPICE jar); ``--no-spice`` in
``scripts/09_score_captions.py`` exercises this code path in real runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_score_predictions_returns_expected_keys(tmp_path):
    from src.captioning.evaluation import score_predictions

    # Three "image_ids" with three references each.
    references = {
        1: ["a brown dog runs in the grass", "a dog running outdoors", "brown dog on green grass"],
        2: ["two cats sleeping on a couch", "a pair of cats on a sofa", "cats napping together"],
        3: ["a person riding a bicycle", "someone on a bike", "a cyclist on the road"],
    }
    # Predictions deliberately exact-match one reference each so scores are
    # high enough for the test to be deterministic across pycocoevalcap versions.
    predictions = [
        {"image_id": 1, "caption": "a brown dog runs in the grass"},
        {"image_id": 2, "caption": "two cats sleeping on a couch"},
        {"image_id": 3, "caption": "a person riding a bicycle"},
    ]
    pred_path = tmp_path / "preds.json"
    pred_path.write_text(json.dumps(predictions))
    out_path = tmp_path / "scores.json"

    scores = score_predictions(
        pred_path, references, out_path, include_meteor=False, include_spice=False
    )

    # Expected metric keys (no METEOR/SPICE in unit tests — see module docstring).
    for key in ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "CIDEr"):
        assert key in scores, f"missing metric: {key}"
        assert isinstance(scores[key], float)
        assert scores[key] == scores[key]   # not NaN

    # With exact-match predictions BLEU-4 and CIDEr should be strong.
    assert scores["BLEU-4"] > 0.5
    assert scores["CIDEr"] > 1.0   # CIDEr can exceed 1 by construction.

    # Output file written with n_evaluated.
    assert out_path.exists()
    blob = json.loads(out_path.read_text())
    assert blob["n_evaluated"] == 3
    assert set(blob["scores"]) == set(scores)


def test_score_predictions_filters_to_references(tmp_path):
    """Predictions outside the references dict are silently dropped."""
    from src.captioning.evaluation import score_predictions

    references = {1: ["a brown dog runs"]}
    predictions = [
        {"image_id": 1, "caption": "a brown dog runs"},
        {"image_id": 999, "caption": "spurious"},   # not in references -> ignored
    ]
    pred_path = tmp_path / "preds.json"
    pred_path.write_text(json.dumps(predictions))
    out_path = tmp_path / "scores.json"

    score_predictions(
        pred_path, references, out_path, include_meteor=False, include_spice=False
    )
    blob = json.loads(out_path.read_text())
    assert blob["n_evaluated"] == 1
