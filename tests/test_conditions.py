"""Smoke tests for the 4-condition orchestrator.

Use --dry-run to verify each condition's command sequence without actually
launching the training/measurement scripts.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
ORCHESTRATOR = REPO_ROOT / "scripts" / "run_condition.py"


def _dry_run(condition: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ORCHESTRATOR), "--condition", condition, "--dry-run"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize("condition", ["C0", "C1", "C2", "C3"])
def test_dry_run_succeeds(condition):
    out = _dry_run(condition)
    assert f"condition {condition} complete" in out


def test_c0_only_measures():
    out = _dry_run("C0")
    assert "07_extract_projected.py" in out
    assert "C0_random" in out
    # No training scripts in C0.
    assert "05_train_stage1.py" not in out
    assert "06_train_stage2.py" not in out


def test_c1_skips_stage1_runs_stage2():
    out = _dry_run("C1")
    assert "05_train_stage1.py" not in out
    assert "06_train_stage2.py" in out
    assert "--init-connector random" in out
    assert "C1_stage2" in out


def test_c2_runs_stage1_skips_stage2():
    out = _dry_run("C2")
    assert "05_train_stage1.py" in out
    assert "06_train_stage2.py" not in out
    assert "C2_stage1" in out


def test_c3_runs_both_with_two_measurements():
    out = _dry_run("C3")
    assert "05_train_stage1.py" in out
    assert "06_train_stage2.py" in out
    assert "C3_stage1" in out
    assert "C3_stage2" in out
    # Stage 2 initialized from Stage-1 checkpoint, not random.
    assert "--init-connector outputs/checkpoints/stage1_connector_C3.pt" in out
