"""Stage 1 projector pretraining entry point.

Skeleton that wires config + components into ``train_stage1``. The data
collator / forward construction for modality substitution must be supplied in
``_build_dataloader`` once the Bunny manifest schema is verified at first
download.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.models.projector import build_projector
from src.training.stage1_pretrain import train_stage1
from src.utils.io import load_yaml, snapshot_run_metadata
from src.utils.reproducibility import set_seed


def _build_dataloader(cfg):
    """Construct text-only DataLoader for modality substitution.

    Placeholder: returns a callable that raises until wired up. The plan calls
    this out — Bunny schema is verified at first download (§6.3).
    """
    raise NotImplementedError(
        "Wire up Bunny modality-substitution dataloader once the dataset schema is verified."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training_stage1.yaml")
    p.add_argument("--projector-config", default="configs/projector.yaml")
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--llm-config", default="configs/llm.yaml")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    proj_cfg = load_yaml(args.projector_config)
    set_seed(cfg["seed"])

    projector = build_projector(proj_cfg["architecture"])

    # Encoder + LLM construction left to wire-up once the dataloader is ready.
    raise SystemExit(
        "Stub: encoder/LLM/dataloader wire-up pending. Verify Bunny schema first."
    )


if __name__ == "__main__":
    main()
