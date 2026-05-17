"""Stage 2 full visual instruction tuning entry point.

A100 required (or set lora_fallback.enabled in configs/training_stage2.yaml).
Skeleton — InternVL conversation collator must be wired before first run.
"""
from __future__ import annotations

import argparse

from src.utils.io import load_yaml
from src.utils.reproducibility import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training_stage2.yaml")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    raise SystemExit(
        "Stub: VLM assembly + InternVL collator wire-up pending."
    )


if __name__ == "__main__":
    main()
