"""Run VLMEvalKit benchmarks against one condition's checkpoint.

Spec §"Downstream Performance" benchmarks:
    General Perception : MME, MMStar, ScienceQA_VAL (image), RealWorldQA
    Complex Reasoning  : MMMU_DEV_VAL, MMMU_Pro, VisuLogic, LogicVista
    Hallucination      : CRPE, POPE, HallusionBench

Evaluated for C1, C2, C3 (C0 has no trained LLM and is skipped).

Usage:
    python scripts/10_run_vlmevalkit.py --condition C3_stage2 \
        --vlm-checkpoint outputs/checkpoints/stage2_vlm_C3.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.vlmevalkit_adapter import LlamaConnectorVLM
from src.utils.io import load_yaml, snapshot_run_metadata


DEFAULT_BENCHMARKS = [
    # General Perception
    "MME", "MMStar", "ScienceQA_VAL", "RealWorldQA",
    # Complex Reasoning
    "MMMU_DEV_VAL", "MMMU_Pro", "VisuLogic", "LogicVista",
    # Hallucination
    "CRPE", "POPE", "HallusionBench",
]

VLM_KEY = "llama2_connector_ablation"


def _run_benchmarks(model: LlamaConnectorVLM, datasets: list[str], out_dir: Path) -> dict:
    """Run ``datasets`` through VLMEvalKit's inference + scoring pipeline."""
    try:
        from vlmeval.inference import infer_data_job
        from vlmeval.utils.result_transfer import result_transfer
    except ImportError as e:
        raise ImportError(
            "vlmeval not installed. Run `pip install vlmeval` or reinstall the project."
        ) from e

    results: dict[str, dict] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        ds_out = out_dir / ds
        ds_out.mkdir(parents=True, exist_ok=True)
        try:
            infer_data_job(
                model=model,
                model_name=VLM_KEY,
                dataset_name=ds,
                work_dir=str(ds_out),
                verbose=True,
            )
            score = result_transfer(model_name=VLM_KEY, dataset_name=ds, work_dir=str(ds_out))
            results[ds] = score if isinstance(score, dict) else {"score": score}
        except Exception as e:   # noqa: BLE001
            results[ds] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[warn] {ds} failed: {e}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True,
                   help="condition tag (used only for output naming, e.g. C1_stage2)")
    p.add_argument("--vlm-checkpoint", required=True,
                   help="path to the Stage-2 checkpoint (or 'random' for C0 / C2 zero-shot)")
    p.add_argument("--stage2-config", default="configs/training_stage2.yaml",
                   help="for LoRA target_modules used to wrap the LLM at load time")
    p.add_argument("--benchmarks", nargs="*", default=DEFAULT_BENCHMARKS)
    p.add_argument("--out-dir", default="outputs/metrics")
    args = p.parse_args()

    stage2_cfg = load_yaml(args.stage2_config)
    lora_cfg = stage2_cfg.get("lora") if stage2_cfg.get("lora", {}).get("enabled") else None

    model = LlamaConnectorVLM(
        vlm_checkpoint=args.vlm_checkpoint,
        lora_cfg=lora_cfg,
    )

    out_root = Path(args.out_dir) / f"vlmevalkit_{args.condition}"
    results = _run_benchmarks(model, args.benchmarks, out_root)

    aggregate_path = Path(args.out_dir) / f"vlmevalkit_{args.condition}.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_path.open("w") as f:
        json.dump({"condition": args.condition, "results": results}, f, indent=2)

    snapshot_run_metadata(
        {"condition": args.condition, "checkpoint": args.vlm_checkpoint,
         "benchmarks": args.benchmarks, "args": vars(args)},
        out_root,
    )
    print(f"[ok] wrote {aggregate_path}")


if __name__ == "__main__":
    main()
