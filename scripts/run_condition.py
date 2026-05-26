"""Run one experimental condition end-to-end (Overleaf §5).

    C0: no training -> extract embeddings @ random connector -> metrics.
    C1: random connector -> Stage 2 -> extract + metrics + downstream evals.
    C2: Stage 1 -> extract + metrics; downstream evals through frozen LLaMA.
    C3: Stage 1 -> extract + metrics @ Stage 1
              -> Stage 2 (init from Stage 1) -> extract + metrics + downstream.

Each step shells out to the existing numbered scripts so the orchestrator
stays small and inspectable. Use ``--dry-run`` to print the planned commands.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def _run(cmd: list[str], dry: bool) -> None:
    print(f"[run] {' '.join(shlex.quote(c) for c in cmd)}")
    if dry:
        return
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _stage1(out_ckpt: str, dry: bool, extra: list[str]) -> None:
    cmd = [PY, "scripts/05_train_stage1.py", *extra]
    _run(cmd, dry)
    # Rename canonical Stage-1 ckpt into the condition-specific filename.
    canonical = REPO_ROOT / "outputs/checkpoints/stage1_connector.pt"
    target = REPO_ROOT / out_ckpt
    if not dry and canonical.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        canonical.replace(target)


def _stage2(init_connector: str, out_ckpt: str, dry: bool, extra: list[str]) -> None:
    cmd = [PY, "scripts/06_train_stage2.py", "--init-connector", init_connector, *extra]
    _run(cmd, dry)
    canonical = REPO_ROOT / "outputs/checkpoints/stage2_vlm.pt"
    target = REPO_ROOT / out_ckpt
    if not dry and canonical.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        canonical.replace(target)


def _measure(condition_tag: str, connector_override: str | None, dry: bool) -> None:
    extract_cmd = [PY, "scripts/07_extract_projected.py", "--condition", condition_tag]
    if connector_override is not None:
        extract_cmd += ["--connector-override", connector_override]
    _run(extract_cmd, dry)
    _run([PY, "scripts/03_compute_gap.py", "--condition", condition_tag], dry)
    _run([PY, "scripts/04_make_plots.py", "--condition", condition_tag], dry)


def _downstream(condition_tag: str, vlm_ckpt: str, dry: bool, skip_benchmarks: bool) -> None:
    """Run COCO val2017 captioning + (optionally) VLMEvalKit."""
    _run([PY, "scripts/08_run_captioning.py",
          "--condition", condition_tag, "--vlm-checkpoint", vlm_ckpt], dry)
    _run([PY, "scripts/09_score_captions.py", "--condition", condition_tag], dry)
    if not skip_benchmarks:
        _run([PY, "scripts/10_run_vlmevalkit.py",
              "--condition", condition_tag, "--vlm-checkpoint", vlm_ckpt], dry)


def run_c0(dry: bool, stage_extra: list[str], skip_downstream: bool = False) -> None:
    # Stage args unused (C0 trains nothing) but accepted for dispatch parity.
    del stage_extra, skip_downstream
    _measure("C0_random", connector_override="random", dry=dry)


def run_c1(dry: bool, stage_extra: list[str], skip_downstream: bool = False) -> None:
    out_ckpt = "outputs/checkpoints/stage2_vlm_C1.pt"
    _stage2(init_connector="random", out_ckpt=out_ckpt, dry=dry, extra=stage_extra)
    _measure("C1_stage2", connector_override=out_ckpt, dry=dry)
    if not skip_downstream:
        _downstream("C1_stage2", vlm_ckpt=out_ckpt, dry=dry, skip_benchmarks=False)


def run_c2(dry: bool, stage_extra: list[str], skip_downstream: bool = False) -> None:
    out_ckpt = "outputs/checkpoints/stage1_connector_C2.pt"
    _stage1(out_ckpt=out_ckpt, dry=dry, extra=stage_extra)
    _measure("C2_stage1", connector_override=out_ckpt, dry=dry)
    # No Stage 2 -> downstream uses random LoRA path through frozen LLaMA;
    # captioning is "zero-shot through Stage-1 connector + base LLaMA-2".
    if not skip_downstream:
        _downstream("C2_stage1", vlm_ckpt=out_ckpt, dry=dry, skip_benchmarks=False)


def run_c3(dry: bool, stage_extra: list[str], from_connector: str | None = None,
           skip_downstream: bool = False) -> None:
    s1_ckpt = "outputs/checkpoints/stage1_connector_C3.pt"
    s2_ckpt = "outputs/checkpoints/stage2_vlm_C3.pt"
    if from_connector is not None:
        # 2x2 factorial design: C2 and C3 share the same Stage-1 connector.
        # Symlink the existing checkpoint into C3's filename so the rest of
        # the pipeline (measure, Stage 2, downstream) stays unchanged.
        src = (REPO_ROOT / from_connector).resolve()
        dst = REPO_ROOT / s1_ckpt
        print(f"[run] reuse Stage-1 connector: {dst} -> {src}")
        if not dry:
            if not src.exists():
                raise FileNotFoundError(f"--from-connector source not found: {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
    else:
        _stage1(out_ckpt=s1_ckpt, dry=dry, extra=stage_extra)
    _measure("C3_stage1", connector_override=s1_ckpt, dry=dry)
    _stage2(init_connector=s1_ckpt, out_ckpt=s2_ckpt, dry=dry, extra=stage_extra)
    _measure("C3_stage2", connector_override=s2_ckpt, dry=dry)
    if not skip_downstream:
        _downstream("C3_stage2", vlm_ckpt=s2_ckpt, dry=dry, skip_benchmarks=False)


DISPATCH = {"C0": run_c0, "C1": run_c1, "C2": run_c2, "C3": run_c3}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=("C0", "C1", "C2", "C3"))
    p.add_argument("--dry-run", action="store_true", help="print commands without running")
    p.add_argument("--max-steps", type=int, default=None,
                   help="forwarded to Stage 1/2 trainers for smoke runs")
    p.add_argument("--from-connector", default=None,
                   help="C3 only: reuse this Stage-1 connector instead of training one. "
                        "E.g. outputs/checkpoints/stage1_connector_C2.pt to share C2's.")
    p.add_argument("--skip-downstream", action="store_true",
                   help="skip captioning / VLMEvalKit so the job only trains + measures. "
                        "Use when all conditions' downstream evals will be batched later.")
    args, unknown = p.parse_known_args()
    extra: list[str] = list(unknown)
    if args.max_steps is not None:
        extra += ["--max-steps", str(args.max_steps)]
    if args.from_connector is not None and args.condition != "C3":
        p.error("--from-connector is only valid with --condition C3")
    kwargs: dict = {"dry": args.dry_run, "stage_extra": extra,
                    "skip_downstream": args.skip_downstream}
    if args.condition == "C3":
        kwargs["from_connector"] = args.from_connector
    DISPATCH[args.condition](**kwargs)
    print(f"[ok] condition {args.condition} complete")


if __name__ == "__main__":
    main()
