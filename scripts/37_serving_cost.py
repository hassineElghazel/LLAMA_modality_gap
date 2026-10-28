"""Serving-cost measurement: what it costs to DEPLOY a condition.

The paper contains no number in milliseconds or megabytes. For a systems venue
that is a gap regardless of prose. This measures, per condition:

  * parameter counts (total / trainable-at-serving / connector / LoRA)
  * checkpoint bytes on disk
  * peak inference VRAM
  * per-caption latency, batch=1 (the serving case): p50, p95, mean
  * throughput, captions/second

The expected RESULT is that every column is identical across conditions, because
the geometry terms act only during training. That identity IS the finding: the
served artifact is the same shape while caption quality differs. Do not "fix" a
null difference here.

Reuses _build_vlm from scripts/08_run_captioning.py so the model is constructed
by exactly the same path the captioning numbers came from. If that builder
changes, this follows it automatically.

Usage:
  python scripts/37_serving_cost.py --condition C3pinr \
      --vlm-checkpoint outputs/checkpoints/stage2_vlm_C3pinr.pt \
      --n-images 500 --warmup 10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

from src.data.coco_val2017_loader import CocoVal2017Dataset
from src.utils.io import load_yaml
from src.utils.reproducibility import set_seed

_HERE = Path(__file__).resolve().parent


def _import_captioning_module():
    """Load scripts/08_run_captioning.py as a module to borrow _build_vlm."""
    path = _HERE / "08_run_captioning.py"
    spec = importlib.util.spec_from_file_location("cap08", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _count_params(vlm):
    def n(m):
        return sum(p.numel() for p in m.parameters()) if m is not None else 0

    def n_train(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad) if m is not None else 0

    connector = getattr(vlm, "_connector", None) or getattr(vlm, "connector", None)
    encoder = getattr(vlm, "_encoder", None) or getattr(vlm, "encoder", None)
    llm = getattr(vlm, "_llm", None)

    lora = 0
    if llm is not None:
        lora = sum(p.numel() for name, p in llm.named_parameters() if "lora" in name.lower())

    return {
        "connector": n(connector),
        "vision_encoder": n(encoder),
        "llm": n(llm),
        "lora": lora,
        "total": n(connector) + n(encoder) + n(llm),
        "trainable_at_serving": n_train(connector) + n_train(encoder) + n_train(llm),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/captioning.yaml")
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--projector-config", default="configs/projector.yaml")
    p.add_argument("--llm-config", default="configs/llm.yaml")
    p.add_argument("--stage2-config", default="configs/training_stage2.yaml")
    p.add_argument("--condition", required=True)
    p.add_argument("--vlm-checkpoint", required=True)
    p.add_argument("--n-images", type=int, default=500,
                   help="timed captions, batch=1 (>=500 recommended for a stable p95)")
    p.add_argument("--warmup", type=int, default=10,
                   help="untimed captions first; CUDA autotuning makes early calls unrepresentative")
    p.add_argument("--out-dir", default="outputs/metrics")
    args = p.parse_args()

    cap_cfg = load_yaml(args.config)
    enc_cfg = load_yaml(args.encoders_config)
    proj_cfg = load_yaml(args.projector_config)
    llm_cfg = load_yaml(args.llm_config)
    stage2_cfg = load_yaml(args.stage2_config)
    lora_cfg = stage2_cfg.get("lora") if stage2_cfg.get("lora", {}).get("enabled") else None

    set_seed(cap_cfg["seed"])

    ds = CocoVal2017Dataset(
        annotations_json=cap_cfg["eval_set"]["annotations_json"],
        image_root=cap_cfg["eval_set"]["image_root"],
    )
    items = list(ds.items())[: args.warmup + args.n_images]
    if len(items) < args.warmup + args.n_images:
        raise SystemExit(
            f"[serving] need {args.warmup + args.n_images} images, dataset has {len(items)}"
        )

    cap08 = _import_captioning_module()

    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.memory_allocated()
    t_load = time.perf_counter()
    vlm = cap08._build_vlm(args.vlm_checkpoint, enc_cfg, proj_cfg, llm_cfg, lora_cfg)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t_load

    params = _count_params(vlm)
    ckpt_bytes = Path(args.vlm_checkpoint).stat().st_size if Path(args.vlm_checkpoint).exists() else None

    prompt = cap_cfg["prompt"]["user"]
    gen_kwargs = dict(cap_cfg["generation"])

    # ---- warmup, untimed ----
    for img, _ in items[: args.warmup]:
        _ = vlm.generate([img], prompt, **gen_kwargs)
    torch.cuda.synchronize()

    # ---- timed, batch=1: this is the serving case ----
    lat = []
    t0 = time.perf_counter()
    for img, _ in items[args.warmup:]:
        t = time.perf_counter()
        _ = vlm.generate([img], prompt, **gen_kwargs)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1000.0)   # ms
    wall_s = time.perf_counter() - t0

    lat.sort()
    def pct(q):
        return lat[min(len(lat) - 1, int(q * len(lat)))]

    out = {
        "condition": args.condition,
        "checkpoint": args.vlm_checkpoint,
        "n_timed": len(lat),
        "n_warmup": args.warmup,
        "batch_size": 1,
        "generation": gen_kwargs,
        "params": params,
        "checkpoint_bytes": ckpt_bytes,
        "checkpoint_mib": round(ckpt_bytes / 2**20, 1) if ckpt_bytes else None,
        "model_load_seconds": round(load_s, 2),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "vram_before_load_mib": round(vram_before / 2**20, 1),
        "latency_ms": {
            "p50": round(statistics.median(lat), 1),
            "p95": round(pct(0.95), 1),
            "p99": round(pct(0.99), 1),
            "mean": round(statistics.fmean(lat), 1),
            "min": round(lat[0], 1),
            "max": round(lat[-1], 1),
            "stdev": round(statistics.pstdev(lat), 1),
        },
        "throughput_captions_per_s": round(len(lat) / wall_s, 4),
        "wall_seconds": round(wall_s, 1),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"serving_cost_{args.condition}.json"
    dest.write_text(json.dumps(out, indent=2))

    print(f"[serving] {args.condition}")
    print(f"  params total={params['total']:,}  trainable@serving={params['trainable_at_serving']:,}"
          f"  connector={params['connector']:,}  lora={params['lora']:,}")
    print(f"  checkpoint={out['checkpoint_mib']} MiB   peak VRAM={out['peak_vram_mib']} MiB"
          f"   load={out['model_load_seconds']} s")
    print(f"  latency ms  p50={out['latency_ms']['p50']}  p95={out['latency_ms']['p95']}"
          f"  mean={out['latency_ms']['mean']}")
    print(f"  throughput={out['throughput_captions_per_s']} cap/s over n={len(lat)}")
    print(f"[serving] wrote {dest}")


if __name__ == "__main__":
    main()
