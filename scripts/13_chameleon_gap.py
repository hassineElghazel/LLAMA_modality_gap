"""Modality-gap trajectory of an EARLY-FUSION model (Chameleon-7B).

Contrast experiment for the thesis. The LLaVA pipeline (CLIP -> connector ->
LLaMA) is late fusion: the gap is a single value at the connector output. Chameleon
shares one embedding table + one 32-layer transformer across image and text tokens,
so the gap is a TRAJECTORY across 33 hidden states (embeddings + 32 layers).

For each measurement mode (independent / fused) we mean-pool the image-token and
text-token hidden states at every layer, then run the EXISTING gap-metric suite
(`src/diagnostics/metrics.compute_all_metrics`) at each layer. The late-fusion C3
value is overlaid on each trajectory panel for the early-vs-late contrast.

RESUMABLE (the cluster preempts / time-limits mid-run, so every expensive step is
checkpointed and skipped on resubmit):
  1. Extraction -> a per-mode pooled-tensor checkpoint
     (`outputs/embeddings/chameleon/chameleon_<mode>_pooled.pt`). If present, the
     ~20-min forward pass is skipped and the tensors are loaded from disk.
  2. Metrics -> the trajectory JSON is written ATOMICALLY after EACH layer, with a
     `complete` flag. On resubmit, already-computed layers are skipped, so an
     interruption costs at most the one layer in flight.
A mode whose trajectory JSON already has all 33 layers is skipped entirely.

ISOLATION: reads only Chameleon (HF cache), COCO, the diagnostic manifest, and the
C3 metrics file (overlay, read-only). Writes ONLY under outputs/*/chameleon/.

Usage:
    # local smoke (CPU/MPS), verify token splitting only, no metrics:
    python scripts/13_chameleon_gap.py --subset-size 2 --batch-size 1 \
        --device cpu --no-4bit --skip-metrics
    # full run (both modes), on GPU:
    python scripts/13_chameleon_gap.py
    python scripts/13_chameleon_gap.py --modes independent --subset-size 1000
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from src.data.coco_val2017_loader import CocoVal2017Dataset, load_diagnostic_manifest
from src.diagnostics.chameleon_extract import extract_trajectory
from src.diagnostics.metrics import compute_all_metrics
from src.utils.io import load_yaml, snapshot_run_metadata

PLOT_METRICS = [
    ("G_mu", "spec", "centroid gap G_mu"),
    ("knn_mixing_rate_k20", "spec", "kNN mixing rate (k=20)"),
    ("subspace_overlap_q64", "derived", "subspace overlap (q=64)"),
    ("js_divergence_angular", "spec", "JS angular divergence"),
    ("residual_ratio", "extras", "residual ratio"),
    ("eff_rank_image", "spec", "effective rank (image)"),
]


def _build_model(hf_id: str, load_in_4bit: bool, device: str):
    from transformers import ChameleonForConditionalGeneration, ChameleonProcessor

    processor = ChameleonProcessor.from_pretrained(hf_id)
    kwargs = {"dtype": torch.bfloat16}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    model = ChameleonForConditionalGeneration.from_pretrained(hf_id, **kwargs)
    if not load_in_4bit:
        model = model.to(device)
    model.eval()
    return model, processor


def _load_pairs(cfg: dict, subset_size: int):
    es = cfg["eval_set"]
    manifest = Path(es["manifest_path"])
    if manifest.exists():
        pairs = load_diagnostic_manifest(manifest)
    else:
        ds = CocoVal2017Dataset(es["annotations_json"], es["image_root"])
        pairs = ds.diagnostic_sample(n=None, caption_per_image=1, seed=es.get("seed", 42))
    return pairs[:subset_size]


def _metric_value(layer_dict: dict, key: str, kind: str):
    if kind == "spec":
        return layer_dict["spec_metrics"].get(key)
    if kind == "extras":
        return layer_dict["extras"].get(key)
    if kind == "derived" and key == "subspace_overlap_q64":
        return layer_dict["extras"]["subspace_overlap_q"].get("64")
    return None


def _overlay_value(ref: dict | None, key: str, kind: str):
    if ref is None:
        return None
    return _metric_value(ref, key, kind)


def _atomic_write_json(obj: dict, path: Path) -> None:
    """Write JSON via temp file + os.replace so a kill mid-write cannot corrupt the
    checkpoint (os.replace is atomic on the same filesystem)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def _pooled_ckpt_path(emb_dir: Path, mode: str) -> Path:
    return emb_dir / f"chameleon_{mode}_pooled.pt"


def _save_pooled(per_layer: dict, path: Path) -> None:
    """Persist per-layer pooled tensors (float64 CPU) so extraction runs only once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {str(l): {"image_pooled": b["image_pooled"], "text_pooled": b["text_pooled"]}
           for l, b in per_layer.items()}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _load_pooled(path: Path) -> dict:
    raw = torch.load(path, map_location="cpu")
    return {int(l): {"image_pooled": v["image_pooled"], "text_pooled": v["text_pooled"]}
            for l, v in raw.items()}


def _read_trajectory(path: Path) -> dict | None:
    """Read an existing trajectory JSON, tolerating a corrupt/partial file."""
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {path.name} unreadable ({type(e).__name__}: {e}); recomputing", flush=True)
        return None


def _plot(trajectory: dict[str, dict], mode: str, overlay: dict | None, fig_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[warn] plot skipped: {type(e).__name__}: {e}", flush=True)
        return

    layers = sorted(int(k) for k in trajectory)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    for ax, (key, kind, label) in zip(axes.flat, PLOT_METRICS):
        ys = [_metric_value(trajectory[str(L)], key, kind) for L in layers]
        ax.plot(layers, ys, marker="o", ms=3, lw=1.5, label="Chameleon (early fusion)")
        ov = _overlay_value(overlay, key, kind)
        if ov is not None:
            ax.axhline(ov, ls="--", c="crimson", lw=1.2,
                       label="LLaVA C3 (late fusion)")
        ax.set_xlabel("layer (0 = embedding output)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Chameleon modality-gap trajectory — {mode} mode", y=1.0, fontsize=14)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"gap_trajectory_{mode}.{ext}", dpi=150, bbox_inches="tight")
    print(f"[ok] wrote {fig_dir / f'gap_trajectory_{mode}.png'}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/chameleon.yaml")
    p.add_argument("--modes", nargs="*", default=None,
                   help="override config modes (independent / fused)")
    p.add_argument("--subset-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="auto", help="auto / cuda / cpu / mps")
    p.add_argument("--no-4bit", action="store_true", help="load in bf16 (no bitsandbytes)")
    p.add_argument("--skip-metrics", action="store_true",
                   help="extraction only (smoke test; checkpoints pooled tensors, no metrics)")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute a mode from scratch, ignoring all checkpoints")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    modes = args.modes or cfg["run"]["modes"]
    subset_size = args.subset_size or cfg["run"]["subset_size"]
    batch_size = args.batch_size or cfg["run"]["batch_size"]
    load_in_4bit = cfg["model"]["load_in_4bit"] and not args.no_4bit

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"[device] {device}  4bit={load_in_4bit}  modes={modes}  N={subset_size}", flush=True)

    metrics_dir = Path(cfg["output"]["metrics_dir"])
    figures_dir = Path(cfg["output"]["figures_dir"])
    emb_dir = Path(cfg["output"]["embeddings_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model, processor = _build_model(cfg["model"]["hf_id"], load_in_4bit, device)
    assert model.config.hidden_size == cfg["model"]["hidden_size"], "hidden_size mismatch"
    expected_image_tokens = getattr(processor, "image_seq_length", None)
    expected_layers = model.config.num_hidden_layers + 1   # embeddings + each layer
    print(f"[model] layers={model.config.num_hidden_layers} "
          f"image_token_id={model.config.image_token_id} "
          f"expected_image_tokens={expected_image_tokens} "
          f"expected_trajectory_layers={expected_layers}", flush=True)

    pairs = _load_pairs(cfg, subset_size)
    print(f"[data] {len(pairs)} (image, caption) pairs", flush=True)

    overlay = None
    ref_path = Path(cfg["overlay_reference"])
    if ref_path.exists():
        with ref_path.open() as f:
            overlay = json.load(f)

    q_ladder = tuple(cfg["run"]["q_ladder"])
    knn_k = cfg["run"]["knn_k"]
    hf_id = cfg["model"]["hf_id"]

    for mode in modes:
        print(f"\n=== mode={mode} ===", flush=True)
        traj_path = metrics_dir / f"gap_chameleon_{mode}_trajectory.json"
        existing = None if args.overwrite else _read_trajectory(traj_path)
        done_layers = set((existing or {}).get("trajectory", {}).keys())

        # --- already complete? skip (backfill the `complete` flag / plot if needed) ---
        if existing and len(done_layers) >= expected_layers:
            traj = existing["trajectory"]
            if not existing.get("complete"):
                existing["complete"] = True
                _atomic_write_json(existing, traj_path)
            print(f"[skip] {mode}: all {len(done_layers)} layers present (complete)", flush=True)
            if not args.skip_metrics and not (figures_dir / f"gap_trajectory_{mode}.png").exists():
                _plot(traj, mode, overlay, figures_dir)
            continue

        # --- extraction (skip if a pooled checkpoint exists) ---
        pooled_path = _pooled_ckpt_path(emb_dir, mode)
        if pooled_path.exists() and not args.overwrite:
            print(f"[{mode}] loading pooled checkpoint -> {pooled_path}", flush=True)
            per_layer = _load_pooled(pooled_path)
        else:
            t0 = time.time()
            n_batches = (len(pairs) + batch_size - 1) // batch_size
            print(f"[{mode}] extracting: {len(pairs)} pairs, batch_size={batch_size} "
                  f"(~{n_batches} batches) ...", flush=True)
            per_layer = extract_trajectory(
                model, processor, pairs, mode,
                batch_size=batch_size, expected_image_tokens=expected_image_tokens,
            )
            _save_pooled(per_layer, pooled_path)
            print(f"[{mode}] extraction done in {(time.time() - t0) / 60:.1f} min "
                  f"-> pooled checkpoint saved ({len(per_layer)} layers, "
                  f"{tuple(per_layer[0]['image_pooled'].shape)})  [safe from here]", flush=True)

        if args.skip_metrics:
            print(f"[{mode}] --skip-metrics: extraction checkpointed, no metrics computed", flush=True)
            continue

        # --- metrics (atomic per-layer checkpoint + resume) ---
        trajectory = {} if args.overwrite else dict((existing or {}).get("trajectory", {}))
        layers = sorted(per_layer)
        if trajectory:
            print(f"[{mode}] resuming metrics: {len(trajectory)}/{len(layers)} layers already done",
                  flush=True)
        for layer in layers:
            if str(layer) in trajectory:
                continue
            t0 = time.time()
            blob = per_layer[layer]
            m = compute_all_metrics(blob["image_pooled"], blob["text_pooled"],
                                    q_ladder=q_ladder, knn_k=knn_k)
            d = m.to_dict()
            trajectory[str(layer)] = d
            _atomic_write_json(
                {"mode": mode, "n": len(pairs), "hf_id": hf_id,
                 "complete": False, "trajectory": trajectory},
                traj_path,
            )
            spec = d["spec_metrics"]
            print(f"  [{mode}] layer {layer:2d}/{layers[-1]}: "
                  f"G_mu={spec['G_mu']:.3f} knn={spec['knn_mixing_rate_k20']:.3f} "
                  f"JS={spec['js_divergence_angular']:.3f}  "
                  f"({time.time() - t0:.1f}s) [checkpointed {len(trajectory)}/{len(layers)}]",
                  flush=True)

        _atomic_write_json(
            {"mode": mode, "n": len(pairs), "hf_id": hf_id,
             "complete": True, "trajectory": trajectory},
            traj_path,
        )
        print(f"[{mode}] COMPLETE -> {traj_path} ({len(trajectory)} layers)", flush=True)
        _plot(trajectory, mode, overlay, figures_dir)

    snapshot_run_metadata(cfg, metrics_dir, config_files={"chameleon": args.config},
                          extra_files={"manifest": cfg["eval_set"]["manifest_path"]})
    print("\n[done] chameleon gap trajectory complete", flush=True)


if __name__ == "__main__":
    main()
