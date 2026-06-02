"""Modality-gap trajectory of an EARLY-FUSION model (Chameleon-7B).

Contrast experiment for the thesis. The LLaVA pipeline (CLIP -> connector ->
LLaMA) is late fusion: the gap is a single value at the connector output. Chameleon
shares one embedding table + one 32-layer transformer across image and text tokens,
so the gap is a TRAJECTORY across 33 hidden states (embeddings + 32 layers).

For each measurement mode (independent / fused) we mean-pool the image-token and
text-token hidden states at every layer, then run the EXISTING gap-metric suite
(`src/diagnostics/metrics.compute_all_metrics`) at each layer. The late-fusion C3
value is overlaid on each trajectory panel for the early-vs-late contrast.

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
from pathlib import Path

import torch

from src.data.coco_val2017_loader import CocoVal2017Dataset, load_diagnostic_manifest
from src.diagnostics.chameleon_extract import extract_trajectory
from src.diagnostics.metrics import compute_all_metrics
from src.utils.io import load_yaml, save_json, snapshot_run_metadata

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


def _plot(trajectory: dict[str, dict], mode: str, overlay: dict | None, fig_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[warn] plot skipped: {type(e).__name__}: {e}")
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
    print(f"[ok] wrote {fig_dir / f'gap_trajectory_{mode}.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/chameleon.yaml")
    p.add_argument("--modes", nargs="*", default=None,
                   help="override config modes (independent / fused)")
    p.add_argument("--subset-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="auto", help="auto / cuda / cpu / mps")
    p.add_argument("--no-4bit", action="store_true", help="load in bf16 (no bitsandbytes)")
    p.add_argument("--save-pooled", action="store_true",
                   help="also save per-layer pooled tensors (float32) under embeddings/chameleon")
    p.add_argument("--skip-metrics", action="store_true",
                   help="extraction only (smoke test; skips compute_all_metrics + plot)")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute a mode even if its trajectory JSON exists")
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
    print(f"[device] {device}  4bit={load_in_4bit}  modes={modes}  N={subset_size}")

    metrics_dir = Path(cfg["output"]["metrics_dir"])
    figures_dir = Path(cfg["output"]["figures_dir"])
    emb_dir = Path(cfg["output"]["embeddings_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model, processor = _build_model(cfg["model"]["hf_id"], load_in_4bit, device)
    assert model.config.hidden_size == cfg["model"]["hidden_size"], "hidden_size mismatch"
    expected_image_tokens = getattr(processor, "image_seq_length", None)
    print(f"[model] layers={model.config.num_hidden_layers} "
          f"image_token_id={model.config.image_token_id} "
          f"expected_image_tokens={expected_image_tokens}")

    pairs = _load_pairs(cfg, subset_size)
    print(f"[data] {len(pairs)} (image, caption) pairs")

    overlay = None
    ref_path = Path(cfg["overlay_reference"])
    if ref_path.exists():
        with ref_path.open() as f:
            overlay = json.load(f)

    for mode in modes:
        traj_path = metrics_dir / f"gap_chameleon_{mode}_trajectory.json"
        if traj_path.exists() and not args.overwrite:
            print(f"[skip] {mode}: {traj_path} exists (use --overwrite to redo)")
            continue

        per_layer = extract_trajectory(
            model, processor, pairs, mode,
            batch_size=batch_size, expected_image_tokens=expected_image_tokens,
        )
        n_layers = len(per_layer)
        print(f"[{mode}] extracted {n_layers} layers, "
              f"shape {tuple(per_layer[0]['image_pooled'].shape)}")

        if args.save_pooled:
            emb_dir.mkdir(parents=True, exist_ok=True)
            for layer, blob in per_layer.items():
                torch.save(blob["image_pooled"].to(torch.float32),
                           emb_dir / f"chameleon_{mode}_layer{layer:02d}_image_pooled.pt")
                torch.save(blob["text_pooled"].to(torch.float32),
                           emb_dir / f"chameleon_{mode}_layer{layer:02d}_text_pooled.pt")

        if args.skip_metrics:
            print(f"[{mode}] --skip-metrics: extraction verified, no metrics computed")
            continue

        trajectory = {}
        for layer in sorted(per_layer):
            blob = per_layer[layer]
            m = compute_all_metrics(blob["image_pooled"], blob["text_pooled"],
                                    q_ladder=tuple(cfg["run"]["q_ladder"]),
                                    knn_k=cfg["run"]["knn_k"])
            trajectory[str(layer)] = m.to_dict()
            spec = m.to_dict()["spec_metrics"]
            print(f"  layer {layer:2d}: G_mu={spec['G_mu']:.3f} "
                  f"knn={spec['knn_mixing_rate_k20']:.3f} "
                  f"JS={spec['js_divergence_angular']:.3f}")

        save_json({"mode": mode, "n": len(pairs), "hf_id": cfg["model"]["hf_id"],
                   "trajectory": trajectory}, traj_path)
        print(f"[ok] wrote {traj_path}")
        _plot(trajectory, mode, overlay, figures_dir)

    snapshot_run_metadata(cfg, metrics_dir, config_files={"chameleon": args.config},
                          extra_files={"manifest": cfg["eval_set"]["manifest_path"]})
    print("[done] chameleon gap trajectory complete")


if __name__ == "__main__":
    main()
