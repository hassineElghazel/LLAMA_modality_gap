"""Step 2 — causal interventions: which geometric component drives captioning?

The C4 sweep is *correlational*: across lambda, location (G_mu), orientation
(subspace_overlap) and shape (trace / eff_rank) move together, so it cannot say
which one is the functional lever. This probe *manipulates one component at a
time* on a trained model, at inference, with NO retraining, and re-measures
captioning. Whichever intervention moves grounding is the causal component.

Three isolated interventions on the projected image tokens ``z`` (plus an
identity control), built read-only from the chosen condition's saved pooled
embeddings (image cloud mu_x,U_x,lam_x ; text cloud mu_y,U_y,lam_y), restricted
to the top-q principal subspace (full 4096-d covariance is rank-deficient):

    identity   : z' = z                                  (control — must match base)
    mean_shift : z' = z + (mu_y - mu_x)                  (DISTANCE: G_mu -> ~0)
    rotate     : z' = mu_x + U_y (U_x^T z_c) + resid      (ORIENTATION: axes -> text)
    recolor    : z' = mu_x + U_x diag(sqrt(lam_y/lam_x)) U_x^T z_c + resid  (SHAPE)

  z_c = z - mu_x ;  resid = z_c - U_x U_x^T z_c  (off-subspace part, kept fixed)

Each intervention is a single affine map applied identically to every token, so
the pooled geometry the connector emits is exactly that map applied to the saved
pooled image cloud. We therefore VERIFY isolation cheaply on CPU: apply the same
map to the pooled embeddings and re-run ``compute_all_metrics`` — confirming the
intended metric moved and the others held — then measure FUNCTION via CLIPScore
(reference-free, immune to the verbose-vs-terse mismatch that floors CIDEr).

ISOLATION: reads only the chosen checkpoint + that condition's saved pooled
embeddings + COCO; writes ONLY under outputs/{predictions,metrics,figures}/intervene/.

Usage:
    python scripts/16_intervene_geometry.py \
        --vlm-checkpoint outputs/checkpoints/stage2_vlm_C4_lam0p1.pt \
        --emb-condition C4_lam0p1 --subset-size 300
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from src.captioning.inference import run_captioning
from src.data.coco_val2017_loader import CocoVal2017Dataset
from src.diagnostics.metrics import compute_all_metrics
from src.encoders.clip_encoder import build_clip_encoder
from src.models.checkpoint import load_projector
from src.models.projector import build_projector
from src.models.vlm import VLM, VLMConfig
from src.utils.io import load_yaml
from src.utils.reproducibility import set_seed


INTERVENTIONS = ("identity", "mean_shift", "rotate", "recolor")

# Geometric component each intervention is designed to move (for the report).
TARGET_OF = {
    "identity":   "none (control)",
    "mean_shift": "distance (G_mu)",
    "rotate":     "orientation (subspace_overlap)",
    "recolor":    "shape (trace / eff_rank)",
}


# --------------------------------------------------------------------------
# VLM build — replicated from scripts/08_run_captioning.py / 12 (numeric module
# names are not importable). Loads connector + optional LoRA in 4-bit.
# --------------------------------------------------------------------------
def build_vlm(vlm_checkpoint: str, enc_cfg, proj_cfg, llm_cfg, lora_cfg) -> VLM:
    encoder = build_clip_encoder(enc_cfg).load()
    blob = torch.load(vlm_checkpoint, map_location="cpu")
    if "config" in blob:
        connector = load_projector(vlm_checkpoint).to(enc_cfg["inference"]["device"])
        llm_trainable = {}
    else:
        connector = build_projector(proj_cfg["architecture"])
        connector.load_state_dict(blob["connector"])
        connector = connector.to(enc_cfg["inference"]["device"])
        llm_trainable = blob.get("llm_trainable") or {}

    vlm = VLM(encoder, connector, VLMConfig(
        llm_hf_id=llm_cfg["model"]["hf_id"],
        weights_dtype=llm_cfg["dtype"]["weights"],
        device=enc_cfg["inference"]["device"],
        load_in_4bit=True,
    )).load_llm()

    if llm_trainable and lora_cfg:
        from peft import LoraConfig, get_peft_model
        peft_cfg = LoraConfig(
            r=int(lora_cfg["r"]), lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            target_modules=list(lora_cfg["target_modules"]),
            bias=str(lora_cfg.get("bias", "none")), task_type="CAUSAL_LM",
        )
        vlm._llm = get_peft_model(vlm._llm, peft_cfg)
        vlm._llm.load_state_dict(llm_trainable, strict=False)
    return vlm


# --------------------------------------------------------------------------
# Read-only geometry from the chosen condition's saved pooled embeddings.
# --------------------------------------------------------------------------
def build_geometry(emb_dir: Path, condition: str, q: int):
    """Return float32 CPU tensors describing both clouds' top-q geometry."""
    img = torch.load(emb_dir / f"projected_{condition}_image_pooled.pt", map_location="cpu").float()
    txt = torch.load(emb_dir / f"projected_{condition}_text_pooled.pt", map_location="cpu").float()
    n = img.shape[0]

    mu_x = img.mean(dim=0)
    mu_y = txt.mean(dim=0)
    xc = img - mu_x
    yc = txt - mu_y
    _, sx, Vhx = torch.linalg.svd(xc, full_matrices=False)
    _, sy, Vhy = torch.linalg.svd(yc, full_matrices=False)
    q = min(q, Vhx.shape[0], Vhy.shape[0])
    Ux = Vhx[:q].t().contiguous()           # (D, q) image principal axes
    Uy = Vhy[:q].t().contiguous()           # (D, q) text principal axes
    lam_x = (sx[:q] ** 2) / max(n - 1, 1)   # image eigenvalue spectrum (top q)
    lam_y = (sy[:q] ** 2) / max(n - 1, 1)   # text  eigenvalue spectrum (top q)
    return {
        "mu_x": mu_x, "mu_y": mu_y, "Ux": Ux, "Uy": Uy,
        "lam_x": lam_x, "lam_y": lam_y, "q": q,
        "img_pooled": img, "txt_pooled": txt,
    }


def apply_geom(z: torch.Tensor, kind: str, g: dict) -> torch.Tensor:
    """Apply one intervention to a (..., D) tensor. Pure / linear / per-token."""
    if kind == "identity":
        return z
    mu_x, mu_y, Ux, Uy = g["mu_x"], g["mu_y"], g["Ux"], g["Uy"]
    if kind == "mean_shift":
        return z + (mu_y - mu_x)

    zc = z - mu_x
    coords = zc @ Ux                         # (..., q) coords on image axes
    in_sub = coords @ Ux.t()                 # in-image-subspace component
    resid = zc - in_sub                      # off-subspace residual (kept fixed)
    if kind == "rotate":
        return mu_x + (coords @ Uy.t()) + resid
    if kind == "recolor":
        scale = torch.sqrt(g["lam_y"] / g["lam_x"].clamp(min=1e-8))
        return mu_x + ((coords * scale) @ Ux.t()) + resid
    raise ValueError(f"unknown intervention: {kind}")


class GeometryProjector(nn.Module):
    """Wraps a trained projector; applies ``kind`` to its output before splice."""

    def __init__(self, base: nn.Module, kind: str, g_dev: dict):
        super().__init__()
        self.base = base
        self.kind = kind
        self.g = g_dev

    def forward(self, x):
        z = self.base(x)
        dt = z.dtype
        return apply_geom(z.float(), self.kind, self.g).to(dt)


@torch.no_grad()
def clipscores(ids, path_by_id, cap_by_id, clip_id, device, batch_size=64) -> dict:
    """cos(CLIP_image, CLIP_text(caption)) — reference-free grounding (from 11/12)."""
    from transformers import CLIPModel, CLIPProcessor
    from tqdm import tqdm
    from src.data.coco_val2017_loader import load_image

    model = CLIPModel.from_pretrained(clip_id).to(device).eval()
    proc = CLIPProcessor.from_pretrained(clip_id)
    out = {}
    for i in tqdm(range(0, len(ids), batch_size), desc="clipscore"):
        batch = ids[i:i + batch_size]
        imgs = [load_image(path_by_id[b]) for b in batch]
        caps = [(cap_by_id[b].strip() or " ") for b in batch]
        inp = proc(images=imgs, text=caps, return_tensors="pt",
                   padding=True, truncation=True, max_length=77).to(device)
        imf = model.get_image_features(pixel_values=inp["pixel_values"])
        txf = model.get_text_features(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
        imf = imf / imf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        txf = txf / txf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        for b, c in zip(batch, (imf * txf).sum(-1).tolist()):
            out[b] = float(c)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def verify_geometry(kind: str, g: dict) -> dict:
    """Apply the map to the pooled image cloud, re-run compute_all_metrics, and
    return a compact summary proving which component moved and which held."""
    Xp = apply_geom(g["img_pooled"], kind, g)
    m = compute_all_metrics(Xp, g["txt_pooled"]).to_dict()
    spec, extra = m["spec_metrics"], m["extras"]
    return {
        "G_mu": spec["G_mu"],
        "subspace_overlap_q64": extra["subspace_overlap_q"].get("64"),
        "eff_rank_image": spec["eff_rank_image"],
        "trace_image": spec["trace_image"],
        "knn_mixing_rate_k20": spec["knn_mixing_rate_k20"],
        "residual_ratio": extra["residual_ratio"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vlm-checkpoint", required=True,
                   help="trained Stage-2 checkpoint to intervene on (pick the step-1 best model)")
    p.add_argument("--emb-condition", required=True,
                   help="condition tag whose saved pooled embeddings define the geometry "
                        "(e.g. C4_lam0p1) — must match the checkpoint")
    p.add_argument("--emb-dir", default="outputs/embeddings")
    p.add_argument("--config", default="configs/captioning.yaml")
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--projector-config", default="configs/projector.yaml")
    p.add_argument("--llm-config", default="configs/llm.yaml")
    p.add_argument("--stage2-config", default="configs/training_stage2.yaml")
    p.add_argument("--subset-size", type=int, default=300)
    p.add_argument("--subspace-q", type=int, default=64)
    p.add_argument("--interventions", nargs="*", default=list(INTERVENTIONS), choices=INTERVENTIONS)
    p.add_argument("--device", default="auto")
    # Isolated output roots — never the canonical predictions/metrics dirs.
    p.add_argument("--pred-dir", default="outputs/predictions/intervene")
    p.add_argument("--metrics-dir", default="outputs/metrics/intervene")
    p.add_argument("--fig-dir", default="outputs/figures/intervene")
    args = p.parse_args()

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[intervene] device={device}  checkpoint={args.vlm_checkpoint}  emb={args.emb_condition}")

    cap_cfg = load_yaml(args.config)
    enc_cfg = load_yaml(args.encoders_config)
    proj_cfg = load_yaml(args.projector_config)
    llm_cfg = load_yaml(args.llm_config)
    stage2_cfg = load_yaml(args.stage2_config)
    lora_cfg = stage2_cfg.get("lora") if stage2_cfg.get("lora", {}).get("enabled") else None
    set_seed(cap_cfg["seed"])

    pred_dir = Path(args.pred_dir); pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.metrics_dir); metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)

    # Fixed subset — identical images for every intervention.
    ds = CocoVal2017Dataset(annotations_json=cap_cfg["eval_set"]["annotations_json"],
                            image_root=cap_cfg["eval_set"]["image_root"])
    items = list(ds.items())[: args.subset_size]
    path_by_id = {int(it.image_id): it.image_path for it in items}
    print(f"[intervene] subset: {len(items)} images")

    # Read-only geometry (CPU master) + a device copy for the projector wrapper.
    g = build_geometry(Path(args.emb_dir), args.emb_condition, args.subspace_q)
    print(f"[intervene] geometry q={g['q']} from condition={args.emb_condition}")
    g_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in g.items()}

    # Build the VLM once; keep a handle to the base (unwrapped) projector.
    vlm = build_vlm(args.vlm_checkpoint, enc_cfg, proj_cfg, llm_cfg, lora_cfg)
    base_projector = vlm.projector

    results = {}
    for kind in args.interventions:
        print(f"[intervene] === {kind}  (targets {TARGET_OF[kind]}) ===")
        vlm.projector = GeometryProjector(base_projector, kind, g_dev).to(device)

        out_path = pred_dir / f"captions_{args.emb_condition}_{kind}.json"
        run_captioning(
            vlm, items,
            prompt_template=cap_cfg["prompt"]["user"],
            out_path=out_path,
            batch_size=cap_cfg["batch"]["per_device_batch_size"],
            gen_kwargs=cap_cfg["generation"],
        )
        with out_path.open() as f:
            caps = {int(r["image_id"]): r["caption"] for r in json.load(f)}

        ids = [iid for iid in caps if iid in path_by_id]
        cs = clipscores(ids, path_by_id, caps, clip_id=enc_cfg["vision_model"]["hf_id"], device=device)
        cs_mean = float(sum(cs.values()) / max(len(cs), 1))
        geom = verify_geometry(kind, g)               # CPU, cheap
        results[kind] = {"target": TARGET_OF[kind], "clipscore_mean": cs_mean,
                         "n": len(ids), "geometry": geom}
        print(f"[intervene] {kind}: clipscore_mean={cs_mean:.4f} "
              f"G_mu={geom['G_mu']:.3f} O64={geom['subspace_overlap_q64']:.4f} "
              f"eff_rank={geom['eff_rank_image']:.2f} n={len(ids)}")

    # Restore the base projector (leave the in-RAM model clean).
    vlm.projector = base_projector

    out_json = metrics_dir / f"intervene_{args.emb_condition}.json"
    with out_json.open("w") as f:
        json.dump({"checkpoint": args.vlm_checkpoint, "emb_condition": args.emb_condition,
                   "subset_size": len(items), "subspace_q": g["q"],
                   "results": results}, f, indent=2)
    print(f"[ok] wrote {out_json}")

    # Plot: CLIPScore per intervention, with the identity control as a reference.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = [k for k in INTERVENTIONS if k in results]
        cs = [results[k]["clipscore_mean"] for k in order]
        base = results.get("identity", {}).get("clipscore_mean")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(order, cs, color=["#999", "#4a86e8", "#16a766", "#e07798"][: len(order)])
        if base is not None:
            ax.axhline(base, ls="--", c="k", lw=1, label="identity (control)")
            ax.legend()
        ax.set_ylabel("CLIPScore (mean)")
        ax.set_title(f"Captioning vs geometric intervention — {args.emb_condition}")
        for i, v in enumerate(cs):
            ax.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig_path = fig_dir / f"intervene_{args.emb_condition}.png"
        fig.savefig(fig_path, dpi=150)
        print(f"[ok] wrote {fig_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] plot skipped: {type(e).__name__}: {e}")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
