"""Step 2 — causal interventions: which geometric component drives captioning?

The C4 sweep is *correlational*: across lambda, location (G_mu), orientation
(subspace_overlap) and shape (trace / eff_rank) move together, so it cannot say
which one is the functional lever. This probe *manipulates one component at a
time* on a trained model, at inference, with NO retraining, and re-measures
captioning. Whichever intervention moves grounding is the causal component.

Interventions on the projected image tokens ``z`` (plus controls), built
read-only from the chosen condition's saved pooled embeddings (image cloud
mu_x,U_x,lam_x ; text cloud mu_y,U_y,lam_y), restricted (for the subspace ones)
to the top-q principal subspace (full 4096-d covariance is rank-deficient). Each
is dosed by ``alpha`` in [0,1] (0 = identity, 1 = full), so we trace a
dose-RESPONSE curve rather than a single on/off point (Liang et al. 2022,
"Mind the Gap", move_features):

    identity     : z' = z                                       (control — must match base)
    mean_shift   : z' = z + alpha (mu_y - mu_x)                 (DISTANCE: G_mu -> ~0)
    rotate       : z' = mu_x + [(1-a) in_sub + a (coords U_y^T)] + resid  (ORIENTATION)
    recolor      : z' = mu_x + U_x diag((sqrt(lam_y/lam_x))^a) U_x^T z_c + resid  (SHAPE)
    realign      : z' = (mu_x + a(mu_y-mu_x)) + (sqrt(tr_y/tr_x))^a (z - mu_x)  (FULL close, ReAlign)
    random_shift : z' = z + alpha r,  ||r|| = ||mu_y - mu_x||   (NULL: norm-matched OOD control)

  z_c = z - mu_x ;  coords = z_c U_x ;  in_sub = coords U_x^T ;  resid = z_c - in_sub

``random_shift`` is the crucial control: a directed gap-closing shift and a
random shift of the SAME magnitude are both equally off-distribution for the
frozen decoder, so any decoder effect ABOVE the random null is the part
attributable to the geometry, not to generic OOD fragility.

TWO readouts, because the single (decoder) readout was the weakness of the
earlier probe:
  - DECODER-FREE (immune to the OOD confound): image->text retrieval R@K +
    mean paired cosine on the pooled clouds after the map (Liang-style; no LLM).
  - DECODER (function-level): CLIPScore on the generated captions. Pass a non-
    circular CLIP via --clip-model (e.g. ~/clip_b32) so it is not the same tower
    as the vision encoder.
The CONTRAST between the two is the result: alignment that moves the decoder-free
readout but not CLIPScore localises the gap as decoder-bound, not representational.

Cost-aware grids: the decoder-free readout + geometry verification run on a FINE
alpha grid (cheap, CPU, no generation); captioning + CLIPScore run on a COARSE
alpha grid (expensive). Each map is also VERIFIED on CPU via compute_all_metrics
(intended component moved, others held).

ISOLATION: reads only the chosen checkpoint + that condition's saved pooled
embeddings + COCO; writes ONLY under outputs/{predictions,metrics,figures}/intervene/.

Usage:
    python scripts/16_intervene_geometry.py \
        --vlm-checkpoint outputs/checkpoints/stage2_vlm_C4_lam0p1.pt \
        --emb-condition C4_lam0p1 --subset-size 300 \
        --clip-model "$HOME/clip_b32"
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


INTERVENTIONS = ("identity", "mean_shift", "rotate", "recolor", "realign", "random_shift")

# Geometric component each intervention is designed to move (for the report).
TARGET_OF = {
    "identity":     "none (control)",
    "mean_shift":   "distance (G_mu)",
    "rotate":       "orientation (subspace_overlap)",
    "recolor":      "shape (trace / eff_rank)",
    "realign":      "full close: distance + scale (ReAlign)",
    "random_shift": "none (norm-matched OOD null control)",
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
    trace_x = (xc ** 2).sum() / max(n - 1, 1)   # total covariance trace (image)
    trace_y = (yc ** 2).sum() / max(n - 1, 1)   # total covariance trace (text)

    # Norm-matched random control direction: deterministic (own seed), magnitude
    # exactly ||mu_y - mu_x|| so random_shift and mean_shift are equal-size moves.
    rng = torch.Generator().manual_seed(0)
    r = torch.randn(img.shape[1], generator=rng)
    r = r / r.norm().clamp(min=1e-12) * (mu_y - mu_x).norm()
    return {
        "mu_x": mu_x, "mu_y": mu_y, "Ux": Ux, "Uy": Uy,
        "lam_x": lam_x, "lam_y": lam_y, "q": q,
        "trace_x": trace_x, "trace_y": trace_y, "rand_shift": r,
        "img_pooled": img, "txt_pooled": txt,
    }


def apply_geom(z: torch.Tensor, kind: str, g: dict, alpha: float = 1.0) -> torch.Tensor:
    """Apply one intervention to a (..., D) tensor at dose ``alpha`` (0=identity,
    1=full). Pure / linear / per-token; alpha interpolates from identity so the
    map traces a dose-response curve."""
    if kind == "identity" or alpha == 0.0:
        return z
    mu_x, mu_y = g["mu_x"], g["mu_y"]
    if kind == "mean_shift":
        return z + alpha * (mu_y - mu_x)
    if kind == "random_shift":
        return z + alpha * g["rand_shift"]
    if kind == "realign":
        # ReAlign full close: anchor (mean -> text) + trace (global scale -> text).
        s = torch.sqrt(g["trace_y"] / g["trace_x"].clamp(min=1e-12)) ** alpha
        mean_a = mu_x + alpha * (mu_y - mu_x)
        return mean_a + s * (z - mu_x)

    Ux, Uy = g["Ux"], g["Uy"]
    zc = z - mu_x
    coords = zc @ Ux                         # (..., q) coords on image axes
    in_sub = coords @ Ux.t()                 # in-image-subspace component
    resid = zc - in_sub                      # off-subspace residual (kept fixed)
    if kind == "rotate":
        rotated = (1.0 - alpha) * in_sub + alpha * (coords @ Uy.t())
        return mu_x + rotated + resid
    if kind == "recolor":
        scale = torch.sqrt(g["lam_y"] / g["lam_x"].clamp(min=1e-8)) ** alpha
        return mu_x + ((coords * scale) @ Ux.t()) + resid
    raise ValueError(f"unknown intervention: {kind}")


class GeometryProjector(nn.Module):
    """Wraps a trained projector; applies ``kind`` at dose ``alpha`` to its output
    before splice."""

    def __init__(self, base: nn.Module, kind: str, g_dev: dict, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.kind = kind
        self.g = g_dev
        self.alpha = float(alpha)

    def forward(self, x):
        z = self.base(x)
        dt = z.dtype
        return apply_geom(z.float(), self.kind, self.g, self.alpha).to(dt)


@torch.no_grad()
def inspace_retrieval(X: torch.Tensor, Y: torch.Tensor, ks=(1, 5, 10)) -> dict:
    """DECODER-FREE alignment readout (Liang et al. style): image->text retrieval
    and mean paired cosine on the pooled clouds, AFTER the intervention. Row i of
    X is paired with row i of Y. No LLM involved, so this readout is immune to the
    off-distribution decoder confound that makes the CLIPScore readout ambiguous."""
    Xn = torch.nn.functional.normalize(X.float(), dim=-1)
    Yn = torch.nn.functional.normalize(Y.float(), dim=-1)
    sim = Xn @ Yn.t()                        # (N, N) cosine
    n = sim.shape[0]
    idx = torch.arange(n)
    out = {"paired_cos_mean": float(sim[idx, idx].mean()), "n": n}
    order = sim.argsort(dim=-1, descending=True)
    for k in ks:
        hit = (order[:, : min(k, n)] == idx.unsqueeze(1)).any(dim=1).float().mean()
        out[f"R@{k}"] = float(hit)
    return out


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


def verify_geometry(kind: str, g: dict, alpha: float = 1.0) -> dict:
    """Apply the map (at dose alpha) to the pooled image cloud, re-run
    compute_all_metrics, and return a compact summary proving which component
    moved and which held."""
    Xp = apply_geom(g["img_pooled"], kind, g, alpha)
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
    p.add_argument("--subset-size", type=int, default=300,
                   help="images captioned per (intervention, caption-dose) — the expensive axis")
    p.add_argument("--subspace-q", type=int, default=64)
    p.add_argument("--interventions", nargs="*", default=list(INTERVENTIONS), choices=INTERVENTIONS)
    p.add_argument("--caption-doses", type=float, nargs="*", default=[0.5, 1.0],
                   help="alpha values at which to RUN GENERATION + CLIPScore (expensive, coarse)")
    p.add_argument("--geom-doses", type=float, nargs="*",
                   default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
                   help="alpha values for the decoder-free readout + geometry verify (cheap, fine)")
    p.add_argument("--retrieval-n", type=int, default=0,
                   help="pooled pairs used for the decoder-free retrieval readout (0 = all)")
    p.add_argument("--clip-model", default=None,
                   help="CLIP model for the CLIPScore readout; pass a NON-circular tower "
                        "(e.g. $HOME/clip_b32) so it differs from the ViT-L/14 vision encoder. "
                        "Default falls back to the vision encoder (circular — flagged in output).")
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
    # The big pooled clouds stay on CPU (used only by the cheap readouts).
    g = build_geometry(Path(args.emb_dir), args.emb_condition, args.subspace_q)
    print(f"[intervene] geometry q={g['q']} from condition={args.emb_condition}")
    g_dev = {k: (v.to(device) if (torch.is_tensor(v) and k not in ("img_pooled", "txt_pooled"))
                 else v) for k, v in g.items()}

    # Decoder-free retrieval cloud (optionally subsample the pooled pairs).
    Xret, Yret = g["img_pooled"], g["txt_pooled"]
    if args.retrieval_n and args.retrieval_n < Xret.shape[0]:
        Xret, Yret = Xret[: args.retrieval_n], Yret[: args.retrieval_n]

    clip_model_id = args.clip_model or enc_cfg["vision_model"]["hf_id"]
    clip_circular = args.clip_model is None
    if clip_circular:
        print("[intervene] WARNING: CLIPScore tower == vision encoder (CIRCULAR). "
              "Pass --clip-model $HOME/clip_b32 for a non-circular readout.")

    non_identity = [k for k in args.interventions if k != "identity"]

    def caption_clipscore(kind: str, alpha: float) -> dict:
        """Expensive leg: wrap projector at (kind, alpha), generate, CLIPScore."""
        vlm.projector = GeometryProjector(base_projector, kind, g_dev, alpha).to(device)
        tag = f"{kind}_a{alpha:g}"
        out_path = pred_dir / f"captions_{args.emb_condition}_{tag}.json"
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
        cs = clipscores(ids, path_by_id, caps, clip_id=clip_model_id, device=device)
        vals = list(cs.values())
        mean = float(sum(vals) / max(len(vals), 1))
        std = float((sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5) if len(vals) > 1 else 0.0
        # Per-image scores retained (ids align across conditions — same subset) so a
        # PAIRED test (directed vs random, directed vs identity) is computable post-hoc.
        return {"clipscore_mean": mean, "clipscore_std": std, "n": len(ids),
                "clipscores": {str(k): float(v) for k, v in cs.items()}}

    # Build the VLM once; keep a handle to the base (unwrapped) projector.
    vlm = build_vlm(args.vlm_checkpoint, enc_cfg, proj_cfg, llm_cfg, lora_cfg)
    base_projector = vlm.projector

    # --- identity baseline (both readouts) ---
    print("[intervene] === identity (control baseline) ===")
    identity = {
        "target": TARGET_OF["identity"],
        "inspace": inspace_retrieval(Xret, Yret),
        "geometry": verify_geometry("identity", g),
        **caption_clipscore("identity", 0.0),
    }
    print(f"[intervene] identity: clipscore={identity['clipscore_mean']:.4f} "
          f"R@1={identity['inspace']['R@1']:.4f} paired_cos={identity['inspace']['paired_cos_mean']:.4f}")

    # --- decoder-free + geometry on the FINE dose grid (cheap, no generation) ---
    geom_grid = []
    for kind in non_identity:
        for a in args.geom_doses:
            ins = inspace_retrieval(apply_geom(Xret, kind, g, a), Yret)
            geom = verify_geometry(kind, g, a)
            geom_grid.append({"kind": kind, "target": TARGET_OF[kind], "alpha": a,
                              "inspace": ins, "geometry": geom})
            print(f"[geom] {kind} a={a:g}: R@1={ins['R@1']:.4f} "
                  f"paired_cos={ins['paired_cos_mean']:.4f} G_mu={geom['G_mu']:.3f} "
                  f"O64={geom['subspace_overlap_q64']:.4f}")

    # --- captioning + CLIPScore on the COARSE dose grid (expensive) ---
    caption_grid = []
    for kind in non_identity:
        print(f"[intervene] === {kind}  (targets {TARGET_OF[kind]}) ===")
        for a in args.caption_doses:
            cap = caption_clipscore(kind, a)
            ins = inspace_retrieval(apply_geom(Xret, kind, g, a), Yret)
            geom = verify_geometry(kind, g, a)
            rec = {"kind": kind, "target": TARGET_OF[kind], "alpha": a,
                   "inspace": ins, "geometry": geom, **cap}
            caption_grid.append(rec)
            print(f"[intervene] {kind} a={a:g}: clipscore={cap['clipscore_mean']:.4f}"
                  f"±{cap['clipscore_std']:.4f} R@1={ins['R@1']:.4f} G_mu={geom['G_mu']:.3f} "
                  f"O64={geom['subspace_overlap_q64']:.4f} n={cap['n']}")

    # Restore the base projector (leave the in-RAM model clean).
    vlm.projector = base_projector

    out_json = metrics_dir / f"intervene_{args.emb_condition}.json"
    with out_json.open("w") as f:
        json.dump({"checkpoint": args.vlm_checkpoint, "emb_condition": args.emb_condition,
                   "subset_size": len(items), "subspace_q": g["q"],
                   "clip_model": clip_model_id, "clip_circular": clip_circular,
                   "retrieval_n": Xret.shape[0],
                   "caption_doses": args.caption_doses, "geom_doses": args.geom_doses,
                   "identity": identity, "geom_grid": geom_grid,
                   "caption_grid": caption_grid}, f, indent=2)
    print(f"[ok] wrote {out_json}")

    # Plot: two panels — decoder-free dose-response (fine) vs decoder CLIPScore (coarse).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"mean_shift": "#4a86e8", "rotate": "#16a766", "recolor": "#e07798",
                  "realign": "#e69138", "random_shift": "#999999"}
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))

        # Panel A: decoder-free R@1 vs dose.
        ax[0].axhline(identity["inspace"]["R@1"], ls="--", c="k", lw=1, label="identity")
        for kind in non_identity:
            pts = sorted([r for r in geom_grid if r["kind"] == kind], key=lambda r: r["alpha"])
            if pts:
                ax[0].plot([r["alpha"] for r in pts], [r["inspace"]["R@1"] for r in pts],
                           "o-", color=colors.get(kind, None), label=kind)
        ax[0].set_xlabel("dose alpha"); ax[0].set_ylabel("image->text R@1")
        ax[0].set_title("decoder-FREE alignment (OOD-immune)"); ax[0].legend(fontsize=8)

        # Panel B: decoder CLIPScore vs dose.
        ax[1].axhline(identity["clipscore_mean"], ls="--", c="k", lw=1, label="identity")
        for kind in non_identity:
            pts = sorted([r for r in caption_grid if r["kind"] == kind], key=lambda r: r["alpha"])
            if pts:
                ax[1].plot([r["alpha"] for r in pts], [r["clipscore_mean"] for r in pts],
                           "s-", color=colors.get(kind, None), label=kind)
        ax[1].set_xlabel("dose alpha"); ax[1].set_ylabel("CLIPScore (mean)")
        circ = " [CIRCULAR]" if clip_circular else ""
        ax[1].set_title(f"decoder function{circ}"); ax[1].legend(fontsize=8)

        fig.suptitle(f"Geometric interventions — {args.emb_condition}")
        fig.tight_layout()
        fig_path = fig_dir / f"intervene_{args.emb_condition}.png"
        fig.savefig(fig_path, dpi=150)
        print(f"[ok] wrote {fig_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] plot skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
