"""Causal intervention probe: does manipulating subspace_energy change grounding?

The per-image linkage found (in C3) that an image token's energy inside the text
cloud's low-rank principal subspace correlates with caption grounding. That is
observational. Here we *manipulate* that energy on the trained C3 model and
re-caption, to test whether the geometry is a causal lever or just a readout.

Mechanism — wrap the (trained) connector so each projected token z is pushed
toward / away from the text subspace before it is spliced into the LLM:

    z'(beta) = P z_c + beta * (z_c - P z_c) + mu_img        z_c = z - mu_img

    beta = 1  -> identity (control, exact)
    beta < 1  -> shrink off-subspace residual  -> MORE subspace_energy
    beta > 1  -> amplify residual              -> LESS subspace_energy

P (the text top-q principal subspace) and mu_img are built read-only from the
saved pooled embeddings. CLIPScore (reference-free image<->caption similarity)
measures grounding, immune to the verbose-vs-terse style mismatch that floors
CIDEr.

ISOLATION: reads only the C3 checkpoint + saved embeddings + COCO; writes ONLY
under outputs/{predictions,metrics,figures}/probe/. Nothing existing is touched.

Usage:
    python scripts/12_intervene_subspace.py            # defaults (N=1000, 6 betas)
    python scripts/12_intervene_subspace.py --subset-size 500 --betas 0 0.5 1 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from src.captioning.inference import run_captioning
from src.data.coco_val2017_loader import CocoVal2017Dataset
from src.encoders.clip_encoder import build_clip_encoder
from src.models.checkpoint import load_projector
from src.models.projector import build_projector
from src.models.vlm import VLM, VLMConfig
from src.utils.io import load_yaml
from src.utils.reproducibility import set_seed


# --------------------------------------------------------------------------
# VLM build — replicated from scripts/08_run_captioning.py::_build_vlm
# (numeric-prefixed module name isn't importable). Loads connector + LoRA.
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
# The intervention: a projector wrapper applying the beta subspace transform.
# --------------------------------------------------------------------------
class SubspaceProjector(nn.Module):
    """Wraps a trained projector; pushes its output toward/away from the text
    subspace by ``beta`` before it is spliced into the LLM. beta=1 is identity."""

    def __init__(self, base: nn.Module, P: torch.Tensor, mu_img: torch.Tensor, beta: float):
        super().__init__()
        self.base = base
        self.register_buffer("P", P)            # (D, q) text principal subspace
        self.register_buffer("mu", mu_img)      # (D,)   image-modality mean
        self.beta = float(beta)

    def forward(self, x):
        z = self.base(x)                        # (B, T, D)
        dt = z.dtype
        zc = z.float() - self.mu                # center
        proj = (zc @ self.P) @ self.P.t()       # in-subspace component
        zt = proj + self.beta * (zc - proj) + self.mu
        return zt.to(dt)


def build_subspace_and_mean(emb_dir: Path, q: int, device: str):
    """Read-only: text top-q principal subspace P and image-modality mean."""
    txt = torch.load(emb_dir / "projected_C3_stage2_text_pooled.pt", map_location="cpu").float()
    img = torch.load(emb_dir / "projected_C3_stage2_image_pooled.pt", map_location="cpu").float()
    mu_img = img.mean(dim=0)
    txt_c = txt - txt.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(txt_c, full_matrices=False)
    q = min(q, Vh.shape[0])
    P = Vh[:q].t().contiguous()                 # (D, q)
    return P.to(device), mu_img.to(device), img


def achieved_energy(img_pooled: torch.Tensor, P: torch.Tensor, mu_img: torch.Tensor,
                    beta: float) -> float:
    """Analytic pooled subspace_energy under the beta transform (sanity check).

    energy(beta) = ||proj||^2 / (||proj||^2 + beta^2 ||resid||^2), mean over images.
    """
    zc = img_pooled.float().to(P.device) - mu_img
    proj = (zc @ P) @ P.t()
    pe = (proj ** 2).sum(dim=1)
    re = ((zc - proj) ** 2).sum(dim=1)
    e = pe / (pe + (beta ** 2) * re).clamp(min=1e-12)
    return float(e.mean())


@torch.no_grad()
def clipscores(ids, path_by_id, cap_by_id, clip_id, device, batch_size=64) -> dict:
    """Replicated from scripts/11. cos(CLIP_image, CLIP_text(caption))."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vlm-checkpoint", default="outputs/checkpoints/stage2_vlm_C3.pt")
    p.add_argument("--emb-dir", default="outputs/embeddings")
    p.add_argument("--config", default="configs/captioning.yaml")
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--projector-config", default="configs/projector.yaml")
    p.add_argument("--llm-config", default="configs/llm.yaml")
    p.add_argument("--stage2-config", default="configs/training_stage2.yaml")
    p.add_argument("--subset-size", type=int, default=1000)
    p.add_argument("--subspace-q", type=int, default=32)
    p.add_argument("--betas", type=float, nargs="*", default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0])
    p.add_argument("--device", default="auto")
    # Isolated output roots — never the canonical predictions/metrics dirs.
    p.add_argument("--pred-dir", default="outputs/predictions/probe")
    p.add_argument("--metrics-dir", default="outputs/metrics/probe")
    p.add_argument("--fig-dir", default="outputs/figures/probe")
    args = p.parse_args()

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={device}")

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

    # Fixed subset, same images for every beta.
    ds = CocoVal2017Dataset(annotations_json=cap_cfg["eval_set"]["annotations_json"],
                            image_root=cap_cfg["eval_set"]["image_root"])
    items = list(ds.items())[: args.subset_size]
    path_by_id = {int(it.image_id): it.image_path for it in items}
    print(f"[probe] subset: {len(items)} images")

    # Read-only geometry.
    P, mu_img, img_pooled = build_subspace_and_mean(Path(args.emb_dir), args.subspace_q, device)

    # Build C3 VLM once; keep a handle to the base (unwrapped) projector.
    vlm = build_vlm(args.vlm_checkpoint, enc_cfg, proj_cfg, llm_cfg, lora_cfg)
    base_projector = vlm.projector

    results = {}
    for beta in args.betas:
        tag = f"beta{beta:.2f}"
        print(f"[probe] === {tag} ===")
        # Fresh wrap from the base each time (never double-wrap).
        vlm.projector = SubspaceProjector(base_projector, P, mu_img, beta).to(device)

        out_path = pred_dir / f"captions_C3_{tag}.json"
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
        e_mean = achieved_energy(img_pooled, P, mu_img, beta)
        results[tag] = {"beta": beta, "clipscore_mean": cs_mean,
                        "achieved_energy_mean": e_mean, "n": len(ids)}
        print(f"[probe] {tag}: clipscore_mean={cs_mean:.4f} achieved_energy={e_mean:.4f} n={len(ids)}")

    # Restore the base projector (leave the in-RAM model clean).
    vlm.projector = base_projector

    out_json = metrics_dir / "clipscore_vs_beta.json"
    with out_json.open("w") as f:
        json.dump({"subset_size": len(items), "subspace_q": args.subspace_q,
                   "checkpoint": args.vlm_checkpoint, "results": results}, f, indent=2)
    print(f"[ok] wrote {out_json}")

    # Plot: CLIPScore vs achieved energy, and vs beta.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = sorted(results.values(), key=lambda r: r["beta"])
        betas = [r["beta"] for r in rows]
        en = [r["achieved_energy_mean"] for r in rows]
        cs = [r["clipscore_mean"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(en, cs, "o-"); ax[0].set_xlabel("achieved subspace_energy (mean)")
        ax[0].set_ylabel("CLIPScore (mean)"); ax[0].set_title("grounding vs achieved energy")
        for b, x, y in zip(betas, en, cs):
            ax[0].annotate(f"β={b}", (x, y), fontsize=8)
        ax[1].plot(betas, cs, "s-"); ax[1].set_xlabel("β (1=identity)")
        ax[1].set_ylabel("CLIPScore (mean)"); ax[1].set_title("grounding vs β")
        fig.tight_layout()
        fig_path = fig_dir / "clipscore_vs_energy.png"
        fig.savefig(fig_path, dpi=150)
        print(f"[ok] wrote {fig_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] plot skipped: {type(e).__name__}: {e}")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
