"""Extract a LLaVA-1.5-7B modality gap at the connector output -- pretrained or per-arm.

Runs the FULL diagnostic metric suite (compute_all_metrics) at the one place image
and text are commensurate, and writes gap_<condition>.json in the SAME schema as
the C-experiment gaps, so G_mu / trace / eff_rank / subspace_overlap line up
directly against the existing table.

Clouds:
  image = mean-pool over the 576 projected image tokens
          (multi_modal_projector output, Vicuna 4096-d)
  text  = Vicuna embed_tokens, mean-pooled over caption content tokens (no BOS/EOS/pad)

TWO MODES
  pretrained : no --connector-path  -> measures stock LLaVA-1.5.
  trained arm: --connector-path outputs/checkpoints/llava_docci_<arm>/connector.pt

WHY NO --adapter-dir. The gap is a pure function of the CONNECTOR. The image cloud
stops at the connector output and the text cloud is embed_tokens; the LoRA adapters
sit on the language model's q/v/o_proj (verified: scripts/26 scopes targets by the
full name "language_model."), which NEITHER path traverses. Attaching the adapter
would change no number here, so it is deliberately not loaded -- and because the
text side is adapter-free, the text cloud Y is identical across arms, which is
exactly what a controlled comparison needs.

DATA
  --manifest  : DOCCI manifest (image_path + gold caption per row)   [preferred]
  --data-config: COCO/dd256 diagnostic manifest                       [original path]

CAVEAT: a different space from the diagnostic models (Vicuna vs base LLaMA-2, 576 vs
257 tokens, 336 vs 224 px) -> absolute magnitudes are NOT 1:1. Read location as
open/closed via the SCALE-NORMALISED G_mu = G_mu / sqrt(trace_image) (printed).
Even that is only indicative here: LLaVA's pooled cloud is extremely anisotropic
(eff_rank ~3.4 vs ~37 for the diagnostic models), so sqrt(trace) concentrates in a
few directions and the normalised value reads high by construction. Compare arms to
EACH OTHER (same space, same anisotropy) rather than across systems.

Usage (GPU node):
    # pretrained reference on DOCCI-test
    python scripts/23_extract_llava15_geometry.py \
        --manifest data/docci/test_manifest.json --condition arm0_docci

    # a trained arm
    python scripts/23_extract_llava15_geometry.py \
        --manifest data/docci/test_manifest.json --condition location_docci \
        --connector-path outputs/checkpoints/llava_docci_location/connector.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.diagnostics.metrics import compute_all_metrics
from src.utils.io import load_yaml, save_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-config", default="configs/data_1300.yaml",
                   help="COCO/dd256 diagnostic manifest config (ignored when --manifest is set).")
    p.add_argument("--manifest", default=None,
                   help="DOCCI manifest json (image_path + caption per row). Preferred.")
    p.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--condition", default="llava15")
    p.add_argument("--connector-path", default=None,
                   help="Trained connector.pt. Omitted -> stock pretrained LLaVA.")
    p.add_argument("--embeddings-dir", default="outputs/embeddings_1300")
    p.add_argument("--out-dir", default="outputs/metrics")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-images", type=int, default=None,
                   help="Cap pairs (smoke test / match a reference n).")
    p.add_argument("--load-4bit", dest="load_4bit", action="store_true", default=True)
    p.add_argument("--no-4bit", dest="load_4bit", action="store_false")
    args = p.parse_args()

    if args.manifest:
        from src.data.docci_loader import load_docci_manifest
        rows = load_docci_manifest(args.manifest)
        pairs = [(r.image_path, r.caption) for r in rows]
        src_desc = args.manifest
    else:
        from src.data.coco_val2017_loader import load_diagnostic_manifest
        data_cfg = load_yaml(args.data_config)
        mpath = data_cfg["diagnostic_sample"]["manifest_path"]
        pairs = [(pr.image_path, pr.caption) for pr in load_diagnostic_manifest(mpath)]
        src_desc = mpath
    if args.max_images:
        pairs = pairs[: args.max_images]
    print(f"[llava-geo] condition={args.condition}  {len(pairs)} pairs from {src_desc}")
    print(f"[llava-geo] connector = {args.connector_path or 'PRETRAINED (stock)'}")

    from transformers import AutoImageProcessor, AutoProcessor, LlavaForConditionalGeneration
    load_kwargs: dict = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={"": 0})
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            # keep the connector out of 4-bit: a trained connector.pt is fp and will not
            # load into a bnb-packed uint8 [n/2, 1] weight (matches scripts/26 and 27).
            llm_int8_skip_modules=["multi_modal_projector"])
    model = LlavaForConditionalGeneration.from_pretrained(args.model_id, **load_kwargs).eval()
    if args.connector_path:
        sd = torch.load(args.connector_path, map_location="cpu")
        model.multi_modal_projector.load_state_dict(sd)
        model.multi_modal_projector.to(torch.float16)
        print(f"[llava-geo] loaded trained connector <- {args.connector_path}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    # Call the image processor directly: this transformers version's LlavaProcessor.__call__
    # dereferences text[0] even when text=None, so processor(images=...) crashes. The image
    # processor has no text branch and yields the identical pixel_values.
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        image_processor = AutoImageProcessor.from_pretrained(args.model_id)
    tok = processor.tokenizer
    embed = model.get_input_embeddings()                       # Vicuna embed_tokens
    vfl = getattr(model.config, "vision_feature_layer", -2)
    vss = getattr(model.config, "vision_feature_select_strategy", "default")

    def image_features(pixel_values):
        """Connector-output image tokens (B, 576, 4096), version-robust."""
        try:
            feats = model.get_image_features(
                pixel_values=pixel_values, vision_feature_layer=vfl,
                vision_feature_select_strategy=vss)
        except TypeError:
            feats = model.get_image_features(pixel_values)
        if isinstance(feats, (list, tuple)):                   # per-image list -> pool each
            return torch.stack([f.mean(dim=0) for f in feats], dim=0)
        return feats.mean(dim=1)                               # (B, 4096)

    img_rows, txt_rows = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(pairs), args.batch_size), desc=f"geo {args.condition}"):
            batch = pairs[i:i + args.batch_size]
            images = [Image.open(ip).convert("RGB") for ip, _ in batch]
            captions = [cap for _, cap in batch]

            # ---- image side: connector output, pooled over 576 tokens ----
            pv = image_processor(images=images, return_tensors="pt").pixel_values.to("cuda", torch.float16)
            img_rows.append(image_features(pv).to(torch.float64).cpu())

            # ---- text side: Vicuna embed_tokens, content-mean (excl BOS/EOS/pad) ----
            enc = tok(captions, return_tensors="pt", padding=True, truncation=True).to("cuda")
            te = embed(enc["input_ids"])                       # (B, L, 4096)
            att = enc["attention_mask"].bool()
            special = torch.zeros_like(att)
            if tok.bos_token_id is not None:
                special |= enc["input_ids"] == tok.bos_token_id
            if tok.eos_token_id is not None:
                special |= enc["input_ids"] == tok.eos_token_id
            mask = (att & ~special).float().unsqueeze(-1)
            pooled = (te * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            txt_rows.append(pooled.to(torch.float64).cpu())

    X = torch.cat(img_rows, dim=0)                             # (N, 4096) image cloud
    Y = torch.cat(txt_rows, dim=0)                             # (N, 4096) text cloud

    emb_dir = Path(args.embeddings_dir); emb_dir.mkdir(parents=True, exist_ok=True)
    torch.save(X, emb_dir / f"projected_{args.condition}_image_pooled.pt")
    torch.save(Y, emb_dir / f"projected_{args.condition}_text_pooled.pt")

    metrics = compute_all_metrics(X, Y)
    d = metrics.to_dict()
    sm = d["spec_metrics"]
    gmu = sm["G_mu"]; tr = sm["trace_image"]
    # provenance so a gap json is self-describing (which weights / which images)
    d["source"] = {
        "condition": args.condition, "n_pairs": int(X.shape[0]),
        "manifest": src_desc, "model_id": args.model_id,
        "connector_path": args.connector_path, "pool": "all576",
        "G_mu_norm": gmu / (tr ** 0.5),
    }
    out_path = Path(args.out_dir) / f"gap_{args.condition}.json"
    save_json(d, out_path)

    mu_img = float(X.mean(0).norm()); mu_txt = float(Y.mean(0).norm())
    print(f"\n[llava-geo] wrote {out_path}")
    print(f"  ||mu_image||={mu_img:.2f}  ||mu_text||={mu_txt:.2f}  trace_image={tr:.1f}")
    print(f"  G_mu={gmu:.2f}   scale-normalised G_mu (G_mu/sqrt(trace))={gmu/(tr**0.5):.3f}")
    q = (d.get("extras") or {}).get("subspace_overlap_q") or {}
    q16 = q.get("16", q.get(16))
    print(f"  eff_rank_image={sm.get('eff_rank_image'):.2f}  trace_text={sm.get('trace_text'):.4f}"
          + (f"  subspace_overlap_q16={q16:.4f}" if isinstance(q16, (int, float)) else ""))
    print("  Read ARM-vs-ARM (same space). Cross-system reference only: diagnostic models "
          "were OPEN at normalised G_mu ~2.7 (C3pinr), CLOSED at ~0.45 (Cloc) -- but their "
          "cloud is eff_rank ~37 vs LLaVA's ~3.4, so that margin is indicative, not exact.")


if __name__ == "__main__":
    main()
