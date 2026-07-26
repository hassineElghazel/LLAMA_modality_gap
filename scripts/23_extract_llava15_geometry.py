"""Extract LLaVA-1.5-7B's OWN modality gap at its connector output.

Measures where a fully-trained model sits on the location axis, to gate the
"does location-closure generalise to a competent model" experiment: if LLaVA's
$G_\mu$ is already small, full training closed location on its own (and the
fine-tune experiment has no headroom); if it is still open, that experiment is
well-motivated.

Clouds (same 1300 dd256 pairs as the diagnostic gap):
  image = mean-pool over the 576 projected image tokens
          (multi_modal_projector output, Vicuna 4096-d)
  text  = Vicuna embed_tokens, mean-pooled over caption content tokens (no BOS/EOS/pad)
Then compute_all_metrics(X, Y) -> gap_llava15.json, IDENTICAL form to the
diagnostic gaps so G_mu/trace/eff_rank/subspace_overlap are directly comparable.

CAVEAT: different space than the diagnostic models (Vicuna vs base LLaMA-2, 576 vs
257 tokens, 336 vs 224 px) -> absolute magnitudes are NOT 1:1. Read location as
open/closed via the SCALE-NORMALISED G_mu = G_mu / sqrt(trace_image) (printed),
which is comparable across spaces.

Usage (GPU node):
    python scripts/23_extract_llava15_geometry.py \
        --data-config configs/data_1300.yaml \
        --model-id llava-hf/llava-1.5-7b-hf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.data.coco_val2017_loader import load_diagnostic_manifest
from src.diagnostics.metrics import compute_all_metrics
from src.utils.io import load_yaml, save_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-config", default="configs/data_1300.yaml")
    p.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--condition", default="llava15")
    p.add_argument("--embeddings-dir", default="outputs/embeddings_1300")
    p.add_argument("--out-dir", default="outputs/metrics")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--load-4bit", dest="load_4bit", action="store_true", default=True)
    p.add_argument("--no-4bit", dest="load_4bit", action="store_false")
    args = p.parse_args()

    data_cfg = load_yaml(args.data_config)
    pairs = load_diagnostic_manifest(data_cfg["diagnostic_sample"]["manifest_path"])
    print(f"[llava15-geo] {len(pairs)} pairs from {data_cfg['diagnostic_sample']['manifest_path']}")

    from transformers import AutoProcessor, LlavaForConditionalGeneration
    load_kwargs: dict = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={"": 0})
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = LlavaForConditionalGeneration.from_pretrained(args.model_id, **load_kwargs).eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
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
        for i in tqdm(range(0, len(pairs), args.batch_size), desc="llava15 geometry"):
            batch = pairs[i:i + args.batch_size]
            images = [Image.open(pr.image_path).convert("RGB") for pr in batch]
            captions = [pr.caption for pr in batch]

            # ---- image side: connector output, pooled over 576 tokens ----
            pv = processor(images=images, return_tensors="pt").pixel_values.to("cuda", torch.float16)
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
    out_path = Path(args.out_dir) / f"gap_{args.condition}.json"
    save_json(d, out_path)

    sm = d["spec_metrics"]
    gmu = sm["G_mu"]; tr = sm["trace_image"]
    mu_img = float(X.mean(0).norm()); mu_txt = float(Y.mean(0).norm())
    print(f"\n[llava15-geo] wrote {out_path}")
    print(f"  ||mu_image||={mu_img:.2f}  ||mu_text||={mu_txt:.2f}  trace_image={tr:.1f}")
    print(f"  G_mu={gmu:.2f}   scale-normalised G_mu (G_mu/sqrt(trace))={gmu/(tr**0.5):.3f}")
    print("  Read: diagnostic models were OPEN at normalised G_mu ~2.7 (C3pinr) and "
          "CLOSED at ~0.45 (Cloc). Compare LLaVA's value to judge open/closed.")


if __name__ == "__main__":
    main()
