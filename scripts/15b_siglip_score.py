"""Reference-free SigLIP alignment score for caption predictions.

An INDEPENDENT, non-CLIP referee for the 3-arm study. SigLIP (Zhai et al., 2023)
is a sigmoid-loss image-text model trained on WebLI -- a different objective,
tokenizer, and data distribution from CLIP. Reporting it alongside CLIPScore
rebuts the "you only moved captions toward CLIP space" objection: the location
term pulls the image centroid to the frozen Vicuna text centroid mu_y, NOT toward
any CLIP/SigLIP embedding, so a gain that shows up on BOTH judges is alignment,
not judge-specific gaming.

Mirrors scripts/15_clipscore.py exactly on I/O: reads
``outputs/predictions/captions_<condition>.json``, resolves images via
``image_root / f"{id:012d}.jpg"`` (so it serves dd256 AND DOCCI-test unchanged by
pointing --config at the right eval yaml), and writes
``outputs/metrics/siglip_<condition>.json``.

Score = mean cosine of L2-normalised SigLIP image/text embeddings (primary,
directly comparable across conditions). We also log SigLIP's own sigmoid
probability (using the model's logit_scale/logit_bias) as a secondary read.

Note: SigLIP was trained with text padded to 64 tokens; we pad/truncate to 64
accordingly. Like CLIP's 77-token cap this truncates very long captions, but the
bias is constant across conditions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.utils.io import load_yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/description_eval.yaml")
    p.add_argument("--condition", required=True,
                   help="condition tag matching captions_<condition>.json")
    p.add_argument("--siglip-model", default="google/siglip-base-patch16-224",
                   help="HF SigLIP id (independent of the CLIP scorer)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=64, help="SigLIP text length (trained @64)")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    pred_path = Path(cfg["output"]["predictions_dir"]) / f"captions_{args.condition}.json"
    scores_dir = Path(cfg["output"]["scores_dir"])
    scores_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(cfg["eval_set"]["image_root"])

    with pred_path.open() as f:
        preds = json.load(f)

    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.siglip_model).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.siglip_model)
    logit_scale = getattr(model, "logit_scale", None)
    logit_bias = getattr(model, "logit_bias", None)

    cosines: list[float] = []
    probs: list[float] = []
    with torch.no_grad():
        for i in range(0, len(preds), args.batch_size):
            chunk = preds[i:i + args.batch_size]
            images = [
                Image.open(image_root / f"{r['image_id']:012d}.jpg").convert("RGB")
                for r in chunk
            ]
            texts = [r["caption"] for r in chunk]
            inp = proc(text=texts, images=images, return_tensors="pt",
                       padding="max_length", truncation=True, max_length=args.max_length).to(device)
            img_emb = model.get_image_features(pixel_values=inp["pixel_values"])
            txt_emb = model.get_text_features(input_ids=inp["input_ids"])
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            cos = (img_emb * txt_emb).sum(-1)
            cosines.extend(cos.cpu().tolist())
            if logit_scale is not None and logit_bias is not None:
                logits = cos * logit_scale.exp() + logit_bias
                probs.extend(torch.sigmoid(logits).cpu().tolist())

    cos_t = torch.tensor(cosines)
    summary = {
        "n_evaluated": len(cosines),
        "siglip_model": args.siglip_model,
        "SigLIPScore": float(cos_t.mean()),            # primary: mean cosine
        "SigLIPScore_std": float(cos_t.std(unbiased=True)),
        "mean_sigmoid_prob": float(torch.tensor(probs).mean()) if probs else None,
    }

    out_path = scores_dir / f"siglip_{args.condition}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    prob_str = f", prob={summary['mean_sigmoid_prob']:.4f}" if probs else ""
    print(f"[ok] {args.condition}: SigLIPScore={summary['SigLIPScore']:.4f} "
          f"(±{summary['SigLIPScore_std']:.4f}, n={summary['n_evaluated']}{prob_str}, "
          f"{args.siglip_model})")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
