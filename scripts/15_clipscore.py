"""Reference-free CLIPScore for caption predictions (Hessel et al., 2021).

CLIP-S = w * max( cos( CLIP_img(image), CLIP_txt(caption) ), 0 ),  w = 2.5.

Scores ``outputs/predictions/captions_<condition>.json`` against the COCO
val2017 images and writes ``outputs/metrics/clipscore_<condition>.json``.

Reference-free: needs only the images and the predicted captions, so it
measures image<->caption semantic alignment directly, with no n-gram or
caption-length confound (unlike BLEU/CIDEr/METEOR).

Notes
-----
* ``--clip-model`` defaults to ViT-B/32 (the CLIPScore-paper backbone *and*
  a different CLIP from the ViT-L/14 vision tower used to generate the
  captions, which avoids a scoring-circularity objection). Pass
  ``openai/clip-vit-large-patch14`` to reuse the cached tower when offline;
  the dose-response across conditions is valid either way since all
  conditions share the backbone.
* CLIP's text encoder is capped at 77 tokens and trained on short alt-text;
  captions are truncated to that limit. The truncation bias is constant
  across conditions, so cross-condition comparison holds, but small absolute
  differences should not be over-read for verbose captions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.utils.io import load_yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/captioning.yaml")
    p.add_argument("--condition", required=True,
                   help="condition tag matching captions_<condition>.json")
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32",
                   help="HF CLIP id; use openai/clip-vit-large-patch14 if offline")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--w", type=float, default=2.5,
                   help="CLIPScore rescaling weight (Hessel et al. default 2.5)")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    pred_path = Path(cfg["output"]["predictions_dir"]) / f"captions_{args.condition}.json"
    scores_dir = Path(cfg["output"]["scores_dir"])
    scores_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(cfg["eval_set"]["image_root"])

    with pred_path.open() as f:
        preds = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    proc = CLIPProcessor.from_pretrained(args.clip_model)
    max_len = proc.tokenizer.model_max_length

    cosines: list[float] = []
    with torch.no_grad():
        for i in range(0, len(preds), args.batch_size):
            chunk = preds[i:i + args.batch_size]
            images = [
                Image.open(image_root / f"{r['image_id']:012d}.jpg").convert("RGB")
                for r in chunk
            ]
            texts = [r["caption"] for r in chunk]
            inp = proc(
                text=texts, images=images, return_tensors="pt",
                padding=True, truncation=True, max_length=max_len,
            ).to(device)
            img_emb = model.get_image_features(pixel_values=inp["pixel_values"])
            txt_emb = model.get_text_features(
                input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]
            )
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            cosines.extend((img_emb * txt_emb).sum(-1).cpu().tolist())

    cos_t = torch.tensor(cosines)
    clip_s = (args.w * cos_t.clamp(min=0))
    summary = {
        "n_evaluated": len(cosines),
        "clip_model": args.clip_model,
        "w": args.w,
        "CLIPScore": float(clip_s.mean()),
        "CLIPScore_std": float(clip_s.std(unbiased=True)),
        "mean_cosine": float(cos_t.mean()),
    }

    out_path = scores_dir / f"clipscore_{args.condition}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[ok] {args.condition}: CLIPScore={summary['CLIPScore']:.4f} "
        f"(±{summary['CLIPScore_std']:.4f}, n={summary['n_evaluated']}, "
        f"{args.clip_model})"
    )
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
