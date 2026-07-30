"""Precompute the FROZEN geometry anchors for the LLaVA 3-arm fine-tune.

Two quantities, both estimated on DOCCI-train with the PRETRAINED LLaVA-1.5
(no adapter, no connector edit) so they are fixed targets the fine-tune moves
toward / holds against:

  mu_y   : the text centroid the location term pulls the image centroid to.
           = mean over train captions of the per-caption content-mean of Vicuna
           embed_tokens (BOS/EOS/pad excluded) -- IDENTICAL text-side definition
           to scripts/23. Shape (4096,). Saved to mu_y.pt.

  trace_x: the frozen denominator of L_dist = ||mean_b(z_img)-mu_y||^2 / trace_x.
           z_img = mean over the 576 projected image tokens (connector output).
           trace_x = mean_i || x_i - mean_i(x) ||^2  over train images
           = ((X - X.mean(0))**2).sum(1).mean(). Saved to anchors.json.

Making L_dist dimensionless via a FROZEN trace keeps the location gradient a pure
translation (it cannot game the objective by shrinking the cloud -- that is what
the separate scale/rank pins guard, and why trace_x must NOT be recomputed live).

Usage (GPU node):
    python scripts/25_precompute_llava_anchors.py \
        --manifest data/docci/train_manifest.json \
        --model-id llava-hf/llava-1.5-7b-hf \
        --out-dir outputs/anchors_docci
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.data.docci_loader import load_docci_manifest
from src.utils.io import save_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/docci/train_manifest.json")
    p.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--out-dir", default="outputs/anchors_docci")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-images", type=int, default=None,
                   help="Cap images used for trace_x (default: all train images).")
    p.add_argument("--load-4bit", dest="load_4bit", action="store_true", default=True)
    p.add_argument("--no-4bit", dest="load_4bit", action="store_false")
    args = p.parse_args()

    items = load_docci_manifest(args.manifest)
    print(f"[anchors] {len(items)} DOCCI-train rows from {args.manifest}")

    from transformers import AutoImageProcessor, AutoProcessor, LlavaForConditionalGeneration
    load_kwargs: dict = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={"": 0})
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = LlavaForConditionalGeneration.from_pretrained(args.model_id, **load_kwargs).eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        image_processor = AutoImageProcessor.from_pretrained(args.model_id)
    tok = processor.tokenizer
    embed = model.get_input_embeddings()
    vfl = getattr(model.config, "vision_feature_layer", -2)
    vss = getattr(model.config, "vision_feature_select_strategy", "default")

    def image_features(pixel_values):
        try:
            feats = model.get_image_features(
                pixel_values=pixel_values, vision_feature_layer=vfl,
                vision_feature_select_strategy=vss)
        except TypeError:
            feats = model.get_image_features(pixel_values)
        if isinstance(feats, (list, tuple)):
            return torch.stack([f.mean(dim=0) for f in feats], dim=0)
        return feats.mean(dim=1)                                # (B, 4096) pooled-576

    # ---- mu_y : text centroid over ALL train captions (embed content-mean) ----
    mu_sum = torch.zeros(model.config.text_config.hidden_size, dtype=torch.float64)
    n_txt = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(items), args.batch_size), desc="mu_y (text)"):
            caps = [it.caption for it in items[i:i + args.batch_size]]
            enc = tok(caps, return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to("cuda")
            te = embed(enc["input_ids"])                        # (B, L, 4096)
            att = enc["attention_mask"].bool()
            special = torch.zeros_like(att)
            if tok.bos_token_id is not None:
                special |= enc["input_ids"] == tok.bos_token_id
            if tok.eos_token_id is not None:
                special |= enc["input_ids"] == tok.eos_token_id
            mask = (att & ~special).float().unsqueeze(-1)
            pooled = (te * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B,4096)
            mu_sum += pooled.to(torch.float64).sum(dim=0).cpu()
            n_txt += pooled.shape[0]
    mu_y = (mu_sum / max(n_txt, 1)).float()                     # (4096,)

    # ---- trace_x : full-cloud trace of the pooled-576 connector output ----
    img_items = items if args.max_images is None else items[: args.max_images]
    img_rows = []
    with torch.no_grad():
        for i in tqdm(range(0, len(img_items), args.batch_size), desc="trace_x (image)"):
            batch = img_items[i:i + args.batch_size]
            images = [Image.open(it.image_path).convert("RGB") for it in batch]
            pv = image_processor(images=images, return_tensors="pt").pixel_values.to("cuda", torch.float16)
            img_rows.append(image_features(pv).to(torch.float64).cpu())
    X = torch.cat(img_rows, dim=0)                              # (N, 4096)
    Xc = X - X.mean(dim=0, keepdim=True)
    trace_x = float((Xc ** 2).sum(dim=1).mean())

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(mu_y, out_dir / "mu_y.pt")
    anchors = {
        "trace_x": trace_x,
        "mu_y_norm": float(mu_y.norm()),
        "mu_img_norm": float(X.mean(dim=0).norm()),
        "n_captions": int(n_txt),
        "n_images": int(X.shape[0]),
        "pool": "all576",
        "model_id": args.model_id,
        "manifest": str(args.manifest),
    }
    save_json(anchors, out_dir / "anchors.json")

    print(f"\n[anchors] mu_y -> {out_dir/'mu_y.pt'}  ||mu_y||={anchors['mu_y_norm']:.3f}")
    print(f"[anchors] trace_x={trace_x:.3f}  ||mu_image||={anchors['mu_img_norm']:.3f} "
          f"(n_img={anchors['n_images']}, n_cap={anchors['n_captions']})")
    print(f"[anchors] json -> {out_dir/'anchors.json'}")


if __name__ == "__main__":
    main()
