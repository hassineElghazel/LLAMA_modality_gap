"""Manipulation check: did the location arm actually MOVE the image centroid?

For a given model (Arm0 pretrained, or a trained arm via --adapter-dir/--connector-path)
this measures the pooled-576 image centroid on DOCCI-train images and reports the
FROZEN-anchor normalised location gap:

    G_mu_norm = || mean_b(z_img) - mu_y || / sqrt(trace_x)

with the SAME frozen mu_y / trace_x the training used (outputs/anchors_docci). This
is the exact quantity L_dist penalises (up to the sqrt), so it is the honest
before/after readout of the intervention:

    location arm  -> G_mu_norm must DROP vs Arm0/pinned (the term did its job)
    pinned  arm   -> G_mu_norm must NOT drop (scale/rank held, location untouched)
    vanilla arm   -> free to drift either way

Also reports batch trace and participation ratio so the scale/rank PINS can be
sanity-checked (pinned/location should hold them near the pretrained values).

Usage (GPU node):
    python scripts/29_manip_check.py --tag arm0
    python scripts/29_manip_check.py --tag location \
        --adapter-dir outputs/checkpoints/llava_docci_location \
        --connector-path outputs/checkpoints/llava_docci_location/connector.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.data.docci_loader import load_docci_manifest
from src.utils.io import save_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True, help="label for the output json")
    p.add_argument("--manifest", default="data/docci/train_manifest.json")
    p.add_argument("--anchors-dir", default="outputs/anchors_docci")
    p.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--adapter-dir", default=None)
    p.add_argument("--connector-path", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-images", type=int, default=None,
                   help="Cap images (default: all DOCCI-train, matching the anchor estimate).")
    p.add_argument("--out-dir", default="outputs/metrics")
    args = p.parse_args()

    mu_y = torch.load(Path(args.anchors_dir) / "mu_y.pt").float().cuda()
    trace_x = float(json.loads((Path(args.anchors_dir) / "anchors.json").read_text())["trace_x"])

    items = load_docci_manifest(args.manifest)
    if args.max_images:
        items = items[: args.max_images]

    from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             llm_int8_skip_modules=["multi_modal_projector"])  # keep connector fp16 so trained connector.pt loads (matches 26/27)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id, quantization_config=bnb, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, device_map={"": 0})
    if args.connector_path:
        sd = torch.load(args.connector_path, map_location="cpu")
        model.multi_modal_projector.load_state_dict(sd)
        model.multi_modal_projector.to(torch.float16)
        print(f"[manip] loaded connector <- {args.connector_path}")
    if args.adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        print(f"[manip] attached adapter <- {args.adapter_dir}")
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_id)
    image_processor = getattr(processor, "image_processor")
    vfl = getattr(model.config, "vision_feature_layer", -2)
    vss = getattr(model.config, "vision_feature_select_strategy", "default")

    def pooled(pixel_values):
        try:
            feats = model.get_image_features(
                pixel_values=pixel_values, vision_feature_layer=vfl,
                vision_feature_select_strategy=vss)
        except TypeError:
            feats = model.get_image_features(pixel_values)
        if isinstance(feats, (list, tuple)):
            return torch.stack([f.mean(dim=0) for f in feats], dim=0)
        return feats.mean(dim=1)

    rows = []
    with torch.no_grad():
        for i in tqdm(range(0, len(items), args.batch_size), desc=f"manip {args.tag}"):
            batch = items[i:i + args.batch_size]
            images = [Image.open(it.image_path).convert("RGB") for it in batch]
            pv = image_processor(images=images, return_tensors="pt").pixel_values.cuda().to(torch.float16)
            rows.append(pooled(pv).float().cpu())
    X = torch.cat(rows, dim=0)                                  # (N, 4096)
    zbar = X.mean(dim=0).cuda()
    g_mu = float((zbar - mu_y).norm())
    g_mu_norm = g_mu / (trace_x ** 0.5)
    Xc = X - X.mean(dim=0, keepdim=True)
    btrace = float((Xc ** 2).sum(dim=1).mean())
    G = (Xc.cuda() @ Xc.cuda().t())
    pr = float(torch.diagonal(G).sum() ** 2 / (G * G).sum().clamp_min(1e-12))

    summary = {
        "tag": args.tag, "n_images": int(X.shape[0]),
        "G_mu": g_mu, "G_mu_norm": g_mu_norm,
        "batch_trace": btrace, "participation_ratio": pr,
        "trace_x": trace_x, "mu_y_norm": float(mu_y.norm()),
        "adapter_dir": args.adapter_dir, "connector_path": args.connector_path,
    }
    out_path = Path(args.out_dir) / f"manip_{args.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, out_path)
    print(f"\n[manip] {args.tag}: G_mu_norm={g_mu_norm:.4f} (G_mu={g_mu:.2f}) "
          f"btrace={btrace:.1f} PR={pr:.2f}")
    print(f"[manip] wrote {out_path}")
    print("[manip] expect: location DROPS G_mu_norm vs arm0/pinned; pinned holds it; "
          "btrace/PR held near arm0 for pinned & location.")


if __name__ == "__main__":
    main()
