"""Zero-shot / fine-tuned captioning on COCO val2017 (5K images).

Loads the VLM for the chosen condition's checkpoint and writes captions to
``outputs/predictions/captions_<condition>.json``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.captioning.inference import run_captioning
from src.data.coco_val2017_loader import CocoVal2017Dataset
from src.encoders.clip_encoder import build_clip_encoder
from src.models.checkpoint import load_projector
from src.models.projector import build_projector
from src.models.vlm import VLM, VLMConfig
from src.utils.io import load_yaml, snapshot_run_metadata
from src.utils.reproducibility import set_seed


def _build_vlm(vlm_checkpoint: str, enc_cfg, proj_cfg, llm_cfg, lora_cfg) -> VLM:
    encoder = build_clip_encoder(enc_cfg).load()
    if str(vlm_checkpoint).lower() == "random":
        connector = build_projector(proj_cfg["architecture"]).to(enc_cfg["inference"]["device"])
        llm_trainable = {}
    else:
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
    )).load_llm()

    if llm_trainable and lora_cfg:
        from peft import LoraConfig, get_peft_model
        peft_cfg = LoraConfig(
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            target_modules=list(lora_cfg["target_modules"]),
            bias=str(lora_cfg.get("bias", "none")),
            task_type="CAUSAL_LM",
        )
        vlm._llm = get_peft_model(vlm._llm, peft_cfg)
        vlm._llm.load_state_dict(llm_trainable, strict=False)
    return vlm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/captioning.yaml")
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--projector-config", default="configs/projector.yaml")
    p.add_argument("--llm-config", default="configs/llm.yaml")
    p.add_argument("--stage2-config", default="configs/training_stage2.yaml")
    p.add_argument("--condition", required=True, help="condition tag for output naming")
    p.add_argument("--vlm-checkpoint", required=True,
                   help="path to condition's Stage-2 checkpoint (or 'random' for C0)")
    args = p.parse_args()

    cap_cfg = load_yaml(args.config)
    enc_cfg = load_yaml(args.encoders_config)
    proj_cfg = load_yaml(args.projector_config)
    llm_cfg = load_yaml(args.llm_config)
    stage2_cfg = load_yaml(args.stage2_config)
    lora_cfg = stage2_cfg.get("lora") if stage2_cfg.get("lora", {}).get("enabled") else None

    set_seed(cap_cfg["seed"])

    ds = CocoVal2017Dataset(
        annotations_json=cap_cfg["eval_set"]["annotations_json"],
        image_root=cap_cfg["eval_set"]["image_root"],
    )
    items = list(ds.items())
    items = items[: cap_cfg["eval_set"]["num_images"]]

    vlm = _build_vlm(args.vlm_checkpoint, enc_cfg, proj_cfg, llm_cfg, lora_cfg)

    pred_dir = Path(cap_cfg["output"]["predictions_dir"])
    out = run_captioning(
        vlm, items,
        prompt_template=cap_cfg["prompt"]["user"],
        out_path=pred_dir / f"captions_{args.condition}.json",
        batch_size=cap_cfg["batch"]["per_device_batch_size"],
        gen_kwargs=cap_cfg["generation"],
    )
    snapshot_run_metadata(
        {"captioning": cap_cfg, "condition": args.condition,
         "checkpoint": args.vlm_checkpoint, "args": vars(args)},
        pred_dir,
    )
    print(f"[ok] predictions written to {out}")


if __name__ == "__main__":
    main()
