"""Zero-shot captioning on COCO Karpathy test 5K with the Stage-2 VLM."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.captioning.inference import run_captioning
from src.data.coco_loader import CocoKarpathyDataset
from src.encoders.llm2clip_encoder import LLM2CLIPConfig, LLM2CLIPEncoder
from src.models.checkpoint import load_projector
from src.models.vlm import VLM, VLMConfig
from src.utils.io import load_yaml, snapshot_run_metadata
from src.utils.reproducibility import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/captioning.yaml")
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--llm-config", default="configs/llm.yaml")
    p.add_argument("--data-config", default="configs/data.yaml")
    args = p.parse_args()

    cap_cfg = load_yaml(args.config)
    enc_cfg = load_yaml(args.encoders_config)
    llm_cfg = load_yaml(args.llm_config)
    data_cfg = load_yaml(args.data_config)

    set_seed(cap_cfg["seed"])

    coco = CocoKarpathyDataset(
        karpathy_json=data_cfg["coco"]["karpathy_split_json"],
        image_root=data_cfg["coco"]["image_root"],
    )
    items = list(coco.items(("test",)))
    items = items[: cap_cfg["eval_set"]["num_images"]]

    encoder = LLM2CLIPEncoder(LLM2CLIPConfig(
        vision_hf_id=enc_cfg["vision_model"]["hf_id"],
        text_hf_id=enc_cfg["text_model"]["hf_id"],
        llm2vec_name_workaround=enc_cfg["text_model"]["llm2vec_name_workaround"],
        image_processor_fallback_hf_id=enc_cfg["image_processor"]["fallback_hf_id"],
        device=enc_cfg["inference"]["device"],
    )).load()

    # Load projector + VLM. For full Stage-2 checkpoint this should restore the
    # whole VLM state_dict; the projector-only loader is shown for clarity.
    projector = load_projector(cap_cfg["checkpoint"]["vlm_checkpoint"])
    vlm = VLM(encoder, projector, VLMConfig(
        llm_hf_id=llm_cfg["model"]["hf_id"],
        weights_dtype=llm_cfg["dtype"]["weights"],
        device=enc_cfg["inference"]["device"],
    )).load_llm()

    out = run_captioning(
        vlm, items,
        prompt_template=cap_cfg["prompt"]["user"],
        out_path=cap_cfg["output"]["predictions_path"],
        batch_size=cap_cfg["batch"]["per_device_batch_size"],
        gen_kwargs=cap_cfg["generation"],
    )
    snapshot_run_metadata(
        {"captioning": cap_cfg, "args": vars(args)},
        Path(out).parent,
    )
    print(f"[ok] predictions written to {out}")


if __name__ == "__main__":
    main()
