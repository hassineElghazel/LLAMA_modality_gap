"""Extract LLM2CLIP encoder-space embeddings on the diagnostic 10K sample.

Outputs:
  outputs/embeddings/encoder_image_embeds.pt
  outputs/embeddings/encoder_text_embeds.pt
  outputs/diagnostics_manifest.json
  outputs/embeddings/metadata.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.coco_loader import (
    CocoKarpathyDataset,
    save_diagnostic_manifest,
)
from src.diagnostics.extract_embeddings import (
    extract_encoder_embeddings,
    save_embeddings,
)
from src.encoders.llm2clip_encoder import LLM2CLIPConfig, LLM2CLIPEncoder
from src.utils.io import load_yaml, snapshot_run_metadata
from src.utils.reproducibility import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--data-config", default="configs/data.yaml")
    p.add_argument("--out-dir", default="outputs/embeddings")
    args = p.parse_args()

    enc_cfg = load_yaml(args.encoders_config)
    data_cfg = load_yaml(args.data_config)

    set_seed(data_cfg["diagnostic_sample"]["seed"])

    coco = CocoKarpathyDataset(
        karpathy_json=data_cfg["coco"]["karpathy_split_json"],
        image_root=data_cfg["coco"]["image_root"],
    )
    pairs = coco.diagnostic_sample(
        n=data_cfg["diagnostic_sample"]["num_pairs"],
        pool=tuple(data_cfg["diagnostic_sample"]["pool"]),
        seed=data_cfg["diagnostic_sample"]["seed"],
    )
    save_diagnostic_manifest(pairs, data_cfg["diagnostic_sample"]["manifest_path"])

    encoder = LLM2CLIPEncoder(LLM2CLIPConfig(
        vision_hf_id=enc_cfg["vision_model"]["hf_id"],
        image_size=enc_cfg["vision_model"]["image_size"],
        num_visual_tokens=enc_cfg["vision_model"]["num_visual_tokens"],
        expected_vision_hidden_dim=enc_cfg["vision_model"]["expected_vision_hidden_dim"],
        contrastive_dim=enc_cfg["vision_model"]["contrastive_dim"],
        text_hf_id=enc_cfg["text_model"]["hf_id"],
        llm2vec_name_workaround=enc_cfg["text_model"]["llm2vec_name_workaround"],
        text_pooling_mode=enc_cfg["text_model"]["pooling_mode"],
        text_max_length=enc_cfg["text_model"]["max_length"],
        text_doc_max_length=enc_cfg["text_model"]["doc_max_length"],
        image_processor_fallback_hf_id=enc_cfg["image_processor"]["fallback_hf_id"],
        device=enc_cfg["inference"]["device"],
    )).load()

    img, txt = extract_encoder_embeddings(
        encoder, pairs, batch_size=enc_cfg["inference"]["batch_size"]
    )
    save_embeddings(img, txt, args.out_dir, tag="encoder")

    snapshot_run_metadata(
        {"encoders": enc_cfg, "data": data_cfg, "args": vars(args)},
        Path(args.out_dir),
    )
    print(f"[ok] saved encoder embeddings shape={tuple(img.shape)} to {args.out_dir}")


if __name__ == "__main__":
    main()
