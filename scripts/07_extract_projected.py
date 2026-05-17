"""Extract projected-token-space embeddings (the §2 conceptual extension).

Run three times:
  --checkpoint random                              -> tag=untrained
  --checkpoint outputs/checkpoints/stage1_projector.pt  -> tag=stage1
  --checkpoint outputs/checkpoints/stage2_full.pt       -> tag=stage2

Outputs:
  outputs/embeddings/projected_<tag>_image_pooled.pt
  outputs/embeddings/projected_<tag>_text_pooled.pt
  outputs/embeddings/projected_<tag>_image_tokens.pt   (raw 576-token tensor)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.coco_loader import load_diagnostic_manifest
from src.diagnostics.extract_projected import extract_projected_embeddings, save_projected
from src.encoders.llm2clip_encoder import LLM2CLIPConfig, LLM2CLIPEncoder
from src.models.checkpoint import load_projector
from src.models.projector import build_projector
from src.utils.io import load_yaml, snapshot_run_metadata
from src.utils.reproducibility import set_seed


def _resolve_tag(checkpoint_arg: str) -> str:
    if checkpoint_arg == "random":
        return "untrained"
    if "stage1" in checkpoint_arg:
        return "stage1"
    if "stage2" in checkpoint_arg:
        return "stage2"
    raise ValueError(f"cannot infer tag from checkpoint path: {checkpoint_arg}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="'random' for untrained baseline, or path to projector/VLM checkpoint")
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--projector-config", default="configs/projector.yaml")
    p.add_argument("--llm-config", default="configs/llm.yaml")
    p.add_argument("--data-config", default="configs/data.yaml")
    p.add_argument("--out-dir", default="outputs/embeddings")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    enc_cfg = load_yaml(args.encoders_config)
    proj_cfg = load_yaml(args.projector_config)
    llm_cfg = load_yaml(args.llm_config)
    data_cfg = load_yaml(args.data_config)

    set_seed(data_cfg["diagnostic_sample"]["seed"])
    pairs = load_diagnostic_manifest(data_cfg["diagnostic_sample"]["manifest_path"])

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

    if args.checkpoint == "random":
        projector = build_projector(proj_cfg["architecture"]).to(enc_cfg["inference"]["device"])
    else:
        projector = load_projector(args.checkpoint).to(enc_cfg["inference"]["device"])

    tokenizer = AutoTokenizer.from_pretrained(llm_cfg["model"]["hf_id"])
    llm = AutoModelForCausalLM.from_pretrained(
        llm_cfg["model"]["hf_id"],
        torch_dtype=getattr(__import__("torch"), llm_cfg["dtype"]["weights"]),
    ).to(enc_cfg["inference"]["device"]).eval()

    blob = extract_projected_embeddings(
        encoder, projector, llm, tokenizer, pairs, batch_size=args.batch_size,
    )
    tag = _resolve_tag(args.checkpoint)
    save_projected(blob, args.out_dir, tag)

    snapshot_run_metadata(
        {"checkpoint": args.checkpoint, "tag": tag, "args": vars(args)},
        Path(args.out_dir),
    )
    print(f"[ok] saved projected embeddings tag={tag} to {args.out_dir}")


if __name__ == "__main__":
    main()
