"""Extract connector-output (4096-d) embeddings for one experimental condition.

Per Overleaf Table 3 there are 5 measurement points across C0/C1/C2/C3:

    --condition C0_random   : random-init connector (C0 baseline)
    --condition C1_stage2   : after C1 Stage 2 (no Stage 1 ran)
    --condition C2_stage1   : after C2 Stage 1 (no Stage 2)
    --condition C3_stage1   : after C3 Stage 1
    --condition C3_stage2   : after C3 Stage 2

Each writes:
    outputs/embeddings/projected_<condition>_image_pooled.pt
    outputs/embeddings/projected_<condition>_text_pooled.pt
    outputs/embeddings/projected_<condition>_image_tokens.pt   (raw 257-token tensor)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.data.coco_val2017_loader import load_diagnostic_manifest
from src.diagnostics.extract_projected import extract_projected_embeddings, save_projected
from src.encoders.clip_encoder import build_clip_encoder
from src.models.checkpoint import load_projector
from src.models.projector import build_projector
from src.utils.io import load_yaml, snapshot_run_metadata
from src.utils.reproducibility import set_seed


CONDITIONS = {
    "C0_random":  {"connector": "random",                                       "uses_stage2": False},
    "C1_stage2":  {"connector": "outputs/checkpoints/stage2_vlm_C1.pt",         "uses_stage2": True},
    "C2_stage1":  {"connector": "outputs/checkpoints/stage1_connector_C2.pt",   "uses_stage2": False},
    "C3_stage1":  {"connector": "outputs/checkpoints/stage1_connector_C3.pt",   "uses_stage2": False},
    "C3_stage2":  {"connector": "outputs/checkpoints/stage2_vlm_C3.pt",         "uses_stage2": True},
}


def _load_llama_embed(hf_id: str, device: str, dtype_str: str):
    """Materialise just the LLaMA-2 embedding lookup + tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_str)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype)
    weight = model.get_input_embeddings().weight.detach().clone()
    embed = nn.Embedding.from_pretrained(weight, freeze=True).to(device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embed, tokenizer


def _load_connector_for_condition(ckpt: str, proj_cfg: dict, device: str):
    if ckpt == "random":
        return build_projector(proj_cfg["architecture"]).to(device)
    blob = torch.load(ckpt, map_location="cpu")
    # Connector ckpt is either a projector-only save (has "config") or a
    # stage-2 VLM blob (has "connector"). Handle both.
    if "config" in blob:
        return load_projector(ckpt).to(device)
    proj = build_projector(proj_cfg["architecture"])
    proj.load_state_dict(blob["connector"])
    return proj.to(device)


class _EmbedLM(nn.Module):
    """Tiny shim so extract_projected_embeddings can call get_input_embeddings()
    on a plain nn.Embedding loaded outside of an LM."""
    def __init__(self, embed: nn.Module):
        super().__init__()
        self._embed = embed

    def get_input_embeddings(self):
        return self._embed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    p.add_argument("--connector-override", default=None,
                   help="override the condition's default connector checkpoint path")
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

    device = enc_cfg["inference"]["device"]
    encoder = build_clip_encoder(enc_cfg).load()

    ckpt = args.connector_override or CONDITIONS[args.condition]["connector"]
    connector = _load_connector_for_condition(ckpt, proj_cfg, device)

    embed, tokenizer = _load_llama_embed(
        llm_cfg["model"]["hf_id"], device=device, dtype_str=llm_cfg["dtype"]["weights"],
    )
    shim = _EmbedLM(embed)

    blob = extract_projected_embeddings(
        encoder, connector, shim, tokenizer, pairs, batch_size=args.batch_size,
    )
    save_projected(blob, args.out_dir, args.condition)

    snapshot_run_metadata(
        {"condition": args.condition, "connector_ckpt": ckpt, "args": vars(args)},
        Path(args.out_dir),
    )
    print(f"[ok] saved projected embeddings condition={args.condition} to {args.out_dir}")


if __name__ == "__main__":
    main()
