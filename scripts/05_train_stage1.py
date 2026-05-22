"""Stage 1: contrastive connector pre-training (InfoNCE).

Trains the connector with symmetric InfoNCE on Bunny-v1.1 image-caption pairs.
ViT (CLIP ViT-L/14) and the LLaMA-2 embedding layer are frozen. Only the
connector parameters + the learnable log-temperature receive gradients.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import torch
from torch import nn

from src.data.bunny_v1_1_loader import BunnyV11Dataset, load_image
from src.encoders.clip_encoder import build_clip_encoder
from src.models.projector import build_projector
from src.training.stage1_pretrain import train_stage1
from src.utils.io import load_yaml, snapshot_run_metadata
from src.utils.reproducibility import set_seed


def _load_llama_embed(hf_id: str, device: str, dtype_str: str) -> tuple[nn.Module, "AutoTokenizer"]:
    """Extract just the LLaMA-2 embedding layer and tokenizer.

    The full LLM is loaded onto CPU RAM (not GPU VRAM) so that GPUs with
    <14 GB (e.g. RTX 2080 Ti at 11 GB) are not affected. We extract the
    32000×4096 embedding weight (~250 MB fp16) then immediately discard the
    rest of the model. The extracted embedding is moved to the target device.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_str)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Load entirely on CPU — uses ~14 GB system RAM, not GPU VRAM.
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, device_map="cpu")
    embed = model.get_input_embeddings()
    weight = embed.weight.detach().clone()
    new_embed = nn.Embedding.from_pretrained(weight, freeze=True).to(device)
    del model, embed, weight
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return new_embed, tokenizer


def _iter_batches(dataset: BunnyV11Dataset, batch_size: int) -> Iterator[dict]:
    """Lazily collate Bunny pairs into ``{"images": list, "captions": list}``."""
    buf_imgs, buf_caps = [], []
    for pair in dataset:
        try:
            img = load_image(pair.image_path)
        except FileNotFoundError:
            continue
        buf_imgs.append(img)
        buf_caps.append(pair.caption)
        if len(buf_imgs) == batch_size:
            yield {"images": buf_imgs, "captions": buf_caps}
            buf_imgs, buf_caps = [], []
    if buf_imgs:
        yield {"images": buf_imgs, "captions": buf_caps}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training_stage1.yaml")
    p.add_argument("--projector-config", default="configs/projector.yaml")
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--llm-config", default="configs/llm.yaml")
    p.add_argument("--data-config", default="configs/data.yaml")
    p.add_argument("--max-steps", type=int, default=None, help="cap training at N steps (smoke runs)")
    p.add_argument("--subset-size", type=int, default=None,
                   help="train on only the first N Bunny pairs (pilot runs)")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    proj_cfg = load_yaml(args.projector_config)
    enc_cfg = load_yaml(args.encoders_config)
    llm_cfg = load_yaml(args.llm_config)
    data_cfg = load_yaml(args.data_config)
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device

    encoder = build_clip_encoder(enc_cfg).load()
    connector = build_projector(proj_cfg["architecture"])
    llm_embed, tokenizer = _load_llama_embed(
        llm_cfg["model"]["hf_id"],
        device=device,
        dtype_str=llm_cfg["dtype"]["weights"],
    )

    bunny_cfg = data_cfg["bunny_v1_1"]
    dataset = BunnyV11Dataset(root=bunny_cfg["local_path"], limit=args.subset_size)
    if args.subset_size is not None:
        print(f"[stage1] subset: training on first {args.subset_size:,} pairs")
    dataloader = _iter_batches(dataset, cfg["batch"]["per_device_batch_size"])

    ckpt = train_stage1(
        encoder=encoder,
        connector=connector,
        llm_embed=llm_embed,
        tokenizer=tokenizer,
        dataloader=dataloader,
        cfg=cfg,
        max_steps=args.max_steps,
    )
    snapshot_run_metadata({"stage1": cfg, "args": vars(args)}, Path(cfg["output"]["log_dir"]))
    print(f"[ok] Stage 1 connector checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
