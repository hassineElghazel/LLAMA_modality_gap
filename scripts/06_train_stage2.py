"""Stage 2: autoregressive captioning with LoRA on LLaVA-Instruct-150K.

CLIP ViT-L/14 frozen, connector refined, LLaMA-2-7B trained via LoRA adapters.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import torch

from src.data.llava_instruct_loader import LLaVAInstruct150KDataset, load_image
from src.encoders.clip_encoder import build_clip_encoder
from src.models.checkpoint import load_projector
from src.models.projector import build_projector
from src.models.vlm import VLM, VLMConfig
from src.training.stage2_sft import train_stage2
from src.utils.io import load_yaml, snapshot_run_metadata
from src.utils.reproducibility import set_seed


IMAGE_PLACEHOLDER = "<image>"


def _format_conversation(convs: list[dict], image_token: str = IMAGE_PLACEHOLDER) -> tuple[str, str]:
    """Return (prompt, response) for a LLaVA conversation.

    Concatenates all human turns into a single prompt (prefixed with the image
    placeholder once) and all gpt turns into a single response. Multi-turn
    conversations are flattened — Stage 2 is captioning-focused, not chat.
    """
    human_parts = []
    gpt_parts = []
    for turn in convs:
        role = str(turn.get("from", "")).lower()
        val = str(turn.get("value", ""))
        if role == "human":
            human_parts.append(val.replace(image_token, "").strip())
        elif role in ("gpt", "assistant"):
            gpt_parts.append(val.strip())
    prompt = f"{image_token}\n" + "\n".join([p for p in human_parts if p])
    response = "\n".join([p for p in gpt_parts if p])
    return prompt, response


def _llava_collate(items, tokenizer, image_token_id: int, max_length: int = 512):
    images = []
    input_ids_list = []
    labels_list = []
    for it in items:
        try:
            img = load_image(it.image_path)
        except FileNotFoundError:
            continue
        prompt, response = _format_conversation(it.conversations)
        # Tokenize prompt and response separately so we can mask prompt tokens
        # in the labels (loss on response positions only).
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        ids = p_ids + r_ids + eos
        labels = [-100] * len(p_ids) + r_ids + eos
        # Truncate to max_length.
        ids = ids[:max_length]
        labels = labels[:max_length]
        images.append(img)
        input_ids_list.append(torch.tensor(ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))

    if not images:
        return None
    # Right-pad to common length using the tokenizer's pad id.
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    max_len = max(t.shape[0] for t in input_ids_list)
    input_ids = torch.full((len(images), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(images), max_len), -100, dtype=torch.long)
    for i, (ids, lbl) in enumerate(zip(input_ids_list, labels_list)):
        input_ids[i, : ids.shape[0]] = ids
        labels[i, : lbl.shape[0]] = lbl
    return {"images": images, "input_ids": input_ids, "labels": labels}


def _iter_batches(dataset, tokenizer, image_token_id, batch_size: int) -> Iterator[dict]:
    buf = []
    for item in dataset:
        buf.append(item)
        if len(buf) == batch_size:
            batch = _llava_collate(buf, tokenizer, image_token_id)
            buf = []
            if batch is not None:
                yield batch
    if buf:
        batch = _llava_collate(buf, tokenizer, image_token_id)
        if batch is not None:
            yield batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training_stage2.yaml")
    p.add_argument("--projector-config", default="configs/projector.yaml")
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--llm-config", default="configs/llm.yaml")
    p.add_argument("--data-config", default="configs/data.yaml")
    p.add_argument("--init-connector",
                   help="override init_from.connector_checkpoint (use 'random' for C1)",
                   default=None)
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    proj_cfg = load_yaml(args.projector_config)
    enc_cfg = load_yaml(args.encoders_config)
    llm_cfg = load_yaml(args.llm_config)
    data_cfg = load_yaml(args.data_config)
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device

    # Encoder (frozen CLIP ViT-L/14).
    encoder = build_clip_encoder(enc_cfg).load()

    # Connector: load from Stage 1 ckpt unless overridden to "random" (C1).
    init = args.init_connector if args.init_connector is not None \
        else cfg["init_from"]["connector_checkpoint"]
    if str(init).lower() == "random":
        connector = build_projector(proj_cfg["architecture"])
        print("[stage2] connector init: random (C1)")
    else:
        connector = load_projector(init)
        print(f"[stage2] connector init: {init}")

    # VLM (loads LLaMA-2-7B + sets up the splice path).
    quant_cfg = llm_cfg.get("quantization", {})
    vlm = VLM(encoder, connector, VLMConfig(
        llm_hf_id=llm_cfg["model"]["hf_id"],
        weights_dtype=llm_cfg["dtype"]["weights"],
        device=device,
        load_in_4bit=bool(quant_cfg.get("load_in_4bit", False)),
    )).load_llm()
    tokenizer = vlm._tokenizer
    image_token_id = vlm._image_token_id

    # LLaVA-Instruct-150K dataloader.
    llava_cfg = data_cfg["llava_instruct_150k"]
    dataset = LLaVAInstruct150KDataset(
        root=llava_cfg["local_path"],
        image_root=llava_cfg["image_root"],
    )
    dataloader = _iter_batches(
        dataset, tokenizer, image_token_id, cfg["batch"]["per_device_batch_size"]
    )

    ckpt = train_stage2(vlm, dataloader, cfg, max_steps=args.max_steps)
    snapshot_run_metadata({"stage2": cfg, "args": vars(args)}, Path(cfg["output"]["log_dir"]))
    print(f"[ok] Stage 2 VLM checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
