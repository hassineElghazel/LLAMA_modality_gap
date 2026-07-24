"""LLaVA-1.5-7B REFERENCE captioner on the dd256 grounding set (n=1300).

Runs the SAME evaluation the diagnostic models get (scripts/08_run_captioning.py):
identical image set (first ``num_images`` val2017 by sorted id), identical detail
instruction, identical greedy decoding and ``max_new_tokens``, all pulled from the
SAME config (configs/description_eval.yaml). Output is written as
``captions_<condition>.json`` in the identical schema (list of {image_id, caption})
so scripts/15_clipscore.py and scripts/19_chair.py score it unchanged.

FAIRNESS: every harness-controlled variable (image set, instruction text, decoding,
token cap, seed) is held identical to the diagnostic runs. The only things that
differ are model-native and intentional — LLaVA-1.5 is run AS RELEASED: its own
336px image processor and its USER/ASSISTANT chat template (that is how the model
is meant to be prompted; the instruction *content* is the same string the
diagnostic models saw). These are the named confounds (336px vision, Vicuna-1.5
LLM, full training), so the resulting numbers are a fully-resourced CEILING.

Vision tower: openai/clip-vit-large-patch14-336 (same CLIP family as the diagnostic
models' ViT-L/14@224, so scoring with a DIFFERENT CLIP — ViT-B/32 in 15_clipscore —
stays non-circular for both).

Usage (on the GPU node):
    python scripts/22_run_reference_captioning.py \
        --config configs/description_eval.yaml --condition llava15_dd256 \
        --model-id llava-hf/llava-1.5-7b-hf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.data.coco_val2017_loader import CocoVal2017Dataset
from src.utils.io import load_yaml
from src.utils.reproducibility import set_seed


def _instruction_from_config(cap_cfg: dict) -> str:
    """The SAME instruction the diagnostic models saw, minus the ``<image>``
    placeholder (LLaVA's processor inserts the image token itself)."""
    return cap_cfg["prompt"]["user"].replace("<image>", "").strip()


def _build_prompt(processor, instruction: str) -> str:
    """LLaVA-1.5's native chat wrapper around the identical instruction text.
    Uses apply_chat_template when available (version-robust), else the canonical
    ``USER: <image>\\n{...} ASSISTANT:`` format."""
    try:
        conversation = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }]
        return processor.apply_chat_template(conversation, add_generation_prompt=True)
    except Exception:
        return f"USER: <image>\n{instruction} ASSISTANT:"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/description_eval.yaml")
    p.add_argument("--condition", default="llava15_dd256",
                   help="output tag -> captions_<condition>.json")
    p.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf",
                   help="HF id or a local snapshot dir (cluster runs offline)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--load-4bit", dest="load_4bit", action="store_true", default=True,
                   help="4-bit NF4 quant (fits the 2080ti; default on)")
    p.add_argument("--no-4bit", dest="load_4bit", action="store_false")
    p.add_argument("--limit", type=int, default=None, help="smoke-test on first N images")
    args = p.parse_args()

    cap_cfg = load_yaml(args.config)
    set_seed(int(cap_cfg.get("seed", 42)))

    # --- EXACT same image selection as 08_run_captioning.py ---
    ds = CocoVal2017Dataset(
        annotations_json=cap_cfg["eval_set"]["annotations_json"],
        image_root=cap_cfg["eval_set"]["image_root"],
    )
    n = int(cap_cfg["eval_set"]["num_images"])
    items = list(ds.items())[:n]
    if args.limit:
        items = items[: args.limit]

    # --- EXACT same generation settings as the diagnostic runs ---
    gen = cap_cfg["generation"]
    max_new = int(gen["max_new_tokens"])          # 256
    do_sample = bool(gen.get("do_sample", False))  # greedy
    num_beams = int(gen.get("num_beams", 1))
    instruction = _instruction_from_config(cap_cfg)

    from transformers import AutoProcessor, LlavaForConditionalGeneration

    # device_map pins ALL modules to the single SLURM-allocated GPU (index 0).
    # REQUIRED for 4-bit: bitsandbytes places quantized layers at load time and a
    # 4-bit model can NOT be .to()-moved afterwards (silent CPU placement / errors).
    # We do NOT use device_map="auto" on purpose — baldur can expose a co-located
    # dead GPU, and "auto" could shard onto it; "{'':0}" forces the healthy card the
    # sbatch preflight already validated.
    load_kwargs: dict = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True,
                             device_map={"": 0})
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = LlavaForConditionalGeneration.from_pretrained(args.model_id, **load_kwargs)
    model.eval()
    compute_dtype = torch.float16

    processor = AutoProcessor.from_pretrained(args.model_id)
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok.padding_side = "left"                 # required for batched generation
        if tok.pad_token_id is None:              # LLaMA/Vicuna ship no pad token
            tok.pad_token = tok.eos_token

    prompt = _build_prompt(processor, instruction)
    print(f"[llava15] model={args.model_id}  n_images={len(items)}  "
          f"max_new_tokens={max_new}  do_sample={do_sample}  beams={num_beams}")
    print(f"[llava15] prompt >>>\n{prompt}\n<<<")

    records: list[dict] = []
    for start in tqdm(range(0, len(items), args.batch_size), desc="llava15 caption"):
        batch = items[start:start + args.batch_size]
        images = [Image.open(it.image_path).convert("RGB") for it in batch]
        prompts = [prompt] * len(batch)
        inputs = processor(images=images, text=prompts, return_tensors="pt",
                           padding=True).to("cuda")
        if "pixel_values" in inputs:                      # cast ONLY pixels; ids stay long
            inputs["pixel_values"] = inputs["pixel_values"].to(compute_dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new,
                                 do_sample=do_sample, num_beams=num_beams,
                                 use_cache=True, pad_token_id=tok.pad_token_id)
        # Decode robustly. Splitting on the assistant delimiter is invariant to
        # whether this transformers version expands the <image> token in the
        # PROCESSOR (input_ids already long) or in the MODEL forward (input_ids
        # short -> a token-index slice would leak image positions into the text).
        input_len = inputs["input_ids"].shape[1]
        full_texts = processor.batch_decode(out, skip_special_tokens=True)
        tail_texts = processor.batch_decode(out[:, input_len:], skip_special_tokens=True)
        for it, full, tail in zip(batch, full_texts, tail_texts):
            cap = full.rsplit("ASSISTANT:", 1)[-1].strip() if "ASSISTANT:" in full else tail.strip()
            records.append({"image_id": it.image_id, "caption": cap})

    out_path = Path(cap_cfg["output"]["predictions_dir"]) / f"captions_{args.condition}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(records, f)
    print(f"[llava15] wrote {len(records)} captions -> {out_path}")
    print(f"[llava15] sample[0]: {records[0]['caption'][:200]!r}")


if __name__ == "__main__":
    main()
