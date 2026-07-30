"""Unified LLaVA captioner for the 3-arm study (Arm0 pretrained OR a trained arm).

Generates captions under the SAME regime as the reference run (scripts/22): detail
instruction from the eval config, greedy decoding, identical max_new_tokens. Works
on either eval set:
  * dd256 (COCO val2017, n=1300)  -> eval_set has ``annotations_json`` (CocoVal2017)
  * DOCCI-test (n=5000)           -> eval_set has ``manifest_path`` (docci_loader)
Output schema is the shared ``captions_<tag>.json`` (list of {image_id, caption}),
so 15_clipscore / 15b_siglip / 19_chair / 28_score_docci_refs all consume it unchanged.

Arm selection:
  * Arm0 (pretrained reference): pass neither --adapter-dir nor --connector-path.
  * A trained arm: pass --adapter-dir (LoRA) and --connector-path (connector.pt);
    the trained connector is loaded into multi_modal_projector, then the LoRA
    adapter is attached. This is what makes the location/pinned/vanilla weights
    take effect at inference.

Usage (GPU node):
    # Arm0 reference on dd256
    python scripts/27_caption_llava.py --eval-config configs/description_eval.yaml \
        --tag arm0_dd256
    # location arm on DOCCI-test
    python scripts/27_caption_llava.py --eval-config configs/docci_eval.yaml \
        --adapter-dir outputs/checkpoints/llava_docci_location \
        --connector-path outputs/checkpoints/llava_docci_location/connector.pt \
        --tag location_docci
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.utils.io import load_yaml
from src.utils.reproducibility import set_seed


def _instruction_from_config(cfg: dict) -> str:
    return cfg["prompt"]["user"].replace("<image>", "").strip()


def _build_prompt(processor, instruction: str) -> str:
    try:
        conv = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": instruction}]}]
        return processor.apply_chat_template(conv, add_generation_prompt=True)
    except Exception:
        return f"USER: <image>\n{instruction} ASSISTANT:"


def _load_items(cfg: dict):
    """-> list of (image_id:int, image_path:str), branching on the eval-set kind."""
    es = cfg["eval_set"]
    if es.get("annotations_json"):                       # dd256 / COCO val2017
        from src.data.coco_val2017_loader import CocoVal2017Dataset
        ds = CocoVal2017Dataset(annotations_json=es["annotations_json"],
                                image_root=es["image_root"])
        items = list(ds.items())[: int(es["num_images"])]
        return [(it.image_id, str(it.image_path)) for it in items]
    if es.get("manifest_path"):                          # DOCCI-test
        from src.data.docci_loader import load_docci_manifest
        items = load_docci_manifest(es["manifest_path"])
        if es.get("num_images"):
            items = items[: int(es["num_images"])]
        return [(it.image_id, str(it.image_path)) for it in items]
    raise SystemExit("eval_set needs either 'annotations_json' (COCO) or 'manifest_path' (DOCCI).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-config", required=True)
    p.add_argument("--tag", required=True, help="output -> captions_<tag>.json")
    p.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--adapter-dir", default=None, help="LoRA adapter dir (omit for Arm0).")
    p.add_argument("--connector-path", default=None, help="connector.pt (omit for Arm0).")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--load-4bit", dest="load_4bit", action="store_true", default=True)
    p.add_argument("--no-4bit", dest="load_4bit", action="store_false")
    p.add_argument("--limit", type=int, default=None, help="smoke-test on first N images")
    args = p.parse_args()

    cfg = load_yaml(args.eval_config)
    set_seed(int(cfg.get("seed", 42)))
    gen = cfg["generation"]
    max_new = int(gen["max_new_tokens"])
    do_sample = bool(gen.get("do_sample", False))
    num_beams = int(gen.get("num_beams", 1))
    instruction = _instruction_from_config(cfg)

    items = _load_items(cfg)
    if args.limit:
        items = items[: args.limit]

    from transformers import AutoProcessor, LlavaForConditionalGeneration
    load_kwargs: dict = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={"": 0})
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = LlavaForConditionalGeneration.from_pretrained(args.model_id, **load_kwargs)

    # --- load trained connector (before attaching the adapter) ---
    if args.connector_path:
        sd = torch.load(args.connector_path, map_location="cpu")
        model.multi_modal_projector.load_state_dict(sd)
        model.multi_modal_projector.to(torch.float16)     # match the fp16 inference pipeline
        print(f"[cap] loaded connector <- {args.connector_path}")
    # --- attach LoRA adapter ---
    if args.adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        print(f"[cap] attached LoRA adapter <- {args.adapter_dir}")
    arm_kind = "trained" if (args.adapter_dir or args.connector_path) else "arm0-pretrained"
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_id)
    tok = processor.tokenizer
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    prompt = _build_prompt(processor, instruction)
    print(f"[cap] tag={args.tag} kind={arm_kind} n={len(items)} max_new={max_new} "
          f"do_sample={do_sample} beams={num_beams}")

    records: list[dict] = []
    for start in tqdm(range(0, len(items), args.batch_size), desc=f"caption {args.tag}"):
        batch = items[start:start + args.batch_size]
        images = [Image.open(pth).convert("RGB") for _, pth in batch]
        inputs = processor(images=images, text=[prompt] * len(batch),
                           return_tensors="pt", padding=True).to("cuda")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=do_sample,
                                 num_beams=num_beams, use_cache=True,
                                 pad_token_id=tok.pad_token_id)
        input_len = inputs["input_ids"].shape[1]
        full_texts = processor.batch_decode(out, skip_special_tokens=True)
        tail_texts = processor.batch_decode(out[:, input_len:], skip_special_tokens=True)
        for (image_id, _), full, tail in zip(batch, full_texts, tail_texts):
            cap = full.rsplit("ASSISTANT:", 1)[-1].strip() if "ASSISTANT:" in full else tail.strip()
            records.append({"image_id": image_id, "caption": cap})

    out_path = Path(cfg["output"]["predictions_dir"]) / f"captions_{args.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(records, f)
    print(f"[cap] wrote {len(records)} captions -> {out_path}")
    print(f"[cap] sample[0]: {records[0]['caption'][:200]!r}")


if __name__ == "__main__":
    main()
