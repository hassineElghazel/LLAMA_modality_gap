"""Live captioning demo: load one checkpoint, caption images on demand.

Built for the thesis defence. The model is loaded ONCE (a 4-bit 7B takes a few
minutes) and then captions any number of images without reloading, so nothing
stalls in front of an audience.

Model construction mirrors scripts/08_run_captioning.py exactly (same connector
load, same LoRA re-attachment, same 4-bit LLM) and generation calls the same
VLM.generate with the same prompt and gen_kwargs from configs/captioning.yaml,
so what you show is what the thesis measured.

Usage
-----
    # interactive: load once, then paste image paths at the prompt
    python scripts/30_demo_caption.py \
        --vlm-checkpoint outputs/checkpoints/stage2_vlm_Cloc_long.pt

    # one-shot / scripted
    python scripts/30_demo_caption.py \
        --vlm-checkpoint outputs/checkpoints/stage2_vlm_Cloc_long.pt \
        --image a.jpg b.jpg

    # side-by-side against the 450-step model (loads both; needs the VRAM)
    python scripts/30_demo_caption.py \
        --vlm-checkpoint outputs/checkpoints/stage2_vlm_Cloc_long.pt \
        --compare outputs/checkpoints/stage2_vlm_C3pinr.pt --image a.jpg
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.coco_val2017_loader import load_image
from src.utils.io import load_yaml


def build(ckpt, enc_cfg, proj_cfg, llm_cfg, lora_cfg):
    """Identical to scripts/08_run_captioning.py::_build_vlm."""
    from src.encoders.clip_encoder import build_clip_encoder
    from src.models.checkpoint import load_projector
    from src.models.projector import build_projector
    from src.models.vlm import VLM, VLMConfig

    encoder = build_clip_encoder(enc_cfg).load()
    device = enc_cfg["inference"]["device"]
    if str(ckpt).lower() == "random":
        connector, llm_trainable = build_projector(proj_cfg["architecture"]).to(device), {}
    else:
        blob = torch.load(ckpt, map_location="cpu")
        if "config" in blob:
            connector, llm_trainable = load_projector(ckpt).to(device), {}
        else:
            connector = build_projector(proj_cfg["architecture"])
            connector.load_state_dict(blob["connector"])
            connector = connector.to(device)
            llm_trainable = blob.get("llm_trainable") or {}

    vlm = VLM(encoder, connector, VLMConfig(
        llm_hf_id=llm_cfg["model"]["hf_id"],
        weights_dtype=llm_cfg["dtype"]["weights"],
        device=device, load_in_4bit=True,
    )).load_llm()

    if llm_trainable and lora_cfg:
        from peft import LoraConfig, get_peft_model
        vlm._llm = get_peft_model(vlm._llm, LoraConfig(
            r=int(lora_cfg["r"]), lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            target_modules=list(lora_cfg["target_modules"]),
            bias=str(lora_cfg.get("bias", "none")), task_type="CAUSAL_LM"))
        vlm._llm.load_state_dict(llm_trainable, strict=False)
    return vlm


@torch.no_grad()
def caption(vlm, path, prompt, gen_kwargs):
    img = load_image(path)
    t0 = time.time()
    out = vlm.generate([img], [prompt], **gen_kwargs)[0].strip()
    return out, time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vlm-checkpoint", required=True)
    p.add_argument("--compare", default=None, help="second checkpoint, captioned side by side")
    p.add_argument("--image", nargs="*", default=None, help="omit for interactive mode")
    p.add_argument("--config", default="configs/captioning.yaml")
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--projector-config", default="configs/projector.yaml")
    p.add_argument("--llm-config", default="configs/llm.yaml")
    p.add_argument("--stage2-config", default="configs/training_stage2.yaml")
    p.add_argument("--prompt", default=None, help="override configs/captioning.yaml prompt")
    p.add_argument("--max-new-tokens", type=int, default=None)
    args = p.parse_args()

    cap_cfg = load_yaml(args.config)
    enc_cfg = load_yaml(args.encoders_config)
    proj_cfg = load_yaml(args.projector_config)
    llm_cfg = load_yaml(args.llm_config)
    lora_cfg = load_yaml(args.stage2_config).get("lora")

    prompt = (args.prompt or cap_cfg["prompt"]["user"]).strip()
    gen_kwargs = dict(cap_cfg["generation"])
    if args.max_new_tokens:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens

    print(f"[demo] loading {args.vlm_checkpoint} ...", flush=True)
    t0 = time.time()
    models = [(Path(args.vlm_checkpoint).stem, build(args.vlm_checkpoint, enc_cfg, proj_cfg, llm_cfg, lora_cfg))]
    if args.compare:
        print(f"[demo] loading {args.compare} ...", flush=True)
        models.append((Path(args.compare).stem, build(args.compare, enc_cfg, proj_cfg, llm_cfg, lora_cfg)))
    print(f"[demo] ready in {time.time()-t0:.0f}s. prompt: {prompt!r}")
    print(f"[demo] generation: {gen_kwargs}\n")

    def run(path):
        if not Path(path).exists():
            print(f"  ! no such file: {path}\n"); return
        for name, vlm in models:
            try:
                text, dt = caption(vlm, path, prompt, gen_kwargs)
                label = f"[{name}]" if len(models) > 1 else ""
                print(f"  {label} {text}   ({dt:.1f}s)")
            except Exception as e:                      # never crash a live demo
                print(f"  ! {name} failed: {type(e).__name__}: {e}")
        print()

    if args.image:
        for path in args.image:
            print(f"> {path}")
            run(path)
        return

    print("Interactive. Paste an image path and press enter. Ctrl-D or 'q' to quit.\n")
    while True:
        try:
            line = input("image> ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print(); break
        if line.lower() in {"q", "quit", "exit"}: break
        if line:
            run(line)


if __name__ == "__main__":
    main()
