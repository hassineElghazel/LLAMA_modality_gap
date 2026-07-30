"""3-arm QLoRA fine-tune of pretrained LLaVA-1.5-7B on DOCCI detail captions.

Ports the diagnostic Cloc loss (src/training/stage2_distance_sft.py) onto the HF
LlavaForConditionalGeneration stack. ONE recipe, three arms selected with --arm;
they differ ONLY in the geometry lambdas on the pooled-576 connector output:

    vanilla   lambda_d=0.0  lambda_s=0.0  lambda_r=0.0   (C3 analog)
    pinned    lambda_d=0.0  lambda_s=1.0  lambda_r=1.0   (C3pinr analog)
    location  lambda_d=0.1  lambda_s=1.0  lambda_r=1.0   (Cloc analog)

Loss (convex; lambda_d=0 -> pure AR):
    L = (1-lambda_d)*L_AR + lambda_d*L_dist + lambda_s*L_scale + lambda_r*L_rank
    z_img   = mean_576( connector(vision(image)) )          (B, 4096)
    L_dist  = || mean_b(z_img) - mu_y ||^2 / trace_x         (frozen mu_y, trace_x)
    L_scale = (btrace / btrace0 - 1)^2                        (btrace0 captured @ step 1)
    L_rank  = (PR / effrank0 - 1)^2                           (effrank0 captured @ step 1)

Geometry step: vision_tower runs under no_grad (frozen), ONLY the connector is
in-graph -> the batch-mean gradient is a pure translation of the image cloud, and
a batch of 32 images fits the 2080ti alongside the 4-bit LLM.

Correctness guards baked in:
  * LoRA is scoped to language_model modules by FULL NAME (the CLIP vision tower
    also owns q_proj/v_proj; a suffix-only target would adapt it too).
  * multi_modal_projector is kept out of 4-bit (llm_int8_skip_modules) and upcast
    to fp32 + trainable -- it must move for the location term.
  * fp16 AMP (Turing sm_75 has no bf16); the geometry algebra is done in fp32.

Usage (GPU node, one arm per job):
    python scripts/26_finetune_llava_3arm.py \
        --config configs/training_llava_docci.yaml --arm location
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from PIL import Image

from src.data.docci_loader import load_docci_manifest
from src.utils.io import load_yaml, save_json, snapshot_run_metadata

ARM_LAMBDAS = {
    "vanilla":  dict(lambda_d=0.0, lambda_s=0.0, lambda_r=0.0),
    "pinned":   dict(lambda_d=0.0, lambda_s=1.0, lambda_r=1.0),
    "location": dict(lambda_d=0.1, lambda_s=1.0, lambda_r=1.0),
}


def build_prompt(processor, instruction: str) -> str:
    """USER: <image>\\n{instruction} ASSISTANT:  (chat template if available)."""
    try:
        conv = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": instruction}]}]
        return processor.apply_chat_template(conv, add_generation_prompt=True)
    except Exception:
        return f"USER: <image>\n{instruction} ASSISTANT:"


def language_model_lora_targets(model, suffixes) -> list[str]:
    """Full module names of language_model linears matching the target suffixes.

    Full-name targets make peft's endswith match exact, so the CLIP vision tower's
    own q_proj/v_proj (same suffix, different subtree) are NOT adapted.
    """
    sset = set(suffixes)
    names = {
        name
        for name, _ in model.named_modules()
        if "language_model." in name and name.split(".")[-1] in sset
    }
    return sorted(names)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/training_llava_docci.yaml")
    ap.add_argument("--arm", required=True, choices=sorted(ARM_LAMBDAS))
    ap.add_argument("--max-steps", type=int, default=None, help="Debug cap on optimizer steps.")
    ap.add_argument("--out-dir", default=None, help="Override checkpoint dir.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    lam = ARM_LAMBDAS[args.arm]
    lambda_d, lambda_s, lambda_r = lam["lambda_d"], lam["lambda_s"], lam["lambda_r"]
    seed = int(cfg.get("seed", 42))
    random.seed(seed); torch.manual_seed(seed)

    out_dir = Path(args.out_dir or (Path(cfg["output"]["checkpoint_root"]) / f"llava_docci_{args.arm}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[3arm] arm={args.arm}  lambda_d={lambda_d} lambda_s={lambda_s} lambda_r={lambda_r}")
    print(f"[3arm] out_dir={out_dir}")

    # ---- data ----
    items = load_docci_manifest(cfg["data"]["train_manifest"])
    print(f"[3arm] {len(items)} DOCCI-train rows")
    instruction = "Describe the following image in detail."
    max_cap = int(cfg["data"].get("max_caption_tokens", 256))

    # ---- anchors (frozen) ----
    dcfg = cfg["distance"]
    mu_y = torch.load(dcfg["mu_y_source"]).float().cuda()          # (4096,)
    if dcfg.get("trace_x_source"):
        trace_x = float(json.loads(Path(dcfg["trace_x_source"]).read_text())["trace_x"])
    else:
        trace_x = float(dcfg["trace_x"])
    print(f"[3arm] anchors: ||mu_y||={float(mu_y.norm()):.3f}  trace_x={trace_x:.3f}")

    # ---- model (4-bit LLM; connector kept fp + trainable) ----
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              LlavaForConditionalGeneration,
                              get_cosine_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_id = cfg["model"]["model_id"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["multi_modal_projector"],   # connector stays fp, trainable
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, quantization_config=bnb, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, device_map={"": 0})
    processor = AutoProcessor.from_pretrained(model_id)
    tok = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_targets = language_model_lora_targets(model, cfg["lora"]["target_modules"])
    print(f"[3arm] LoRA on {len(lora_targets)} language_model linears "
          f"(0 vision-tower modules -> {all('vision_tower' not in t for t in lora_targets)})")
    lora_cfg = LoraConfig(
        r=int(cfg["lora"]["r"]), lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]), bias=cfg["lora"].get("bias", "none"),
        target_modules=lora_targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)

    # connector: upcast to fp32 + trainable (the location/scale/rank terms move it)
    connector = model.base_model.model.multi_modal_projector
    connector.to(torch.float32)
    for p in connector.parameters():
        p.requires_grad_(True)
    connector_dtype = torch.float32

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"[3arm] trainable params: {n_train/1e6:.2f}M "
          f"(LoRA + connector {sum(p.numel() for p in connector.parameters())/1e6:.2f}M)")

    # ---- vision-tower geometry pieces (mirror get_image_features internals) ----
    vfl = getattr(model.config, "vision_feature_layer", -2)
    vss = getattr(model.config, "vision_feature_select_strategy", "default")
    vision_tower = model.base_model.model.vision_tower
    image_processor = getattr(processor, "image_processor")

    def pooled_image_tokens(pixel_values):
        """(B,4096) mean over the 576 projected tokens; vision frozen, connector in-graph."""
        with torch.no_grad():
            vt = vision_tower(pixel_values, output_hidden_states=True)
            hs = vt.hidden_states[vfl]                      # (B, 577, 1024)
            if vss == "default":
                hs = hs[:, 1:, :]                           # drop CLS -> (B, 576, 1024)
        z = connector(hs.to(connector_dtype))              # (B, 576, 4096) WITH grad
        return z.mean(dim=1)                               # (B, 4096)

    # ---- optimizer / schedule ----
    accum = int(cfg["batch"]["gradient_accumulation_steps"])
    epochs = int(cfg["schedule"]["num_epochs"])
    steps_per_epoch = math.ceil(len(items) / accum)
    total_steps = steps_per_epoch * epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(cfg["schedule"]["warmup_ratio"] * total_steps))
    opt = torch.optim.AdamW(
        trainable, lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
        betas=tuple(cfg["optimizer"]["betas"]), eps=float(cfg["optimizer"]["eps"]))
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
    scaler = torch.cuda.amp.GradScaler()
    print(f"[3arm] total_steps={total_steps} (warmup={warmup}, accum={accum})")

    geo_bs = int(dcfg["batch_size"])
    use_scale = lambda_s > 0.0
    use_rank = lambda_r > 0.0
    run_geo = (lambda_d > 0.0) or use_scale or use_rank
    w_ar = 1.0 - lambda_d
    btrace0: float | None = None
    effrank0: float | None = None

    def ar_micro(item) -> torch.Tensor:
        """AR captioning loss for one example (prompt masked to -100)."""
        image = Image.open(item.image_path).convert("RGB")
        # pre-truncate the long DOCCI caption in token space
        cap_ids = tok(item.caption, add_special_tokens=False).input_ids[:max_cap]
        caption = tok.decode(cap_ids, skip_special_tokens=True)
        prompt = build_prompt(processor, instruction)
        full = f"{prompt} {caption}{tok.eos_token}"
        enc_p = processor(text=prompt, images=image, return_tensors="pt")
        enc_f = processor(text=full, images=image, return_tensors="pt")
        input_ids = enc_f["input_ids"].cuda()
        attn = enc_f["attention_mask"].cuda()
        pv = enc_f["pixel_values"].cuda().to(torch.float16)
        labels = input_ids.clone()
        lp = enc_p["input_ids"].shape[1]
        labels[:, :lp] = -100
        labels[labels == tok.pad_token_id] = -100
        out = model(input_ids=input_ids, attention_mask=attn, pixel_values=pv, labels=labels)
        return out.loss

    def geo_losses():
        nonlocal btrace0, effrank0
        batch = random.sample(items, min(geo_bs, len(items)))
        images = [Image.open(it.image_path).convert("RGB") for it in batch]
        pv = image_processor(images=images, return_tensors="pt").pixel_values.cuda().to(torch.float16)
        z = pooled_image_tokens(pv)
        zf = z.float()
        zbar = zf.mean(dim=0)
        l_dist = ((zbar - mu_y) ** 2).sum() / trace_x
        # scale
        btrace = ((zf - zbar) ** 2).sum(dim=1).mean()
        if use_scale and btrace0 is None:
            btrace0 = float(btrace.detach())
        l_scale = ((btrace / btrace0 - 1.0) ** 2) if use_scale else zf.new_zeros(())
        # rank (participation ratio of the centered batch Gram)
        zc = zf - zbar
        G = zc @ zc.t()
        pr = torch.diagonal(G).sum() ** 2 / (G * G).sum().clamp_min(1e-12)
        if use_rank and effrank0 is None:
            effrank0 = float(pr.detach())
        l_rank = ((pr / effrank0 - 1.0) ** 2) if use_rank else zf.new_zeros(())
        return l_dist, l_scale, l_rank

    # ---- train loop ----
    model.train()
    log_every = int(cfg["logging"]["log_every_steps"])
    save_every = int(cfg["logging"]["save_every_steps"])
    order = list(range(len(items)))
    ptr = 0
    for step in range(total_steps):
        opt.zero_grad(set_to_none=True)
        ar_val = 0.0
        for _ in range(accum):
            if ptr == 0:
                random.shuffle(order)
            item = items[order[ptr]]
            ptr = (ptr + 1) % len(items)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = ar_micro(item)
            scaler.scale(w_ar * loss / accum).backward()
            ar_val += float(loss.detach()) / accum

        gd = gs = gr = 0.0
        if run_geo:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                l_dist, l_scale, l_rank = geo_losses()
                geo = lambda_d * l_dist + lambda_s * l_scale + lambda_r * l_rank
            scaler.scale(geo).backward()
            gd, gs, gr = float(l_dist.detach()), float(l_scale.detach()), float(l_rank.detach())

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % log_every == 0 or step == total_steps - 1:
            print(f"[{step:04d}/{total_steps}] AR={ar_val:.4f} "
                  f"dist={gd:.4f} scale={gs:.5f} rank={gr:.5f} lr={sched.get_last_lr()[0]:.2e}",
                  flush=True)
        if save_every and (step + 1) % save_every == 0:
            model.save_pretrained(out_dir)                          # adapter
            torch.save(connector.state_dict(), out_dir / "connector.pt")  # + connector (crash-safe)

    # ---- save ----
    model.save_pretrained(out_dir)                                  # LoRA adapter
    torch.save(connector.state_dict(), out_dir / "connector.pt")    # trained connector
    sidecar = {
        "arm": args.arm, "lambda_d": lambda_d, "lambda_s": lambda_s, "lambda_r": lambda_r,
        "total_steps": total_steps, "accum": accum, "geo_batch_size": geo_bs,
        "btrace0": btrace0, "effrank0": effrank0, "trace_x": trace_x,
        "mu_y_source": dcfg["mu_y_source"], "model_id": model_id, "seed": seed,
        "n_train_examples": len(items), "lora_targets": len(lora_targets),
    }
    save_json(sidecar, out_dir / "train_sidecar.json")
    snapshot_run_metadata(cfg, out_dir, config_files=[args.config])
    print(f"[3arm] done -> {out_dir}  (adapter + connector.pt + train_sidecar.json)")


if __name__ == "__main__":
    main()
