"""VQAv2 accuracy eval, filtered to COCO val2017 images (guaranteed NOT in training).

VQAv2 val annotations are on COCO val2014. Since train2017 (the LLaVA training
images) absorbed most of val2014, we keep ONLY the questions whose image is in
val2017 -- the held-out split disjoint from train2017 -- so eval images are
provably unseen. COCO image IDs are stable across 2014/2017, so filtering by the
val2017 image-id set is exact.

Needs the VQAv2 val files (download once):
  data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json
  data/vqav2/v2_mscoco_val2014_annotations.json

Usage:
    python scripts/23_eval_vqav2.py --condition C3pinr \
        --vlm-checkpoint outputs/checkpoints/stage2_vlm_C3pinr.pt --num-questions 5000
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from tqdm import tqdm

from src.captioning.inference import _format_prompt
from src.data.coco_val2017_loader import load_image
from src.encoders.clip_encoder import build_clip_encoder
from src.models.checkpoint import load_projector
from src.models.projector import build_projector
from src.models.vlm import VLM, VLMConfig
from src.utils.io import load_yaml, snapshot_run_metadata
from src.utils.reproducibility import set_seed

PROMPT = "<image>\n{q}\nAnswer the question using a single word or phrase."

# ---- official VQA answer normalization (Antol et al.) ----
_MANUAL = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
           "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ARTICLES = {"a", "an", "the"}
_CONTRACT = {"dont": "don't", "isnt": "isn't", "arent": "aren't", "cant": "can't",
             "couldnt": "couldn't", "wasnt": "wasn't", "wouldnt": "wouldn't",
             "doesnt": "doesn't", "didnt": "didn't", "wont": "won't", "havent": "haven't",
             "hasnt": "hasn't", "thats": "that's", "youre": "you're", "im": "i'm"}
_PUNCT = ";/[]\"{}()=+\\_-><@`,?!"
_PERIOD = re.compile(r"(?<!\d)\.(?!\d)")
_COMMA = re.compile(r"(\d)(,)(\d)")


def _process_punct(t: str) -> str:
    for p in _PUNCT:
        if (p + " " in t) or (" " + p in t) or (_COMMA.search(t) is not None):
            t = t.replace(p, "")
        else:
            t = t.replace(p, " ")
    return _PERIOD.sub("", t)


def _process_digit_article(t: str) -> str:
    out = [_MANUAL.get(w, w) for w in t.lower().split() if _MANUAL.get(w, w) not in _ARTICLES]
    out = [_CONTRACT.get(w, w) for w in out]
    return " ".join(out)


def norm_answer(s: str) -> str:
    s = s.replace("\n", " ").replace("\t", " ").strip()
    return _process_digit_article(_process_punct(s))


def vqa_accuracy(pred: str, gt_answers: list[str]) -> float:
    p = norm_answer(pred)
    gts = [norm_answer(a) for a in gt_answers]
    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1:]
        matches = sum(1 for g in others if g == p)
        accs.append(min(1.0, matches / 3.0))
    return sum(accs) / len(accs) if accs else 0.0


def _build_vlm(vlm_checkpoint, enc_cfg, proj_cfg, llm_cfg, lora_cfg) -> VLM:
    encoder = build_clip_encoder(enc_cfg).load()
    dev = enc_cfg["inference"]["device"]
    if str(vlm_checkpoint).lower() == "random":
        connector = build_projector(proj_cfg["architecture"]).to(dev); llm_trainable = {}
    else:
        blob = torch.load(vlm_checkpoint, map_location="cpu")
        if "config" in blob:
            connector = load_projector(vlm_checkpoint).to(dev); llm_trainable = {}
        else:
            connector = build_projector(proj_cfg["architecture"])
            connector.load_state_dict(blob["connector"]); connector = connector.to(dev)
            llm_trainable = blob.get("llm_trainable") or {}
    vlm = VLM(encoder, connector, VLMConfig(
        llm_hf_id=llm_cfg["model"]["hf_id"], weights_dtype=llm_cfg["dtype"]["weights"],
        device=dev, load_in_4bit=True)).load_llm()
    if llm_trainable and lora_cfg:
        from peft import LoraConfig, get_peft_model
        peft_cfg = LoraConfig(r=int(lora_cfg["r"]), lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]), target_modules=list(lora_cfg["target_modules"]),
            bias=str(lora_cfg.get("bias", "none")), task_type="CAUSAL_LM")
        vlm._llm = get_peft_model(vlm._llm, peft_cfg)
        vlm._llm.load_state_dict(llm_trainable, strict=False)
    return vlm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True)
    p.add_argument("--vlm-checkpoint", required=True)
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--projector-config", default="configs/projector.yaml")
    p.add_argument("--llm-config", default="configs/llm.yaml")
    p.add_argument("--stage2-config", default="configs/training_stage2.yaml")
    p.add_argument("--questions", default="data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json")
    p.add_argument("--annotations", default="data/vqav2/v2_mscoco_val2014_annotations.json")
    p.add_argument("--val2017-instances", default="data/coco/annotations/instances_val2017.json")
    p.add_argument("--image-root", default="data/coco/val2017")
    p.add_argument("--num-questions", type=int, default=5000, help="subsample after val2017 filter (0=all)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="outputs/metrics")
    args = p.parse_args()

    set_seed(args.seed)
    enc_cfg = load_yaml(args.encoders_config)
    proj_cfg = load_yaml(args.projector_config)
    llm_cfg = load_yaml(args.llm_config)
    stage2_cfg = load_yaml(args.stage2_config)
    lora_cfg = stage2_cfg.get("lora") if stage2_cfg.get("lora", {}).get("enabled") else None

    val2017 = {im["id"] for im in json.loads(Path(args.val2017_instances).read_text())["images"]}
    ques = json.loads(Path(args.questions).read_text())["questions"]
    anns = {a["question_id"]: a for a in json.loads(Path(args.annotations).read_text())["annotations"]}

    # keep ONLY val2017 images -> clean, disjoint from train2017
    rows = [q for q in ques if q["image_id"] in val2017]
    print(f"[vqav2] {len(ques)} val questions -> {len(rows)} on val2017 images (clean)")
    rng = random.Random(args.seed); rng.shuffle(rows)
    if args.num_questions and args.num_questions < len(rows):
        rows = rows[: args.num_questions]

    image_root = Path(args.image_root)

    # ---- resume: incremental JSONL of per-question predictions ----
    preds_path = Path("outputs/predictions") / f"vqav2_{args.condition}.jsonl"
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_store = {}
    if preds_path.exists():
        for line in preds_path.open():
            try:
                r = json.loads(line); preds_store[r["qid"]] = r["pred"]
            except Exception:
                pass
    todo = [q for q in rows if q["question_id"] not in preds_store]
    print(f"[vqav2] resume: {len(preds_store)} done, {len(todo)} to do")

    if todo:
        vlm = _build_vlm(args.vlm_checkpoint, enc_cfg, proj_cfg, llm_cfg, lora_cfg)
        gen_kwargs = {"do_sample": False, "num_beams": 1, "max_new_tokens": 10}
        with torch.no_grad(), preds_path.open("a") as fout:
            for i in tqdm(range(0, len(todo), args.batch_size), desc=f"vqav2 {args.condition}"):
                batch = todo[i:i + args.batch_size]
                images = [load_image(image_root / f"{q['image_id']:012d}.jpg") for q in batch]
                prompts = [_format_prompt(PROMPT.format(q=q["question"])) for q in batch]
                preds = vlm.generate(images, prompts, **gen_kwargs)
                for q, pred in zip(batch, preds):
                    preds_store[q["question_id"]] = pred.strip()
                    fout.write(json.dumps({"qid": q["question_id"], "pred": pred.strip()}) + "\n")
                fout.flush()
    else:
        print(f"[vqav2] {args.condition} already complete -> re-scoring only")

    total = 0.0
    n = 0
    for q in rows:
        pred = preds_store.get(q["question_id"])
        if pred is None:
            continue
        gt = [a["answer"] for a in anns[q["question_id"]]["answers"]]
        total += vqa_accuracy(pred, gt)
        n += 1
    acc = total / n if n else 0.0
    out = Path(args.out_dir) / f"vqav2_{args.condition}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"condition": args.condition, "image_split": "coco_val2017",
                               "n_questions": n, "vqa_accuracy": acc}, indent=2))
    snapshot_run_metadata({"condition": args.condition, "checkpoint": args.vlm_checkpoint,
                           "args": vars(args)}, Path(args.out_dir) / f"vqav2_{args.condition}")
    print(f"[ok] {args.condition} VQAv2(val2017) accuracy = {acc:.4f} on {n} questions -> {out}")


if __name__ == "__main__":
    main()
