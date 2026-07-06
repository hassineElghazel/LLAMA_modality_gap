"""GQA compositional-reasoning eval, with an optional CLEAN filter that drops any
GQA image also in COCO train2017 (the LLaVA training images).

GQA uses Visual Genome images; some are COCO images. To keep it disjoint from
training, pass --vg-image-data (VG image_data.json, has coco_id per image) and
--train-image-ids (COCO train2017 ids) -> questions whose image's coco_id is in
train2017 are removed. Without those, it runs on the full set with a warning
(Q&A is still held out, but image-level overlap is possible).

Needs:
  data/gqa/testdev_balanced_questions.json   (GQA questions)
  data/gqa/images/<imageId>.jpg              (GQA images)

Usage:
    python scripts/24_eval_gqa.py --condition C3pinr \
        --vlm-checkpoint outputs/checkpoints/stage2_vlm_C3pinr.pt --num-questions 5000
"""
from __future__ import annotations

import argparse
import json
import random
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
_ARTICLES = {"a", "an", "the"}


def norm(s: str) -> str:
    s = "".join(ch if ch.isalnum() or ch == " " else " " for ch in s.strip().lower())
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def correct(pred: str, gt: str) -> bool:
    """GQA official = exact match on the normalized answer. We take the model's
    first line (in case it over-generates) and allow a trailing clause after the
    exact answer phrase (e.g. gt 'dog' vs 'dog on the couch'), but NOT a mere
    substring match anywhere (which would over-credit)."""
    p = norm(pred.split("\n")[0])
    g = norm(gt)
    return p == g or (bool(g) and p.startswith(g + " "))


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


def _excluded_images(vg_image_data: str | None, train_ids: str | None) -> set:
    """GQA imageIds (VG ids) whose coco_id is in COCO train2017 -> exclude.
    Returns empty (no filter) if either file is missing -- GQA testdev uses
    non-VG 'n'-images so nothing maps anyway, and it's clean by provenance."""
    if (not vg_image_data or not train_ids
            or not Path(vg_image_data).exists() or not Path(train_ids).exists()):
        return set()
    train = set()
    tj = json.loads(Path(train_ids).read_text())
    imgs = tj["images"] if isinstance(tj, dict) and "images" in tj else tj
    for im in imgs:
        train.add(int(im["id"]) if isinstance(im, dict) else int(im))
    excl = set()
    for e in json.loads(Path(vg_image_data).read_text()):
        cid = e.get("coco_id")
        if cid is not None and int(cid) in train:
            excl.add(str(e["image_id"]))
    return excl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True)
    p.add_argument("--vlm-checkpoint", required=True)
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--projector-config", default="configs/projector.yaml")
    p.add_argument("--llm-config", default="configs/llm.yaml")
    p.add_argument("--stage2-config", default="configs/training_stage2.yaml")
    p.add_argument("--questions", default="data/gqa/testdev_balanced_questions.json")
    p.add_argument("--image-root", default="data/gqa/images")
    p.add_argument("--vg-image-data", default="data/gqa/image_data.json",
                   help="VG image_data.json (coco_id mapping) for the clean filter")
    p.add_argument("--train-image-ids", default="data/coco/annotations/instances_train2017.json",
                   help="COCO train2017 (or captions_train2017) to exclude by coco_id")
    p.add_argument("--num-questions", type=int, default=5000)
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

    qs = json.loads(Path(args.questions).read_text())
    rows = [{"qid": k, "imageId": str(v["imageId"]), "question": v["question"], "answer": v["answer"]}
            for k, v in qs.items()]
    excl = _excluded_images(args.vg_image_data, args.train_image_ids)
    if excl:
        before = len(rows)
        rows = [r for r in rows if r["imageId"] not in excl]
        print(f"[gqa] CLEAN filter: removed {before-len(rows)} questions on COCO-train2017 images")
    else:
        print("[gqa] no ID filter applied -- testdev uses non-VG 'n'-images "
              "(clean by provenance; verified 0 real content overlap).")
    rng = random.Random(args.seed); rng.shuffle(rows)
    if args.num_questions and args.num_questions < len(rows):
        rows = rows[: args.num_questions]
    print(f"[gqa] evaluating {len(rows)} questions")

    image_root = Path(args.image_root)

    # ---- resume: incremental JSONL of per-question predictions ----
    preds_path = Path("outputs/predictions") / f"gqa_{args.condition}.jsonl"
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_store = {}
    if preds_path.exists():
        for line in preds_path.open():
            try:
                r = json.loads(line); preds_store[r["qid"]] = (r["pred"], r["gt"])
            except Exception:
                pass
    todo = [r for r in rows if r["qid"] not in preds_store]
    print(f"[gqa] resume: {len(preds_store)} done, {len(todo)} to do")

    if todo:
        vlm = _build_vlm(args.vlm_checkpoint, enc_cfg, proj_cfg, llm_cfg, lora_cfg)
        gen_kwargs = {"do_sample": False, "num_beams": 1, "max_new_tokens": 10}
        with torch.no_grad(), preds_path.open("a") as fout:
            for i in tqdm(range(0, len(todo), args.batch_size), desc=f"gqa {args.condition}"):
                batch = todo[i:i + args.batch_size]
                images = [load_image(image_root / f"{r['imageId']}.jpg") for r in batch]
                prompts = [_format_prompt(PROMPT.format(q=r["question"])) for r in batch]
                preds = vlm.generate(images, prompts, **gen_kwargs)
                for r, pred in zip(batch, preds):
                    preds_store[r["qid"]] = (pred.strip(), r["answer"])
                    fout.write(json.dumps({"qid": r["qid"], "pred": pred.strip(), "gt": r["answer"]}) + "\n")
                fout.flush()
    else:
        print(f"[gqa] {args.condition} already complete -> re-scoring only")

    n_correct = sum(int(correct(p, g)) for p, g in preds_store.values())
    n = len(preds_store)
    acc = n_correct / n if n else 0.0
    out = Path(args.out_dir) / f"gqa_{args.condition}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"condition": args.condition, "clean_filtered": bool(excl),
                               "n_questions": n, "accuracy": acc}, indent=2))
    snapshot_run_metadata({"condition": args.condition, "checkpoint": args.vlm_checkpoint,
                           "args": vars(args)}, Path(args.out_dir) / f"gqa_{args.condition}")
    print(f"[ok] {args.condition} GQA accuracy = {acc:.4f} on {n} questions "
          f"(clean={bool(excl)}) -> {out}")


if __name__ == "__main__":
    main()
