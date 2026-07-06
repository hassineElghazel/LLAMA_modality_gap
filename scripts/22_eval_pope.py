"""POPE object-hallucination eval on COCO val2017 (guaranteed NOT in training).

Training images are COCO train2017; the standard POPE uses val2014, which
train2017 largely absorbed -> possible contamination. We instead BUILD POPE on
val2017 (the held-out split, disjoint from train2017), so the eval images are
provably unseen. Questions ("Is there a <object> in the image?") are synthetic
and never appear in LLaVA-150K.

Three negative-sampling splits (Li et al. 2023): random / popular / adversarial.
Metrics per split: accuracy, precision, recall, F1, yes-ratio.

Usage:
    python scripts/22_eval_pope.py --condition C3pinr \
        --vlm-checkpoint outputs/checkpoints/stage2_vlm_C3pinr.pt
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
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

PROMPT = "<image>\nIs there a {obj} in the image? Please answer with a single word: yes or no."


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


def build_pope(instances_path: Path, image_root: Path, n_images: int, k: int, seed: int):
    """Return (questions, per_image_objs). questions = list of dicts with
    image_id, file, obj, label(1/0), split."""
    inst = json.loads(instances_path.read_text())
    id2name = {c["id"]: c["name"].strip().lower() for c in inst["categories"]}
    all_cats = sorted(set(id2name.values()))
    file_of = {im["id"]: im["file_name"] for im in inst["images"]}
    gt = defaultdict(set)
    for a in inst["annotations"]:
        gt[a["image_id"]].add(id2name[a["category_id"]])
    # global frequency (for 'popular') and co-occurrence (for 'adversarial')
    freq = Counter()
    cooc = defaultdict(Counter)
    for iid, objs in gt.items():
        for c in objs:
            freq[c] += 1
        for a in objs:
            for b in objs:
                if a != b:
                    cooc[a][b] += 1
    popular = [c for c, _ in freq.most_common()]

    rng = random.Random(seed)
    imgs = [i for i in gt if len(gt[i]) >= 1 and i in file_of]
    rng.shuffle(imgs)
    imgs = imgs[:n_images]

    Q = []
    for iid in imgs:
        G = gt[iid]; kk = min(k, len(G))
        pos = rng.sample(sorted(G), kk)
        absent = [c for c in all_cats if c not in G]
        neg_random = rng.sample(absent, min(kk, len(absent)))
        neg_pop = [c for c in popular if c not in G][:kk]
        adv_rank = sorted(absent, key=lambda c: sum(cooc[g][c] for g in G), reverse=True)
        neg_adv = adv_rank[:kk]
        for obj in pos:
            for sp in ("random", "popular", "adversarial"):
                Q.append(dict(image_id=iid, file=file_of[iid], obj=obj, label=1, split=sp))
        for sp, negs in (("random", neg_random), ("popular", neg_pop), ("adversarial", neg_adv)):
            for obj in negs:
                Q.append(dict(image_id=iid, file=file_of[iid], obj=obj, label=0, split=sp))
    for q in Q:
        q["qid"] = f"{q['image_id']}_{q['obj']}_{q['split']}"
    rng.shuffle(Q)
    return Q, imgs


def parse_yes(ans: str) -> int:
    a = ans.strip().lower()
    return 1 if a.startswith("yes") else 0


def score(rows):
    out = {}
    for sp in ("random", "popular", "adversarial", "overall"):
        rs = rows if sp == "overall" else [r for r in rows if r["split"] == sp]
        if not rs:
            continue
        tp = sum(1 for r in rs if r["label"] == 1 and r["pred"] == 1)
        fp = sum(1 for r in rs if r["label"] == 0 and r["pred"] == 1)
        fn = sum(1 for r in rs if r["label"] == 1 and r["pred"] == 0)
        tn = sum(1 for r in rs if r["label"] == 0 and r["pred"] == 0)
        n = len(rs)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[sp] = dict(accuracy=(tp + tn) / n, precision=prec, recall=rec, f1=f1,
                       yes_ratio=(tp + fp) / n, n=n)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True)
    p.add_argument("--vlm-checkpoint", required=True)
    p.add_argument("--encoders-config", default="configs/encoders.yaml")
    p.add_argument("--projector-config", default="configs/projector.yaml")
    p.add_argument("--llm-config", default="configs/llm.yaml")
    p.add_argument("--stage2-config", default="configs/training_stage2.yaml")
    p.add_argument("--instances", default="data/coco/annotations/instances_val2017.json")
    p.add_argument("--image-root", default="data/coco/val2017")
    p.add_argument("--num-images", type=int, default=500)
    p.add_argument("--k", type=int, default=3, help="pos (and neg per split) per image")
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

    image_root = Path(args.image_root)
    Q, imgs = build_pope(Path(args.instances), image_root, args.num_images, args.k, args.seed)
    print(f"[pope] {len(imgs)} val2017 images (clean, NOT in train2017) | {len(Q)} questions")

    # ---- resume: incremental JSONL of per-question predictions ----
    preds_path = Path("outputs/predictions") / f"pope_{args.condition}.jsonl"
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    done, recs = set(), []
    if preds_path.exists():
        for line in preds_path.open():
            try:
                r = json.loads(line); done.add(r["qid"]); recs.append(r)
            except Exception:
                pass
    todo = [q for q in Q if q["qid"] not in done]
    print(f"[pope] resume: {len(done)} done, {len(todo)} to do")

    if todo:
        vlm = _build_vlm(args.vlm_checkpoint, enc_cfg, proj_cfg, llm_cfg, lora_cfg)
        gen_kwargs = {"do_sample": False, "num_beams": 1, "max_new_tokens": 8}
        with torch.no_grad(), preds_path.open("a") as fout:
            for i in tqdm(range(0, len(todo), args.batch_size), desc=f"pope {args.condition}"):
                batch = todo[i:i + args.batch_size]
                images = [load_image(image_root / r["file"]) for r in batch]
                prompts = [_format_prompt(PROMPT.format(obj=r["obj"])) for r in batch]
                ans = vlm.generate(images, prompts, **gen_kwargs)
                for r, a in zip(batch, ans):
                    rec = {"qid": r["qid"], "label": r["label"], "split": r["split"],
                           "answer": a.strip(), "pred": parse_yes(a)}
                    recs.append(rec)
                    fout.write(json.dumps(rec) + "\n")
                fout.flush()
    else:
        print(f"[pope] {args.condition} already complete -> re-scoring only")

    metrics = score(recs)
    out = Path(args.out_dir) / f"pope_{args.condition}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"condition": args.condition, "gt_split": "coco_val2017",
                               "n_images": len(imgs), "n_questions": len(Q),
                               "metrics": metrics}, indent=2))
    snapshot_run_metadata({"condition": args.condition, "checkpoint": args.vlm_checkpoint,
                           "args": vars(args)}, Path(args.out_dir) / f"pope_{args.condition}")
    print(f"[ok] {args.condition} POPE overall: "
          f"acc={metrics['overall']['accuracy']:.3f} f1={metrics['overall']['f1']:.3f} "
          f"yes={metrics['overall']['yes_ratio']:.3f}  -> {out}")


if __name__ == "__main__":
    main()
