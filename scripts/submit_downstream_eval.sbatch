#!/bin/bash
#SBATCH --job-name=mllm-downstream
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=47:00:00
#SBATCH --output=outputs/logs/downstream_%j.out
#SBATCH --error=outputs/logs/downstream_%j.err
set -Eeuo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap

# ---------------------------------------------------------------------------
# Downstream (in-distribution) evaluation of the pinned ABLATION set:
#   - POPE  : object hallucination, yes/no VQA -> built on COCO val2017
#             (CLEAN: val2017 is disjoint from the train2017 LLaVA images; the
#             standard POPE uses val2014, which train2017 absorbed -> we avoid it).
#   - GQA   : compositional reasoning VQA (VLMEvalKit). CAVEAT below.
# Inference only, no training. Run from the repo root: sbatch scripts/submit_downstream_eval.sbatch
# ---------------------------------------------------------------------------

CKPT=outputs/checkpoints
CONDS="C3pinr C5bp_lam0p1 Cscale1500 C4bp_lam0p9 Crank15 Cloc"

# ============================ POPE (clean, primary) ========================
for c in $CONDS; do
  ck="$CKPT/stage2_vlm_${c}.pt"
  if [ ! -f "$ck" ]; then echo "[skip] $c (no checkpoint $ck)"; continue; fi
  echo "==================== POPE  $c ===================="
  python scripts/22_eval_pope.py --condition "$c" --vlm-checkpoint "$ck" --num-images 500
done

# ---- POPE summary table across the ablation set ----
python - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob("outputs/metrics/pope_*.json")):
    d = json.load(open(f)); m = d["metrics"]["overall"]
    rows.append((d["condition"], m["accuracy"], m["f1"], m["yes_ratio"]))
print("\n%-16s %8s %8s %9s" % ("condition", "acc", "F1", "yes_ratio"))
for c, a, f1, y in rows:
    print("%-16s %8.3f %8.3f %9.3f" % (c, a, f1, y))
PY

# ============================ VQAv2 (clean, val2017 subset) ================
# Needs data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json + v2_mscoco_val2014_annotations.json
# (download once). Only val2017-image questions are kept -> disjoint from train2017.
if [ -f data/vqav2/v2_mscoco_val2014_annotations.json ]; then
  for c in $CONDS; do
    ck="$CKPT/stage2_vlm_${c}.pt"
    if [ ! -f "$ck" ]; then echo "[skip] $c"; continue; fi
    echo "==================== VQAv2  $c ===================="
    python scripts/23_eval_vqav2.py --condition "$c" --vlm-checkpoint "$ck" --num-questions 5000
  done
  python - <<'PY'
import json, glob
rows = [(json.load(open(f))["condition"], json.load(open(f))["vqa_accuracy"])
        for f in sorted(glob.glob("outputs/metrics/vqav2_*.json"))]
print("\n%-16s %10s" % ("condition", "VQAv2-acc"))
for c, a in rows: print("%-16s %10.4f" % (c, a))
PY
else
  echo "[vqav2] skipped -- download VQAv2 val files to data/vqav2/ first"
fi

# ============================ GQA (compositional reasoning) ================
# Needs data/gqa/testdev_balanced_questions.json + data/gqa/images/ .
# CLEAN filter: pass VG image_data.json + train2017 ids -> drops GQA images whose
# coco_id is in train2017. Without them it still runs (Q&A held out) but with a
# warning that image overlap is possible.
if [ -f data/gqa/testdev_balanced_questions.json ]; then
  for c in $CONDS; do
    ck="$CKPT/stage2_vlm_${c}.pt"
    if [ ! -f "$ck" ]; then echo "[skip] $c"; continue; fi
    echo "==================== GQA  $c ===================="
    python scripts/24_eval_gqa.py --condition "$c" --vlm-checkpoint "$ck" --num-questions 5000
  done
  python - <<'PY'
import json, glob
rows = [(json.load(open(f))["condition"], json.load(open(f))["accuracy"], json.load(open(f)).get("clean_filtered"))
        for f in sorted(glob.glob("outputs/metrics/gqa_*.json"))]
print("\n%-16s %10s %8s" % ("condition", "GQA-acc", "clean"))
for c, a, cl in rows: print("%-16s %10.4f %8s" % (c, a, cl))
PY
else
  echo "[gqa] skipped -- download GQA testdev + images to data/gqa/ first"
fi

echo "[downstream] done"
