#!/bin/bash
#SBATCH --job-name=mllm-ood-eval
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/oodeval_%j.out
#SBATCH --error=outputs/logs/oodeval_%j.err
set -Eeuo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap

# ---- GPU allocation sanity banner (diagnose "No CUDA GPUs are available") ----
echo "[node] host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[node] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}  SLURM_GPUS=${SLURM_GPUS:-<unset>}"

# ---- assert the allocated GPU is HEALTHY ------------------------------------
# baldur has a faulted card (PCI 60:00.0 -> NVML "Unknown Error"). A co-allocated
# dead GPU aborts cuInit() for the whole process, so masking it is not enough --
# we must simply not be given it. If we landed on it, fail fast (a sentinel job
# should be holding the bad GPU; see submit_gpu_hold.sbatch).
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "[fatal] allocated GPU is faulted on $(hostname) (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)." >&2
  echo "        Keep the sentinel (submit_gpu_hold.sbatch) running and resubmit." >&2
  exit 1
fi
nvidia-smi -L

# ---------------------------------------------------------------------------
# OUT-OF-DISTRIBUTION grounding eval on 100 NoCaps (Open Images) images.
#
# NoCaps is disjoint from COCO -> these checkpoints (trained on LLaVA-Instruct /
# COCO train2017) have never seen this domain. Only two conditions, only one
# metric: does Cloc's location-closure CLIPScore win survive off-distribution?
#
#   baseline : C3pinr  (gap left open)
#   treatment: Cloc    (location closed, rank pinned)
#
# Same detail prompt + 256-tok decode as dd256 (configs/ood_eval.yaml), so the
# _ood CLIPScore is directly comparable to the in-distribution _dd256 CLIPScore.
#
# Prereq (run once, needs the NoCaps annotation file or a local image folder):
#   python scripts/prep_ood_eval.py --nocaps-json <nocaps_val_4500_captions.json> \
#       --out-root data/nocaps --n 300 --seed 42
#
# Inference only. Run from repo root: sbatch scripts/submit_ood_eval.sbatch
# ---------------------------------------------------------------------------

CFG=configs/ood_eval.yaml
CKPT=outputs/checkpoints
CLIP_MODEL="${CLIP_MODEL:-/nfs/home/elghazel/clip_b32}"
CONDS="Cloc C3pinr"   # Cloc first (treatment), then baseline

if [ ! -f data/nocaps/ood_eval_300.json ]; then
  echo "[fatal] data/nocaps/ood_eval_300.json missing -- run scripts/prep_ood_eval.py first" >&2
  exit 1
fi

for c in $CONDS; do
  ck="$CKPT/stage2_vlm_${c}.pt"
  if [ ! -f "$ck" ]; then echo "[skip] $c (no checkpoint $ck)"; continue; fi
  tag="${c}_ood"
  echo "==================== DESCRIBE (OOD)  $tag ===================="
  python scripts/08_run_captioning.py --config "$CFG" --condition "$tag" --vlm-checkpoint "$ck"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
      --config "$CFG" --condition "$tag" --clip-model "$CLIP_MODEL" \
      || echo "[ood] $tag clipscore FAILED (rerun 15_clipscore.py --config $CFG --condition $tag)"
done

# ---- CLIPScore summary: baseline vs Cloc on the OOD set ----
python - <<'PY'
import json, os
print("\n%-16s %10s" % ("condition (OOD)", "CLIPScore"))
print("-" * 28)
vals = {}
for c in ("Cloc_ood", "C3pinr_ood"):
    f = f"outputs/metrics/clipscore_{c}.json"
    cs = json.load(open(f))["CLIPScore"] if os.path.exists(f) else None
    vals[c] = cs
    print("%-16s %10s" % (c, ("%.4f" % cs) if cs is not None else "--"))
if all(v is not None for v in vals.values()):
    d = vals["Cloc_ood"] - vals["C3pinr_ood"]
    print("-" * 28)
    print("%-16s %+10.4f  (Cloc - C3pinr)" % ("delta", d))
PY

echo "[ood-eval] done"
