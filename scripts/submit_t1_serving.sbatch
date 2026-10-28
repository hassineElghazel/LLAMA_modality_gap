#!/bin/bash
#SBATCH --job-name=t1-serving
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/t1_serving_%j.out
#SBATCH --error=outputs/logs/t1_serving_%j.err

# ---------------------------------------------------------------------------
# TIER 1 / job 1.  SERVING COST.  The venue-fit item.
#
# The paper contains no millisecond and no megabyte. This produces both, for
# the baseline and the method. EVERY COLUMN IS EXPECTED TO BE IDENTICAL -- the
# geometry terms act only during training, so the deployed artifact is the same
# shape. That identity is the result; do not treat a null delta as a bug.
#
# Budget: 2 conditions x (10 warmup + 500 timed) brief captions at a measured
# ~2.96 s/image (from cap_82397: 2x5000 brief captions in 8.23 h) = ~0.85 h,
# plus two model loads. 4 h requested leaves ~4x headroom.
#
#   sbatch scripts/submit_t1_serving.sbatch
#   N_IMAGES=200 sbatch scripts/submit_t1_serving.sbatch   # faster, wider p95
# ---------------------------------------------------------------------------

set -Eeuo pipefail
echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || true)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/metrics

N_IMAGES="${N_IMAGES:-500}"
WARMUP="${WARMUP:-10}"
CONDS="${CONDS:-C3pinr Cloc}"

python -c "
import sys, torch
if not torch.cuda.is_available(): print('[preflight] no CUDA'); sys.exit(1)
x = torch.randn(2048,2048,device='cuda'); _=(x@x).sum().item(); torch.cuda.synchronize()
print('[preflight] GPU op OK on', torch.cuda.get_device_name(0))
" || { echo '[serving] GPU preflight FAILED' >&2; exit 1; }

for c in $CONDS; do
  ck="outputs/checkpoints/stage2_vlm_${c}.pt"
  if [ ! -f "$ck" ]; then echo "[skip] $c (no $ck)"; continue; fi
  echo "==================== SERVING COST  $c ===================="
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/37_serving_cost.py \
      --condition "$c" --vlm-checkpoint "$ck" \
      --n-images "$N_IMAGES" --warmup "$WARMUP" \
    || echo "[serving] $c FAILED (continuing)" >&2
done

echo "[serving] complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[serving] results: outputs/metrics/serving_cost_*.json"
