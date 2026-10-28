#!/bin/bash
#SBATCH --job-name=t1-grid
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/t1_grid_%j.out
#SBATCH --error=outputs/logs/t1_grid_%j.err

# ---------------------------------------------------------------------------
# TIER 1 / job 2.  COMPLETE THE INTERVENTION GRID'S CLIPScore CELLS.
#
# WHY.  outputs/logs/intervene_geom_84849.out already holds the full decoder-free
# retrieval grid: mean_shift raises R@1 0.0022 -> 0.0326 peaking exactly at
# G_mu=0 and collapsing on overshoot, while random_shift at matched or larger
# magnitude stays flat at ~0.0020. That answers direction-vs-magnitude.
# BUT only 2 of 31 CLIPScore cells were ever computed, and they point the OTHER
# way (identity 0.2160 -> mean_shift a=0.5 0.1966). Reporting the grid with two
# generation cells invites a cherry-picking charge, which is about conduct and
# therefore worse than a methodological gap.
#
# This completes ONLY the mean_shift and random_shift rows (12 cells). The other
# three interventions are inert on retrieval and not worth generation time.
#
# EXPECTED RESULT: CLIPScore falls as retrieval rises. That is not a
# contradiction, it is the mechanism -- the benefit needs the connector to
# RELEARN under the constraint, not to be shifted at inference. It is the direct
# answer to "why not just shift the embeddings at serving time?".
#
# Budget: 12 cells x 300 images at a measured ~2.96 s/image = ~3.0 h.
# 8 h requested absorbs a slow node (baldur has run 1.6x slow before).
#
#   sbatch scripts/submit_t1_grid.sbatch
# ---------------------------------------------------------------------------

set -Eeuo pipefail
echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || true)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/metrics/intervene outputs/predictions/intervene

# same checkpoint + embedding condition as the original grid (job 84849),
# so the new cells are directly comparable to the two that already exist.
CKPT="${CKPT:-outputs/checkpoints/stage2_vlm_C3.pt}"
EMB="${EMB:-C3_stage2}"
SUBSET="${SUBSET:-300}"
DOSES="${DOSES:-0.25 0.5 0.75 1 1.25 1.5}"

test -f "$CKPT" || { echo "[grid] FATAL: $CKPT missing" >&2; exit 1; }

python -c "
import sys, torch
if not torch.cuda.is_available(): print('[preflight] no CUDA'); sys.exit(1)
x = torch.randn(2048,2048,device='cuda'); _=(x@x).sum().item(); torch.cuda.synchronize()
print('[preflight] GPU op OK on', torch.cuda.get_device_name(0))
" || { echo '[grid] GPU preflight FAILED' >&2; exit 1; }

echo "[grid] checkpoint=$CKPT emb=$EMB subset=$SUBSET"
echo "[grid] caption doses: $DOSES  (generation + CLIPScore at each)"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/16_intervene_geometry.py \
    --vlm-checkpoint "$CKPT" \
    --emb-condition "$EMB" \
    --subset-size "$SUBSET" \
    --interventions mean_shift random_shift \
    --caption-doses $DOSES \
    --geom-doses $DOSES

echo "[grid] complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
