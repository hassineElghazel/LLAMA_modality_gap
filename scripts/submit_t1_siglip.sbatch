#!/bin/bash
#SBATCH --job-name=t1-siglip
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/t1_siglip_%j.out
#SBATCH --error=outputs/logs/t1_siglip_%j.err

# ---------------------------------------------------------------------------
# TIER 1 / job 5.  SigLIP AS A SECOND, INDEPENDENT JUDGE.
#
# WHY.  Every reference-free number in the paper is CLIPScore, i.e. CLIP
# features judged by a CLIP scorer. SigLIP differs in objective, training data,
# tokenizer and architecture, so agreement is a genuine circularity check.
# scripts/15b_siglip_score.py has only ever been run on the EXCLUDED DOCCI arm;
# it has never touched the conditions the paper actually reports.
#
# NOT a truncation fix: SigLIP caps at 64 text tokens, WORSE than CLIP's 77.
# Present it as an independence check only.
#
# Scores captions already on disk. No generation. Minutes, not hours --
# clipscore jobs median 0.02 h. 2 h requested is almost all model-load margin.
#
#   sbatch scripts/submit_t1_siglip.sbatch
# ---------------------------------------------------------------------------

set -Eeuo pipefail
echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || true)

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/metrics

# dd256 tags: captions_<tag>.json must exist for each
CONDS="${CONDS:-C3pinr_dd256 Cloc_dd256 Cscale1500_dd256 Crank15_dd256 Corient_dd256 Clocorient_dd256}"

for c in $CONDS; do
  f="outputs/predictions/captions_${c}.json"
  if [ ! -f "$f" ]; then echo "[skip] $c (no $f)"; continue; fi
  echo "==================== SigLIP  $c ===================="
  python scripts/15b_siglip_score.py --condition "$c" \
    || echo "[siglip] $c FAILED (continuing)" >&2
done

echo "[siglip] complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[siglip] results: outputs/metrics/siglip_*.json"
