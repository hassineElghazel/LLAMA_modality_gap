#!/bin/bash
#SBATCH --job-name=mllm-eval
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --output=outputs/logs/eval_%j.out
#SBATCH --error=outputs/logs/eval_%j.err

set -euo pipefail

echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Telegram notifications (optional)
source ~/.telegram_env 2>/dev/null || true

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/predictions outputs/metrics

# --- GPU sanity check ---
echo "[slurm] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
python -c "
import sys, torch
ok = torch.cuda.is_available() and torch.cuda.device_count() >= 1
print(f'[torch] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')
sys.exit(0 if ok else 1)
" || { echo "[slurm] CUDA init failed" >&2; exit 1; }

# --- Captioning + scoring for C0, C1, C2 ---
# Each block runs generation then scoring. We do not `set +e` — if a stage
# fails the whole job exits so the surviving outputs are unambiguous.

echo "[eval] === C1_stage2 (Stage 2 LoRA from random connector) ==="
python scripts/08_run_captioning.py \
    --condition C1_stage2 \
    --vlm-checkpoint outputs/checkpoints/stage2_vlm_C1.pt
python scripts/09_score_captions.py --condition C1_stage2 --no-spice

echo "[eval] === C0_random (untrained connector, no LoRA) ==="
python scripts/08_run_captioning.py \
    --condition C0_random \
    --vlm-checkpoint random
python scripts/09_score_captions.py --condition C0_random --no-spice

echo "[eval] === C2_stage1 (Stage 1 contrastive connector, no LoRA) ==="
python scripts/08_run_captioning.py \
    --condition C2_stage1 \
    --vlm-checkpoint outputs/checkpoints/stage1_connector_C2.pt
python scripts/09_score_captions.py --condition C2_stage1 --no-spice

echo "[slurm] all three conditions complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
