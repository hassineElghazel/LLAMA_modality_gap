#!/bin/bash
#SBATCH --job-name=mllm-c1
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=36:00:00
#SBATCH --output=outputs/logs/c1_%j.out
#SBATCH --error=outputs/logs/c1_%j.err

set -euo pipefail

echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

# CUDA allocator: expandable segments reduces fragmentation OOMs on 11 GB cards.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load Telegram credentials if present (set these up once, see README)
source ~/.telegram_env 2>/dev/null || true

cd ~/LLAMA_modality_gap

# Ensure log dir exists (SLURM creates the file but not the directory)
mkdir -p outputs/logs

echo "[slurm] running C1 — Stage 2 LoRA from random connector on LLaVA-Instruct-150K"
python scripts/run_condition.py --condition C1

echo "[slurm] C1 complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
