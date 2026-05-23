#!/bin/bash
#SBATCH --job-name=mllm-c2
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/c2_%j.out
#SBATCH --error=outputs/logs/c2_%j.err

set -euo pipefail

echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

# Load Telegram credentials if present (set these up once, see README)
source ~/.telegram_env 2>/dev/null || true

cd ~/LLAMA_modality_gap

# Ensure log dir exists (SLURM creates the file but not the directory)
mkdir -p outputs/logs

echo "[slurm] running C2 — full Bunny-v1.1 (~750k pairs, no subset limit)"
python scripts/run_condition.py --condition C2 --subset-size 750000

echo "[slurm] C2 complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
