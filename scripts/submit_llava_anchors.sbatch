#!/bin/bash
#SBATCH --job-name=llava-anchors
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/llava_anchors_%j.out
#SBATCH --error=outputs/logs/llava_anchors_%j.err

# Precompute the FROZEN geometry anchors (mu_y + trace_x) on DOCCI-train with the
# PRETRAINED LLaVA-1.5. Must run before any 3-arm training job.
set -Eeuo pipefail
echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || true)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/anchors_docci

nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
test -f data/docci/train_manifest.json || { echo "[anchors] FATAL: run submit_docci_prep.sbatch first" >&2; exit 1; }

HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} \
python scripts/25_precompute_llava_anchors.py \
    --manifest data/docci/train_manifest.json \
    --model-id "${MODEL_ID:-llava-hf/llava-1.5-7b-hf}" \
    --out-dir outputs/anchors_docci

echo "[anchors] $(cat outputs/anchors_docci/anchors.json)"
echo "[slurm] llava-anchors complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
