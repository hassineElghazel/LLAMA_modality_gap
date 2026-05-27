#!/bin/bash
#SBATCH --job-name=mllm-vlmeval-C0
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=22:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/logs/vlmeval_C0_%j.out
#SBATCH --error=outputs/logs/vlmeval_C0_%j.err

# VLMEvalKit benchmarks for C0 (random connector, no LoRA — base LLaMA-2 only).
#
# This is the no-training-at-all baseline. The connector is freshly-initialised
# Kaiming weights, so the LLM receives noise-like visual tokens; benchmark
# scores are expected to be near-chance. Their value is anchoring "no Stage 1,
# no Stage 2" — without C0 numbers, we cannot claim either stage adds anything
# downstream.

set -Eeuo pipefail

CONDITION="C0_random"
CHECKPOINT="random"

echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) cond=${CONDITION} started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/.telegram_env 2>/dev/null || true

tg_notify() {
    local msg="$1"
    [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]] && return 0
    curl -fsS -X POST \
        "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=${msg}" \
        --data-urlencode "parse_mode=Markdown" \
        > /dev/null || true
}

on_error() {
    local exit_code=$?
    local line_no=$1
    tg_notify "❌ *VLMEvalKit ${CONDITION} FAILED* (job \`${SLURM_JOB_ID}\`)
Line: \`${line_no}\`  Exit: \`${exit_code}\`
Log: \`outputs/logs/vlmeval_C0_${SLURM_JOB_ID}.err\`"
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/metrics

echo "[slurm] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
python -c "
import sys, torch
ok = torch.cuda.is_available() and torch.cuda.device_count() >= 1
print(f'[torch] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')
sys.exit(0 if ok else 1)
" || { echo "[slurm] CUDA init failed" >&2; exit 1; }

# No checkpoint to preflight — C0 uses 'random' which builds a fresh projector.
# Seed for the random projector is controlled by configs/captioning.yaml::seed
# (and propagated through scripts/08; the vlmevalkit adapter does not re-seed,
# but the model is built once at construction so all benchmarks see the same
# random projector within this job).

tg_notify "🚀 *VLMEvalKit ${CONDITION}* started (job \`${SLURM_JOB_ID}\`)
ckpt: random (no Stage 1, no Stage 2)
Expected: near-chance scores; this is the no-training baseline."

python scripts/10_run_vlmevalkit.py \
    --condition "${CONDITION}" \
    --vlm-checkpoint "${CHECKPOINT}"

tg_notify "✅ *VLMEvalKit ${CONDITION}* complete (job \`${SLURM_JOB_ID}\`)
Aggregate: \`outputs/metrics/vlmevalkit_${CONDITION}.json\`"

echo "[slurm] ${CONDITION} VLMEvalKit complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
