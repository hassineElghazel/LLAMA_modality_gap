#!/bin/bash
#SBATCH --job-name=mllm-c3
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=20:00:00
#SBATCH --output=outputs/logs/c3_%j.out
#SBATCH --error=outputs/logs/c3_%j.err

set -Eeuo pipefail

echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/.telegram_env 2>/dev/null || true

# --- Telegram notification helpers ---
# No-op if TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID aren't exported by ~/.telegram_env.
# `|| true` on the curl call keeps a transient network blip from killing the job.
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
    tg_notify "❌ *C3 FAILED* (job \`${SLURM_JOB_ID}\`)
Line: \`${line_no}\`  Exit: \`${exit_code}\`
Node: \`$(hostname)\`
Log: \`outputs/logs/c3_${SLURM_JOB_ID}.err\`"
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/checkpoints outputs/predictions outputs/metrics

tg_notify "🚀 *C3 started* (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
Plan: reuse C2 connector → measure → Stage 2 (450 steps) → measure
Skipping downstream; will batch all conditions later."

# --- GPU sanity check ---
echo "[slurm] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
python -c "
import sys, torch
ok = torch.cuda.is_available() and torch.cuda.device_count() >= 1
print(f'[torch] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')
sys.exit(0 if ok else 1)
" || { echo "[slurm] CUDA init failed" >&2; exit 1; }

# --- C3 pipeline: reuse C2's Stage-1 connector, then Stage 2 LoRA, then measure ---
#
# 2x2 factorial design: C2 and C3 share the contrastive Stage-1 connector by
# construction, so we symlink stage1_connector_C2.pt into C3's slot rather
# than retraining Stage 1 (which would be the same artifact with a different
# seed at best, redundant compute at worst).
#
# --max-steps 450 matches C1's effective training length so the C1-vs-C3
# comparison is apples-to-apples on captioning loss exposure.
# --skip-downstream defers captioning so all four conditions are evaluated
# together in a separate batch job for apples-to-apples decoding.

echo "[c3] starting train + measure pipeline: reuse-C2-connector -> measure -> Stage2 -> measure"
echo "[c3] downstream captioning intentionally skipped; will be run for all 4 conditions in a separate batch"

python scripts/run_condition.py \
    --condition C3 \
    --from-connector outputs/checkpoints/stage1_connector_C2.pt \
    --max-steps 450 \
    --skip-downstream

tg_notify "✅ *C3 complete* (job \`${SLURM_JOB_ID}\`)
Stage 2 checkpoint: \`outputs/checkpoints/stage2_vlm_C3.pt\`
Metrics: \`outputs/metrics/{C3_stage1,C3_stage2}/\`
Next: run all-conditions downstream eval."

echo "[slurm] C3 complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
