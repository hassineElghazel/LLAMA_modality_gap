#!/bin/bash
#SBATCH --job-name=mllm-vlmeval-C1
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=22:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/logs/vlmeval_C1_%j.out
#SBATCH --error=outputs/logs/vlmeval_C1_%j.err

# VLMEvalKit benchmarks for C1 (Stage-2 LoRA from random connector).
# 11 benchmarks: MME, MMStar, ScienceQA_VAL, RealWorldQA, MMMU_DEV_VAL, MMMU_Pro,
# VisuLogic, LogicVista, CRPE, POPE, HallusionBench.
# Each benchmark is wrapped in try/except inside 10_run_vlmevalkit.py, so a
# single failing dataset records {"error": ...} and the job continues.

set -Eeuo pipefail

CONDITION="C1_stage2"
CHECKPOINT="outputs/checkpoints/stage2_vlm_C1.pt"

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
Log: \`outputs/logs/vlmeval_C1_${SLURM_JOB_ID}.err\`"
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/metrics

# --- GPU sanity check ---
echo "[slurm] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
python -c "
import sys, torch
ok = torch.cuda.is_available() and torch.cuda.device_count() >= 1
print(f'[torch] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')
sys.exit(0 if ok else 1)
" || { echo "[slurm] CUDA init failed" >&2; exit 1; }

if [[ ! -e "${CHECKPOINT}" ]]; then
    echo "[preflight] missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi
echo "[preflight] checkpoint OK: ${CHECKPOINT}"

tg_notify "🚀 *VLMEvalKit ${CONDITION}* started (job \`${SLURM_JOB_ID}\`)
ckpt: \`${CHECKPOINT}\`
11 benchmarks; per-benchmark errors are tolerated."

python scripts/10_run_vlmevalkit.py \
    --condition "${CONDITION}" \
    --vlm-checkpoint "${CHECKPOINT}"

tg_notify "✅ *VLMEvalKit ${CONDITION}* complete (job \`${SLURM_JOB_ID}\`)
Aggregate: \`outputs/metrics/vlmevalkit_${CONDITION}.json\`"

echo "[slurm] ${CONDITION} VLMEvalKit complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
