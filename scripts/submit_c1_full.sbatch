#!/bin/bash
#SBATCH --job-name=mllm-c1-full
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=20:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/logs/c1_full_%j.out
#SBATCH --error=outputs/logs/c1_full_%j.err

# Full-epoch Stage-2 retrain for C1 (random connector + LoRA), into the isolated
# experiment dir outputs/checkpoints/full_epoch/. The original 9% checkpoint
# outputs/checkpoints/stage2_vlm_C1.pt is left untouched.
#
# Resume model: auto-resume from the latest checkpoint + manual resubmit. A full
# epoch (~4,929 steps) needs several sequential ~20h jobs on the RTX 2080 Ti;
# just re-run `sbatch scripts/submit_c1_full.sbatch` after each ends and it
# continues from where it stopped. The .done marker makes post-completion
# re-runs no-ops.

set -Eeuo pipefail

CKPT="outputs/checkpoints/full_epoch/stage2_vlm_C1.pt"
DONE="${CKPT}.done"

echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

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
    tg_notify "❌ *C1-full FAILED* (job \`${SLURM_JOB_ID}\`)
Line: \`${line_no}\`  Exit: \`${exit_code}\`  Node: \`$(hostname)\`
Resubmit \`sbatch scripts/submit_c1_full.sbatch\` to resume from latest checkpoint.
Log: \`outputs/logs/c1_full_${SLURM_JOB_ID}.err\`"
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/checkpoints/full_epoch outputs/logs/stage2_full

# --- already complete? ---
if [[ -f "${DONE}" ]]; then
    echo "[c1-full] ${DONE} present — training already complete; nothing to do."
    tg_notify "ℹ️ *C1-full* already complete (\`${DONE}\` exists). No-op."
    exit 0
fi

# --- GPU sanity check ---
echo "[slurm] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
python -c "
import sys, torch
ok = torch.cuda.is_available() and torch.cuda.device_count() >= 1
print(f'[torch] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')
sys.exit(0 if ok else 1)
" || { echo "[slurm] CUDA init failed" >&2; exit 1; }

# --- auto-resume from latest checkpoint if present ---
RESUME_ARG=()
RESUME_MSG="fresh start (step 0)"
if [[ -f "${CKPT}" ]]; then
    RESUME_ARG=(--resume "${CKPT}")
    STEP=$(python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu').get('step','?'))" "${CKPT}" 2>/dev/null || echo "?")
    RESUME_MSG="resuming from \`${CKPT}\` at step ${STEP}"
fi

tg_notify "🚀 *C1-full* started (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
${RESUME_MSG}
Full epoch (~4,929 steps), random connector + LoRA → \`${CKPT}\`"

echo "[c1-full] ${RESUME_MSG}"

# Full epoch: no --max-steps. Random connector (C1). Isolated config/output.
python scripts/06_train_stage2.py \
    --config configs/training_stage2_full.yaml \
    --init-connector random \
    --output-name stage2_vlm_C1.pt \
    ${RESUME_ARG[@]+"${RESUME_ARG[@]}"}

# Reached here only if the python run finished the full epoch (exit 0).
touch "${DONE}"
tg_notify "✅ *C1-full COMPLETE* (job \`${SLURM_JOB_ID}\`)
Full-epoch checkpoint: \`${CKPT}\`
Marker: \`${DONE}\`"
echo "[slurm] C1-full complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
