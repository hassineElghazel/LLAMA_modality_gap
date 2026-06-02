#!/bin/bash
#SBATCH --job-name=mllm-chameleon-gap
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/logs/chameleon_gap_%j.out
#SBATCH --error=outputs/logs/chameleon_gap_%j.err

# Early-fusion modality-gap trajectory on Chameleon-7B (the late-fusion contrast).
# Measures the gap metrics at all 33 hidden states (embeddings + 32 layers) in two
# modes: independent (LLaVA-comparable) and fused (true early fusion).
#
# ISOLATED: reads only Chameleon (HF cache, GATED — accept the license + set
# HF_TOKEN first), COCO, the diagnostic manifest, and metrics/gap_C3_stage2.json
# (overlay, read-only). Writes ONLY under outputs/{embeddings,metrics,figures}/chameleon/.
# Per-mode idempotence: a mode whose trajectory JSON already exists is skipped, so
# a killed job continues per-mode on resubmit.

set -Eeuo pipefail

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
    tg_notify "❌ *chameleon-gap FAILED* (job \`${SLURM_JOB_ID}\`)
Line: \`${line_no}\`  Exit: \`${exit_code}\`  Node: \`$(hostname)\`
Resubmit \`sbatch scripts/submit_chameleon_gap.sbatch\` to resume per-mode.
Log: \`outputs/logs/chameleon_gap_${SLURM_JOB_ID}.err\`"
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs \
    outputs/embeddings/chameleon outputs/metrics/chameleon outputs/figures/chameleon

# --- GPU sanity check ---
echo "[slurm] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
python -c "
import sys, torch
ok = torch.cuda.is_available() and torch.cuda.device_count() >= 1
print(f'[torch] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')
sys.exit(0 if ok else 1)
" || { echo "[slurm] CUDA init failed" >&2; exit 1; }

# --- preflight: read-only inputs present ---
for f in \
    data/coco/annotations/captions_val2017.json \
    data/coco/val2017; do
    if [[ ! -e "$f" ]]; then
        echo "[preflight] missing required input: $f" >&2
        tg_notify "❌ *chameleon-gap* — missing input \`$f\`."
        exit 1
    fi
done
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[preflight] HF_TOKEN unset — facebook/chameleon-7b is gated" >&2
    tg_notify "❌ *chameleon-gap* — HF_TOKEN unset (model is gated)."
    exit 1
fi
echo "[preflight] inputs present"

tg_notify "🚀 *chameleon-gap* started (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
Early-fusion gap trajectory, N=500, modes={independent, fused}, 33 layers each.
Isolated outputs under \`outputs/*/chameleon/\`."

python scripts/13_chameleon_gap.py --config configs/chameleon.yaml

tg_notify "✅ *chameleon-gap COMPLETE* (job \`${SLURM_JOB_ID}\`)
Trajectories: \`outputs/metrics/chameleon/gap_chameleon_{independent,fused}_trajectory.json\`
Plots: \`outputs/figures/chameleon/gap_trajectory_{independent,fused}.png\`"
echo "[slurm] chameleon-gap complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
