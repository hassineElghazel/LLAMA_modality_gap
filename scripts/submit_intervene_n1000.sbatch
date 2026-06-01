#!/bin/bash
#SBATCH --job-name=mllm-intervene-n1000
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/logs/intervene_n1000_%j.out
#SBATCH --error=outputs/logs/intervene_n1000_%j.err

# Confirmation run: N=1000, beta {1.0, 1.5} only — tighten the one soft spot
# from the N=200 sweep (beta=1.5 was +0.003 over beta=1.0, ~1 SE = noise).
# At N=1000 the standard error drops to ~0.0013, resolving whether energy-down
# gives any real gain. Isolated dirs (probe_n1000) — nothing existing touched.
#
#   beta=1.0 -> control (must match C3)
#   beta=1.5 -> energy slightly DOWN (residual amplified)

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
    tg_notify "❌ *intervene-n1000 FAILED* (job \`${SLURM_JOB_ID}\`)
Line: \`${line_no}\`  Exit: \`${exit_code}\`  Node: \`$(hostname)\`
Resubmit \`sbatch scripts/submit_intervene_n1000.sbatch\` to resume per-beta.
Log: \`outputs/logs/intervene_n1000_${SLURM_JOB_ID}.err\`"
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs \
    outputs/predictions/probe_n1000 outputs/metrics/probe_n1000 outputs/figures/probe_n1000

echo "[slurm] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
python -c "
import sys, torch
ok = torch.cuda.is_available() and torch.cuda.device_count() >= 1
print(f'[torch] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')
sys.exit(0 if ok else 1)
" || { echo "[slurm] CUDA init failed" >&2; exit 1; }

for f in \
    outputs/checkpoints/stage2_vlm_C3.pt \
    outputs/embeddings/projected_C3_stage2_text_pooled.pt \
    outputs/embeddings/projected_C3_stage2_image_pooled.pt; do
    if [[ ! -e "$f" ]]; then
        echo "[preflight] missing required input: $f" >&2
        tg_notify "❌ *intervene-n1000* — missing input \`$f\`."
        exit 1
    fi
done
echo "[preflight] inputs present"

tg_notify "🚀 *intervene-n1000* started (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
N=1000, β∈{1.0,1.5} — confirm whether energy-down gives a real gain (SE≈0.0013).
Isolated outputs under \`outputs/*/probe_n1000/\`."

python scripts/12_intervene_subspace.py \
    --subset-size 1000 \
    --betas 1.0 1.5 \
    --pred-dir outputs/predictions/probe_n1000 \
    --metrics-dir outputs/metrics/probe_n1000 \
    --fig-dir outputs/figures/probe_n1000

tg_notify "✅ *intervene-n1000 COMPLETE* (job \`${SLURM_JOB_ID}\`)
Results: \`outputs/metrics/probe_n1000/clipscore_vs_beta.json\`"
echo "[slurm] intervene-n1000 complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
