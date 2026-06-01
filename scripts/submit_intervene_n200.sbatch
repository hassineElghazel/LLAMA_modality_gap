#!/bin/bash
#SBATCH --job-name=mllm-intervene-n200
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/logs/intervene_n200_%j.out
#SBATCH --error=outputs/logs/intervene_n200_%j.err

# Fast focused intervention probe: N=200 images across beta {0,0.5,1,1.5,2},
# into ISOLATED dirs (probe_n200) so nothing existing is touched.
#
# Purpose:
#   beta=1.0  -> sanity gate: must match old C3 (wrapper is a true no-op)
#   beta<1    -> energy UP   -> expect collapse / low CLIPScore
#   beta>1    -> energy DOWN, residual amplified -> does CLIPScore rise?
#
# Reads only: stage2_vlm_C3.pt + projected_C3_stage2_{image,text}_pooled.pt + COCO.
# Writes only: outputs/{predictions,metrics,figures}/probe_n200/.

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
    tg_notify "❌ *intervene-n200 FAILED* (job \`${SLURM_JOB_ID}\`)
Line: \`${line_no}\`  Exit: \`${exit_code}\`  Node: \`$(hostname)\`
Resubmit \`sbatch scripts/submit_intervene_n200.sbatch\` to resume per-beta.
Log: \`outputs/logs/intervene_n200_${SLURM_JOB_ID}.err\`"
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs \
    outputs/predictions/probe_n200 outputs/metrics/probe_n200 outputs/figures/probe_n200

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
    outputs/checkpoints/stage2_vlm_C3.pt \
    outputs/embeddings/projected_C3_stage2_text_pooled.pt \
    outputs/embeddings/projected_C3_stage2_image_pooled.pt; do
    if [[ ! -e "$f" ]]; then
        echo "[preflight] missing required input: $f" >&2
        tg_notify "❌ *intervene-n200* — missing input \`$f\`."
        exit 1
    fi
done
echo "[preflight] inputs present"

tg_notify "🚀 *intervene-n200* started (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
N=200, β∈{0,0.5,1,1.5,2}. β=1 = sanity gate (must match C3); β>1 = energy down.
Isolated outputs under \`outputs/*/probe_n200/\`."

python scripts/12_intervene_subspace.py \
    --subset-size 200 \
    --betas 0.0 0.5 1.0 1.5 2.0 \
    --pred-dir outputs/predictions/probe_n200 \
    --metrics-dir outputs/metrics/probe_n200 \
    --fig-dir outputs/figures/probe_n200

tg_notify "✅ *intervene-n200 COMPLETE* (job \`${SLURM_JOB_ID}\`)
Results: \`outputs/metrics/probe_n200/clipscore_vs_beta.json\`
Plot: \`outputs/figures/probe_n200/clipscore_vs_energy.png\`"
echo "[slurm] intervene-n200 complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
