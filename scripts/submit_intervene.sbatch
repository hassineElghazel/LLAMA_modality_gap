#!/bin/bash
#SBATCH --job-name=mllm-intervene
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/logs/intervene_%j.out
#SBATCH --error=outputs/logs/intervene_%j.err

# Causal intervention probe: manipulate subspace_energy on the trained C3 model
# (beta-transform on the connector output) and re-caption a subset, to test
# whether geometry is a causal lever for grounding.
#
# ISOLATED: reads only the C3 checkpoint + saved embeddings + COCO; writes ONLY
# under outputs/{predictions,metrics,figures}/probe/. Existing results untouched.
# Each beta's captioning has run_captioning resume support, so a killed job
# continues per-beta on resubmit.

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
    tg_notify "❌ *intervention probe FAILED* (job \`${SLURM_JOB_ID}\`)
Line: \`${line_no}\`  Exit: \`${exit_code}\`  Node: \`$(hostname)\`
Resubmit \`sbatch scripts/submit_intervene.sbatch\` to resume per-beta.
Log: \`outputs/logs/intervene_${SLURM_JOB_ID}.err\`"
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/predictions/probe outputs/metrics/probe outputs/figures/probe

# --- GPU sanity check ---
echo "[slurm] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
python -c "
import sys, torch
ok = torch.cuda.is_available() and torch.cuda.device_count() >= 1
print(f'[torch] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')
sys.exit(0 if ok else 1)
" || { echo "[slurm] CUDA init failed" >&2; exit 1; }

# --- preflight: C3 checkpoint + embeddings present (read-only inputs) ---
for f in \
    outputs/checkpoints/stage2_vlm_C3.pt \
    outputs/embeddings/projected_C3_stage2_text_pooled.pt \
    outputs/embeddings/projected_C3_stage2_image_pooled.pt; do
    if [[ ! -e "$f" ]]; then
        echo "[preflight] missing required input: $f" >&2
        tg_notify "❌ *intervention probe* — missing input \`$f\`."
        exit 1
    fi
done
echo "[preflight] inputs present"

tg_notify "🚀 *intervention probe* started (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
β-transform on C3 connector → re-caption N=1000 → CLIPScore vs β.
Isolated outputs under \`outputs/*/probe/\`."

python scripts/12_intervene_subspace.py --subset-size 1000

tg_notify "✅ *intervention probe COMPLETE* (job \`${SLURM_JOB_ID}\`)
Results: \`outputs/metrics/probe/clipscore_vs_beta.json\`
Plot: \`outputs/figures/probe/clipscore_vs_energy.png\`"
echo "[slurm] intervention probe complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
