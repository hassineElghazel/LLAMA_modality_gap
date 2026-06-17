#!/bin/bash
#SBATCH --job-name=mllm-intervene-geom
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/logs/intervene_geom_%j.out
#SBATCH --error=outputs/logs/intervene_geom_%j.err

# Step 2 — causal geometry interventions on the step-1 BEST model.
# Four inference-time, no-retrain interventions on the projected image tokens:
#   identity   -> control (must match the base model's captions)
#   mean_shift -> DISTANCE   (G_mu -> ~0)
#   rotate     -> ORIENTATION (image axes -> text axes)
#   recolor    -> SHAPE      (image spectrum -> text spectrum)
# Verifies isolation on the pooled embeddings, then measures CLIPScore.
#
# Set the target before submitting (the step-1 winner), e.g.:
#   CKPT=outputs/checkpoints/stage2_vlm_C4_lam0p1.pt EMB_COND=C4_lam0p1 \
#       sbatch scripts/submit_intervene_geometry.sbatch
#
# Reads only: ${CKPT} + projected_${EMB_COND}_{image,text}_pooled.pt + COCO.
# Writes only: outputs/{predictions,metrics,figures}/intervene/.

set -Eeuo pipefail

: "${CKPT:?set CKPT=outputs/checkpoints/<best>.pt (the step-1 winning model)}"
: "${EMB_COND:?set EMB_COND=<condition tag> matching CKPT (e.g. C4_lam0p1)}"
SUBSET="${SUBSET:-300}"
SUBSPACE_Q="${SUBSPACE_Q:-64}"

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
    tg_notify "❌ *intervene-geom FAILED* (job \`${SLURM_JOB_ID}\`)
Line: \`${line_no}\`  Exit: \`${exit_code}\`  Node: \`$(hostname)\`
Log: \`outputs/logs/intervene_geom_${SLURM_JOB_ID}.err\`"
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs \
    outputs/predictions/intervene outputs/metrics/intervene outputs/figures/intervene

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
    "${CKPT}" \
    "outputs/embeddings/projected_${EMB_COND}_text_pooled.pt" \
    "outputs/embeddings/projected_${EMB_COND}_image_pooled.pt"; do
    if [[ ! -e "$f" ]]; then
        echo "[preflight] missing required input: $f" >&2
        tg_notify "❌ *intervene-geom* — missing input \`$f\`."
        exit 1
    fi
done
echo "[preflight] inputs present (CKPT=${CKPT} EMB_COND=${EMB_COND})"

tg_notify "🚀 *intervene-geom* started (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
Target: \`${EMB_COND}\`  N=${SUBSET}  q=${SUBSPACE_Q}
Interventions: identity / mean_shift / rotate / recolor.
Isolated outputs under \`outputs/*/intervene/\`."

python scripts/16_intervene_geometry.py \
    --vlm-checkpoint "${CKPT}" \
    --emb-condition "${EMB_COND}" \
    --subset-size "${SUBSET}" \
    --subspace-q "${SUBSPACE_Q}"

tg_notify "✅ *intervene-geom COMPLETE* (job \`${SLURM_JOB_ID}\`)
Results: \`outputs/metrics/intervene/intervene_${EMB_COND}.json\`
Plot: \`outputs/figures/intervene/intervene_${EMB_COND}.png\`"
echo "[slurm] intervene-geom complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
