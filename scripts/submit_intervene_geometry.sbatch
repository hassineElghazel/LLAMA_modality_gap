#!/bin/bash
#SBATCH --job-name=mllm-intervene-geom
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/logs/intervene_geom_%j.out
#SBATCH --error=outputs/logs/intervene_geom_%j.err

# Step 2 — causal geometry interventions on the step-1 BEST model.
# Dose-response, no-retrain interventions on the projected image tokens (alpha in
# [0,1], 0=identity .. 1=full), with TWO readouts per the Mind-the-Gap design:
#   identity     -> control (must match the base model's captions)
#   mean_shift   -> DISTANCE   (G_mu -> ~0)
#   rotate       -> ORIENTATION (image axes -> text axes)
#   recolor      -> SHAPE      (image spectrum -> text spectrum)
#   realign      -> FULL close (mean + global scale, ReAlign)
#   random_shift -> NULL control (||mu_y-mu_x||-matched random shift; OOD baseline)
# Readouts: (a) DECODER-FREE image->text retrieval on the pooled clouds (OOD-immune,
# fine dose grid); (b) CLIPScore on generated captions (decoder, coarse dose grid).
# Each map is verified on the pooled embeddings (intended component moved, others held).
#
# Set the target before submitting (the step-1 winner), e.g.:
#   CKPT=outputs/checkpoints/stage2_vlm_C4_lam0p1.pt EMB_COND=C4_lam0p1 \
#       sbatch scripts/submit_intervene_geometry.sbatch
#
# Optional env: SUBSET, SUBSPACE_Q, CLIP_MODEL (non-circular tower; default ~/clip_b32),
#   CAPTION_DOSES, GEOM_DOSES (space-separated alpha lists).
#
# Reads only: ${CKPT} + projected_${EMB_COND}_{image,text}_pooled.pt + COCO + CLIP tower.
# Writes only: outputs/{predictions,metrics,figures}/intervene/.

set -Eeuo pipefail

: "${CKPT:?set CKPT=outputs/checkpoints/<best>.pt (the step-1 winning model)}"
: "${EMB_COND:?set EMB_COND=<condition tag> matching CKPT (e.g. C4_lam0p1)}"
SUBSET="${SUBSET:-300}"
SUBSPACE_Q="${SUBSPACE_Q:-64}"
CLIP_MODEL="${CLIP_MODEL:-$HOME/clip_b32}"
CAPTION_DOSES="${CAPTION_DOSES:-0.5 1.0}"
GEOM_DOSES="${GEOM_DOSES:-0.25 0.5 0.75 1.0 1.25 1.5}"

echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1   # live .out progress (no block-buffering under SLURM)

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

# Non-circular CLIPScore tower (falls back to the circular ViT-L/14 if absent).
CLIP_ARGS=()
if [[ -d "${CLIP_MODEL}" ]]; then
    CLIP_ARGS=(--clip-model "${CLIP_MODEL}")
    echo "[preflight] non-circular CLIPScore tower: ${CLIP_MODEL}"
else
    echo "[preflight] WARNING: ${CLIP_MODEL} not found; CLIPScore will be CIRCULAR (ViT-L/14)."
fi

tg_notify "🚀 *intervene-geom* started (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
Target: \`${EMB_COND}\`  N=${SUBSET}  q=${SUBSPACE_Q}
Interventions: identity / mean_shift / rotate / recolor / realign / random_shift.
Readouts: decoder-free retrieval + CLIPScore. Isolated outputs under \`outputs/*/intervene/\`."

python scripts/16_intervene_geometry.py \
    --vlm-checkpoint "${CKPT}" \
    --emb-condition "${EMB_COND}" \
    --subset-size "${SUBSET}" \
    --subspace-q "${SUBSPACE_Q}" \
    --caption-doses ${CAPTION_DOSES} \
    --geom-doses ${GEOM_DOSES} \
    "${CLIP_ARGS[@]}"

tg_notify "✅ *intervene-geom COMPLETE* (job \`${SLURM_JOB_ID}\`)
Results: \`outputs/metrics/intervene/intervene_${EMB_COND}.json\`
Plot: \`outputs/figures/intervene/intervene_${EMB_COND}.png\`"
echo "[slurm] intervene-geom complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
