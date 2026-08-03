#!/bin/bash
#SBATCH --job-name=llava-eval
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/llava_eval_%j.out
#SBATCH --error=outputs/logs/llava_eval_%j.err

# Gap + caption + score ONE arm on DOCCI-test ONLY (no training). Launch per arm
# (QOS=1 serial):
#     sbatch --export=ALL,ARM=arm0     scripts/submit_llava_eval.sbatch   # pretrained reference
#     sbatch --export=ALL,ARM=vanilla  scripts/submit_llava_eval.sbatch
#     sbatch --export=ALL,ARM=pinned   scripts/submit_llava_eval.sbatch
#     sbatch --export=ALL,ARM=location scripts/submit_llava_eval.sbatch
#
# Use this when the checkpoint ALREADY EXISTS -- submit_llava_arm.sbatch has no
# skip-guard on training and would retrain from scratch.
#
# Metric families (DOCCI-test, n=5000):
#   geometry       : full gap suite at the connector output (23) -> gap_<tag>.json,
#                    same schema as the diagnostic gap_C*.json so the axes line up.
#   reference-free : CLIPScore (15) + SigLIP (15b; needs sentencepiece+protobuf)
#   reference-based: single-gold BLEU/CIDEr/METEOR vs the DOCCI description (28, --no-spice).
#   SPICE dropped on DOCCI: its Stanford parser OOMs per-sentence on DOCCI's long
#   captions (135-354 tok) and NPE-crashes; METEOR is the single-ref metric here.
# NOTE: COCO/dd256 eval is intentionally dropped (DOCCI-only study). That also drops
# hallucination (CHAIR): CHAIR needs COCO-80 instance annotations and cannot be
# computed on DOCCI. If hallucination is wanted, run 19_chair.py on dd256 separately
# (no training dependency) -- and dd256 is now genuinely OOD for DOCCI-trained arms.
set -Eeuo pipefail
START_TS=$(date +%s)
echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

: "${ARM:?set ARM=arm0|vanilla|pinned|location via --export}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || true)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.telegram_env 2>/dev/null || true

tg_notify() {
    [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]] && return 0
    curl -fsS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=$1" --data-urlencode "parse_mode=Markdown" >/dev/null || true
}
elapsed() { local s=$(( $(date +%s) - START_TS )); printf '%dh%02dm' $((s/3600)) $(((s%3600)/60)); }

STAGE="startup"
on_err() {
    local rc=$1
    tg_notify "❌ *LLaVA eval ${ARM} FAILED*
stage: \`${STAGE}\`
cmd: \`${FAILED_CMD}\`
exit \`${rc}\` after $(elapsed)
job \`${SLURM_JOB_ID}\` on \`$(hostname)\`
log: \`outputs/logs/llava_eval_${SLURM_JOB_ID}.err\`"
}
trap 'rc=$?; FAILED_CMD="${BASH_COMMAND}"; on_err "${rc}"' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/predictions outputs/metrics outputs/reports outputs/embeddings_docci

nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"

DOCCI=configs/docci_eval.yaml                   # DOCCI-test (n=5000)
TAG="${ARM}_docci"
CLIP_MODEL="${CLIP_MODEL:-$HOME/clip_b32}"      # local ViT-B/32 for offline CLIPScore
SIGLIP_MODEL="${SIGLIP_MODEL:-google/siglip-base-patch16-224}"
CONTROL="${CONTROL:-pinned_docci}"
[ "${ARM}" = "pinned" ] && CONTROL="arm0_docci"
[ "${ARM}" = "arm0" ]   && CONTROL=""

# --- adapter / connector flags: empty for arm0 (pretrained reference) ---
ADAPTER_FLAGS=()
GAP_FLAGS=()
if [ "${ARM}" != "arm0" ]; then
    CKPT="outputs/checkpoints/llava_docci_${ARM}"
    test -f "${CKPT}/connector.pt" || { echo "[eval] FATAL: ${CKPT}/connector.pt missing (train arm first)" >&2; exit 1; }
    ADAPTER_FLAGS=(--adapter-dir "${CKPT}" --connector-path "${CKPT}/connector.pt")
    GAP_FLAGS=(--connector-path "${CKPT}/connector.pt")
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo "?")
tg_notify "🚀 *LLaVA eval ${ARM}* started (no training)
job \`${SLURM_JOB_ID}\` on \`$(hostname)\` (${GPU_NAME})
plan: gap -> caption -> score, DOCCI-test n=5000"

# ---------------- geometry: full gap suite at the connector output ----------------
# Connector-only by design: LoRA lives on the LLM's q/v/o_proj, which neither the
# image cloud (stops at the connector output) nor the text cloud (embed_tokens)
# traverses, so the text side is identical across arms and every difference here is
# the intervention.
STAGE="gap"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/23_extract_llava15_geometry.py \
    --manifest data/docci/test_manifest.json \
    --condition "${TAG}" \
    --embeddings-dir outputs/embeddings_docci \
    "${GAP_FLAGS[@]}" \
    || echo "[eval] ${TAG} gap FAILED (non-fatal)"

# ---------------- caption DOCCI-test ----------------
STAGE="caption"
python scripts/27_caption_llava.py --eval-config "${DOCCI}" --tag "${TAG}" --batch-size 8 "${ADAPTER_FLAGS[@]}"

# ---------------- reference-free (DOCCI) ----------------
STAGE="score"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
    --config "${DOCCI}" --condition "${TAG}" --clip-model "${CLIP_MODEL}" \
    || echo "[eval] ${TAG} CLIPScore FAILED (non-fatal)"
python scripts/15b_siglip_score.py --config "${DOCCI}" --condition "${TAG}" \
    --siglip-model "${SIGLIP_MODEL}" || echo "[eval] ${TAG} SigLIP FAILED (non-fatal)"

# ---------------- reference-based (DOCCI single gold: BLEU/CIDEr/METEOR) ----------------
# Circular vs arm0 (arms 1-3 trained on DOCCI style) but FAIR between the geometry arms.
python scripts/28_score_docci_refs.py --config "${DOCCI}" --tag "${TAG}" --no-spice \
    || echo "[eval] ${TAG} DOCCI-ref scoring FAILED (non-fatal; needs Java)"

STAGE="summary"
if [ -n "${CONTROL}" ]; then
    SUMMARY=$(python scripts/30_run_summary.py --arm "${ARM}" --tag "${TAG}" --compare "${CONTROL}" 2>/dev/null || echo "summary unavailable")
else
    SUMMARY=$(python scripts/30_run_summary.py --arm "${ARM}" --tag "${TAG}" 2>/dev/null || echo "summary unavailable")
fi
echo "${SUMMARY}"
tg_notify "🏁 *LLaVA eval ${ARM} DONE* ($(elapsed))
${SUMMARY}"
echo "[slurm] llava-eval ${ARM} complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
