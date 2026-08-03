#!/bin/bash
#SBATCH --job-name=llava-arm
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=47:00:00
#SBATCH --output=outputs/logs/llava_arm_%j.out
#SBATCH --error=outputs/logs/llava_arm_%j.err

# ONE arm END-TO-END in a single job (QOS=1 serial): anchors-if-missing -> train
# -> gap -> caption+score on DOCCI-test. The FIRST arm to run computes the shared
# frozen anchors (mu_y + mu_x0 + trace_x + pin setpoints); the others reuse them.
#     sbatch --export=ALL,ARM=pinned   scripts/submit_llava_arm.sbatch
#     sbatch --export=ALL,ARM=location scripts/submit_llava_arm.sbatch
#     sbatch --export=ALL,ARM=vanilla  scripts/submit_llava_arm.sbatch
#
# Arms (scripts/26 ARM_LAMBDAS):
#   vanilla   l_d=0.0  l_p=0.0   l_s=0.0  l_r=0.0   geometry free           (C3 analog)
#   pinned    l_d=0.0  l_p=10.0  l_s=1.0  l_r=1.0   all four axes held      (C3pinr analog)
#   location  l_d=0.1  l_p=0.0   l_s=1.0  l_r=1.0   location driven to mu_y (Cloc analog)
# The causal contrast is location - pinned: identical recipes except the location term.
#
# Eval = DOCCI-test (n=5000): gap suite (23), reference-free CLIPScore (15) + SigLIP
# (15b), reference-based BLEU/CIDEr/METEOR (28, --no-spice). Arm0 (pretrained) runs
# separately via submit_llava_eval.sbatch.
set -Eeuo pipefail
START_TS=$(date +%s)
echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

: "${ARM:?set ARM=vanilla|pinned|location via --export}"

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

# Report WHICH stage and WHICH command died. A bare line number sent us chasing the
# wrong bug once (the failure was a metadata call after a successful train, but the
# alert pointed at the training line).
STAGE="startup"
on_err() {
    local rc=$1
    tg_notify "❌ *LLaVA arm ${ARM} FAILED*
stage: \`${STAGE}\`
cmd: \`${FAILED_CMD}\`
exit \`${rc}\` after $(elapsed)
job \`${SLURM_JOB_ID}\` on \`$(hostname)\`
log: \`outputs/logs/llava_arm_${SLURM_JOB_ID}.err\`"
}
trap 'rc=$?; FAILED_CMD="${BASH_COMMAND}"; on_err "${rc}"' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/checkpoints outputs/anchors_docci outputs/predictions outputs/metrics outputs/embeddings_docci

nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
test -f data/docci/train_manifest.json || { echo "[arm] FATAL: run submit_docci_prep.sbatch first" >&2; exit 1; }
test -f data/docci/test_manifest.json  || { echo "[arm] FATAL: run submit_docci_prep.sbatch first" >&2; exit 1; }

MODEL_ID="${MODEL_ID:-llava-hf/llava-1.5-7b-hf}"
DOCCI=configs/docci_eval.yaml
TAG="${ARM}_docci"
CKPT="outputs/checkpoints/llava_docci_${ARM}"
CLIP_MODEL="${CLIP_MODEL:-$HOME/clip_b32}"
SIGLIP_MODEL="${SIGLIP_MODEL:-google/siglip-base-patch16-224}"
# what this arm is contrasted against in the closing summary
CONTROL="${CONTROL:-pinned_docci}"
[ "${ARM}" = "pinned" ] && CONTROL="arm0_docci"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo "?")
tg_notify "🚀 *LLaVA arm ${ARM}* started
job \`${SLURM_JOB_ID}\` on \`$(hostname)\` (${GPU_NAME})
plan: anchors -> train -> gap -> caption -> score
eval: DOCCI-test n=5000, contrast vs \`${CONTROL}\`"

# ---------------- 1. anchors (shared across arms; compute once) ----------------
# mu_x0.pt is REQUIRED by the location pin, so anchor sets predating it are stale
# and must be rebuilt -- otherwise the pinned arm fails at startup.
STAGE="anchors"
if [ ! -f outputs/anchors_docci/mu_y.pt ] || [ ! -f outputs/anchors_docci/mu_x0.pt ]; then
    echo "[arm] anchors missing/stale (need mu_y.pt AND mu_x0.pt) -> computing"
    HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} \
    python scripts/25_precompute_llava_anchors.py \
        --manifest data/docci/train_manifest.json \
        --model-id "${MODEL_ID}" \
        --geo-batch-size 32 \
        --out-dir outputs/anchors_docci
    tg_notify "📐 *anchors computed* (frozen, shared by all arms)
\`\`\`
$(cat outputs/anchors_docci/anchors.json)
\`\`\`"
else
    echo "[arm] anchors present -> reuse: $(cat outputs/anchors_docci/anchors.json)"
fi

# ---------------- 2. train this arm ----------------
STAGE="train"
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} \
python scripts/26_finetune_llava_3arm.py \
    --config configs/training_llava_docci.yaml --arm "${ARM}"

test -f "${CKPT}/connector.pt" || { echo "[arm] FATAL: ${CKPT}/connector.pt not written" >&2; exit 1; }
tg_notify "✅ *${ARM} trained* in $(elapsed) -> \`${CKPT}\`
$(python scripts/30_run_summary.py --arm "${ARM}" --tag "${TAG}" 2>/dev/null || echo 'summary unavailable')"

# ---------------- 3. gap suite on DOCCI-test (the geometry readout) ----------------
# Depends only on connector.pt: the LoRA sits on the LLM's q/v/o_proj, which neither
# the image cloud (stops at the connector output) nor the text cloud (embed_tokens)
# traverses. So this measures the intervention itself, on the EVAL set rather than
# on a 32-image training batch.
STAGE="gap"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/23_extract_llava15_geometry.py \
    --manifest data/docci/test_manifest.json \
    --condition "${TAG}" \
    --connector-path "${CKPT}/connector.pt" \
    --embeddings-dir outputs/embeddings_docci \
    || echo "[arm] ${TAG} gap FAILED (non-fatal)"

# ---------------- 4. caption DOCCI-test with this arm ----------------
STAGE="caption"
python scripts/27_caption_llava.py --eval-config "${DOCCI}" --tag "${TAG}" \
    --batch-size 8 --adapter-dir "${CKPT}" --connector-path "${CKPT}/connector.pt"

# ---------------- 5. score (all non-fatal so one metric can't sink the row) ----------------
STAGE="score"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
    --config "${DOCCI}" --condition "${TAG}" --clip-model "${CLIP_MODEL}" \
    || echo "[arm] ${TAG} CLIPScore FAILED (non-fatal)"
python scripts/15b_siglip_score.py --config "${DOCCI}" --condition "${TAG}" \
    --siglip-model "${SIGLIP_MODEL}" || echo "[arm] ${TAG} SigLIP FAILED (non-fatal)"
python scripts/28_score_docci_refs.py --config "${DOCCI}" --tag "${TAG}" --no-spice \
    || echo "[arm] ${TAG} DOCCI-ref scoring FAILED (non-fatal; needs Java)"

STAGE="summary"
SUMMARY=$(python scripts/30_run_summary.py --arm "${ARM}" --tag "${TAG}" --compare "${CONTROL}" 2>/dev/null \
          || echo "summary unavailable")
echo "${SUMMARY}"
tg_notify "🏁 *LLaVA arm ${ARM} DONE* (train+gap+eval, total $(elapsed))
${SUMMARY}"
echo "[slurm] llava-arm ${ARM} complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
