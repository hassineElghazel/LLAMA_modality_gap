#!/bin/bash
#SBATCH --job-name=llava-short
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/llava_short_%j.out
#SBATCH --error=outputs/logs/llava_short_%j.err

# ONE arm with a configurable schedule, location dose and learning rate, evaluated
# on the 1000-image slice. Knobs: ARM, STEPS, TARGET_GAP, LR, TAG, EVAL_CFG.
#
#   # short-schedule dose study
#   sbatch --export=ALL,ARM=pinned,TAG=pinned_s150                   scripts/submit_llava_short.sbatch
#   sbatch --export=ALL,ARM=location,TAG=locfull_s150,TARGET_GAP=0.0  scripts/submit_llava_short.sbatch
#   sbatch --export=ALL,ARM=location,TAG=loc045_s150,TARGET_GAP=0.45  scripts/submit_llava_short.sbatch
#
#   # learning-rate probe for the termination collapse (full epoch, gentler lr)
#   sbatch --export=ALL,ARM=vanilla,TAG=vanilla_lr2e5,STEPS=603,LR=2e-5 scripts/submit_llava_short.sbatch
#
# THE TERMINATION COLLAPSE. Fine-tuning destroys the model's ability to stop:
# arm0 loops on 0.7% of captions, 150 steps on 84.1%, 603 steps on 76.3%. It
# breaks almost immediately and then relearns slowly, so it is an UNDER-training
# artifact rather than over-training damage -- but the recovery rate (4x the
# steps for -7.8pp) is far too slow for more epochs alone to fix it. lr=2e-4 came
# from the diagnostic recipe, which fine-tuned an UNDERTRAINED connector; on a
# fully-converged LLaVA it is the leading untested suspect. Ruled out already:
# EOS masking (pad 32001 is distinct from eos 2) and training-caption truncation
# (only 4.8% of DOCCI-train exceeds 256 tokens).
#
# WHY THIS EXPERIMENT. At a full epoch (603 steps) closing the location gap 72x
# (3.755 -> 0.052) changed grounding by z=-0.0 on quality-matched captions. Two
# explanations survive: (H1) the gap is a residue of co-adaptation, so a fully
# trained model has nothing to gain -- the lever should return as training is cut
# short; (H2) LLaVA's pooled cloud is eff_rank ~3.4, so the location axis carries
# too little structure to matter at any training length. 150 steps discriminates.
#
# WHY TWO DOSES. The diagnostic Cloc used lambda_d=0.1 -- the SAME value as the
# DOCCI location arm -- yet landed at normalised gap 0.447 while LLaVA ran to
# 0.052, nine times further and well past the knee of the C dose-response.
# Matching the lambda NUMBER is not matching the dose (cf. the C4bp lambda=0.1
# failure); matching the ENDPOINT is. TARGET_GAP=0.45 lands where Cloc landed;
# TARGET_GAP=0.0 keeps the aggressive dose at the same training length, so the
# schedule and the dose can be told apart.
#
# Eval: DOCCI-test first 1000 (configs/docci_eval_1000.yaml) -- ~45 min per arm
# instead of ~3.5h, with a 2-sigma floor of ~0.008 against a C effect of +0.049.
# Caption quality is checked automatically: at 150 steps the model sits closer to
# the pretrained checkpoint (0.7% looping), so the termination collapse seen at
# 603 steps may not appear at all.
set -Eeuo pipefail
START_TS=$(date +%s)
echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

: "${ARM:?set ARM=vanilla|pinned|location via --export}"
STEPS="${STEPS:-150}"
TARGET_GAP="${TARGET_GAP:-0.0}"
LR="${LR:-}"                       # empty -> use configs/training_llava_docci.yaml (2e-4)
TAG="${TAG:-${ARM}_s${STEPS}}"
LR_FLAGS=()
[ -n "${LR}" ] && LR_FLAGS=(--lr "${LR}")
# expand as ${LR_FLAGS[@]+"${LR_FLAGS[@]}"}: plain "${LR_FLAGS[@]}" on an empty
# array is an unbound-variable error under `set -u` in bash < 4.4.

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
    tg_notify "❌ *short-run ${TAG} FAILED*
stage: \`${STAGE}\`
cmd: \`${FAILED_CMD}\`
exit \`${rc}\` after $(elapsed)
job \`${SLURM_JOB_ID}\` on \`$(hostname)\`
log: \`outputs/logs/llava_short_${SLURM_JOB_ID}.err\`"
}
trap 'rc=$?; FAILED_CMD="${BASH_COMMAND}"; on_err "${rc}"' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/checkpoints outputs/predictions outputs/metrics outputs/embeddings_docci

nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
test -f outputs/anchors_docci/mu_x0.pt || { echo "[short] FATAL: anchors missing (run an arm job first)" >&2; exit 1; }

EVAL_CFG="${EVAL_CFG:-configs/docci_eval_1000.yaml}"
CKPT="outputs/checkpoints/llava_docci_${TAG}"
CLIP_MODEL="${CLIP_MODEL:-$HOME/clip_b32}"
SIGLIP_MODEL="${SIGLIP_MODEL:-google/siglip-base-patch16-224}"
CONTROL="${CONTROL:-pinned_s150_docci}"

tg_notify "⏱ *run ${TAG}* started
job \`${SLURM_JOB_ID}\` on \`$(hostname)\`
arm=\`${ARM}\` steps=\`${STEPS}\` target_gap=\`${TARGET_GAP}\` lr=\`${LR:-config 2e-4}\`
eval: \`${EVAL_CFG}\`, contrast vs \`${CONTROL}\`"

# ---------------- 1. train (short schedule; cosine rescales to STEPS) ----------------
STAGE="train"
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} \
python scripts/26_finetune_llava_3arm.py \
    --config configs/training_llava_docci.yaml \
    --arm "${ARM}" \
    --max-steps "${STEPS}" \
    --target-gap "${TARGET_GAP}" \
    ${LR_FLAGS[@]+"${LR_FLAGS[@]}"} \
    --out-dir "${CKPT}"

test -f "${CKPT}/connector.pt" || { echo "[short] FATAL: ${CKPT}/connector.pt not written" >&2; exit 1; }

# ---------------- 2. gap on the eval slice ----------------
STAGE="gap"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/23_extract_llava15_geometry.py \
    --manifest data/docci/test_manifest.json --max-images 1000 \
    --condition "${TAG}_docci" \
    --connector-path "${CKPT}/connector.pt" \
    --embeddings-dir outputs/embeddings_docci \
    || echo "[short] ${TAG} gap FAILED (non-fatal)"

# ---------------- 3. caption + score ----------------
STAGE="caption"
python scripts/27_caption_llava.py --eval-config "${EVAL_CFG}" --tag "${TAG}_docci" \
    --batch-size 8 --adapter-dir "${CKPT}" --connector-path "${CKPT}/connector.pt"

STAGE="score"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
    --config "${EVAL_CFG}" --condition "${TAG}_docci" --clip-model "${CLIP_MODEL}" \
    || echo "[short] ${TAG} CLIPScore FAILED (non-fatal)"
python scripts/15b_siglip_score.py --config "${EVAL_CFG}" --condition "${TAG}_docci" \
    --siglip-model "${SIGLIP_MODEL}" || echo "[short] ${TAG} SigLIP FAILED (non-fatal)"
python scripts/28_score_docci_refs.py --config "${EVAL_CFG}" --tag "${TAG}_docci" --no-spice \
    || echo "[short] ${TAG} DOCCI-ref scoring FAILED (non-fatal)"

# ---------------- 4. caption-quality gate ----------------
# The 603-step arms looped on ~75% of images and ran off the token cap on ~half,
# which silently invalidated the whole-set grounding numbers. Check it every time.
STAGE="quality"
QUALITY=$(python scripts/31_caption_quality.py --tags "${TAG}_docci" \
              --refs data/docci/test_manifest.json 2>/dev/null | head -20 || echo "quality check unavailable")
echo "${QUALITY}"

STAGE="summary"
SUMMARY=$(python scripts/30_run_summary.py --arm "${TAG}" --tag "${TAG}_docci" \
              --compare "${CONTROL}" 2>/dev/null || echo "summary unavailable")
echo "${SUMMARY}"
tg_notify "🏁 *short run ${TAG} DONE* ($(elapsed))
${SUMMARY}
\`\`\`
${QUALITY}
\`\`\`"
echo "[slurm] llava-short ${TAG} complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
