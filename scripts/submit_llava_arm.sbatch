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
# -> caption+score on DOCCI-test. The FIRST arm to run computes the shared frozen
# anchors (mu_y + trace_x); the other arms reuse them. 3 jobs total:
#     sbatch --export=ALL,ARM=pinned   scripts/submit_llava_arm.sbatch
#     sbatch --export=ALL,ARM=location scripts/submit_llava_arm.sbatch
#     sbatch --export=ALL,ARM=vanilla  scripts/submit_llava_arm.sbatch
#
# Eval = DOCCI-test (n=5000): reference-free CLIPScore (15) + SigLIP (15b), and
# reference-based BLEU/CIDEr/METEOR (28, --no-spice). Arm0 (pretrained) is run
# separately via submit_llava_eval.sbatch (no training).
set -Eeuo pipefail
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
trap 'tg_notify "❌ *LLaVA arm ${ARM} FAILED* (job \`${SLURM_JOB_ID}\`, line \`${LINENO}\`)"' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/checkpoints outputs/anchors_docci outputs/predictions outputs/metrics

nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
test -f data/docci/train_manifest.json || { echo "[arm] FATAL: run submit_docci_prep.sbatch first" >&2; exit 1; }
test -f data/docci/test_manifest.json  || { echo "[arm] FATAL: run submit_docci_prep.sbatch first" >&2; exit 1; }

MODEL_ID="${MODEL_ID:-llava-hf/llava-1.5-7b-hf}"
DOCCI=configs/docci_eval.yaml
CLIP_MODEL="${CLIP_MODEL:-$HOME/clip_b32}"
SIGLIP_MODEL="${SIGLIP_MODEL:-google/siglip-base-patch16-224}"

# ---------------- 1. anchors (shared across arms; compute once) ----------------
if [ ! -f outputs/anchors_docci/mu_y.pt ]; then
    echo "[arm] anchors missing -> computing (this arm is first; others will reuse)"
    HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} \
    python scripts/25_precompute_llava_anchors.py \
        --manifest data/docci/train_manifest.json \
        --model-id "${MODEL_ID}" \
        --out-dir outputs/anchors_docci
    echo "[arm] anchors -> $(cat outputs/anchors_docci/anchors.json)"
else
    echo "[arm] anchors present -> reuse: $(cat outputs/anchors_docci/anchors.json)"
fi

tg_notify "🚀 *LLaVA arm ${ARM}* started (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)"

# ---------------- 2. train this arm ----------------
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} \
python scripts/26_finetune_llava_3arm.py \
    --config configs/training_llava_docci.yaml --arm "${ARM}"

CKPT="outputs/checkpoints/llava_docci_${ARM}"
test -f "${CKPT}/connector.pt" || { echo "[arm] FATAL: ${CKPT}/connector.pt not written by training" >&2; exit 1; }
tg_notify "✅ *LLaVA arm ${ARM} trained* -> \`${CKPT}\`
$(cat ${CKPT}/train_sidecar.json 2>/dev/null | head -c 400)"

# ---------------- 3. caption DOCCI-test with this arm ----------------
python scripts/27_caption_llava.py --eval-config "${DOCCI}" --tag "${ARM}_docci" \
    --adapter-dir "${CKPT}" --connector-path "${CKPT}/connector.pt"

# ---------------- 4. score (all non-fatal so one metric can't sink the row) ----------------
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
    --config "${DOCCI}" --condition "${ARM}_docci" --clip-model "${CLIP_MODEL}" \
    || echo "[arm] ${ARM}_docci CLIPScore FAILED (non-fatal)"
python scripts/15b_siglip_score.py --config "${DOCCI}" --condition "${ARM}_docci" \
    --siglip-model "${SIGLIP_MODEL}" || echo "[arm] ${ARM}_docci SigLIP FAILED (non-fatal)"
python scripts/28_score_docci_refs.py --config "${DOCCI}" --tag "${ARM}_docci" --no-spice \
    || echo "[arm] ${ARM}_docci DOCCI-ref scoring FAILED (non-fatal; needs Java)"

tg_notify "🏁 *LLaVA arm ${ARM} DONE* (train+eval, job \`${SLURM_JOB_ID}\`)"
echo "[slurm] llava-arm ${ARM} complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
