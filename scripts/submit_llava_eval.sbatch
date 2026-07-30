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

# Caption + score ONE arm on BOTH eval sets. Launch per arm (QOS=1 serial):
#     sbatch --export=ALL,ARM=arm0     scripts/submit_llava_eval.sbatch   # pretrained reference
#     sbatch --export=ALL,ARM=vanilla  scripts/submit_llava_eval.sbatch
#     sbatch --export=ALL,ARM=pinned   scripts/submit_llava_eval.sbatch
#     sbatch --export=ALL,ARM=location scripts/submit_llava_eval.sbatch
#
# Metric families per arm:
#   reference-free : CLIPScore (15) + SigLIP (15b)          on dd256 AND DOCCI-test
#   reference-based: COCO 5-ref CIDEr/METEOR/SPICE (09)     on dd256
#                    DOCCI single-gold SPICE/METEOR   (28)  on DOCCI-test
#   hallucination  : CHAIR_s/_i + recall (19)               on dd256, run ONCE across all
#                    four arms by whichever eval job finishes last.
set -Eeuo pipefail
echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

: "${ARM:?set ARM=arm0|vanilla|pinned|location via --export}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || true)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/predictions outputs/metrics outputs/reports

nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"

DESC=configs/description_eval.yaml     # dd256 (COCO val2017, n=1300)
DOCCI=configs/docci_eval.yaml          # DOCCI-test (n=5000)
CLIP_MODEL="${CLIP_MODEL:-$HOME/clip_b32}"      # local ViT-B/32 for offline CLIPScore
SIGLIP_MODEL="${SIGLIP_MODEL:-google/siglip-base-patch16-224}"

# --- adapter / connector flags: empty for arm0 (pretrained reference) ---
ADAPTER_FLAGS=()
if [ "${ARM}" != "arm0" ]; then
    CKPT="outputs/checkpoints/llava_docci_${ARM}"
    test -f "${CKPT}/connector.pt" || { echo "[eval] FATAL: ${CKPT}/connector.pt missing (train arm first)" >&2; exit 1; }
    ADAPTER_FLAGS=(--adapter-dir "${CKPT}" --connector-path "${CKPT}/connector.pt")
fi

# ---------------- caption both eval sets ----------------
python scripts/27_caption_llava.py --eval-config "${DESC}"  --tag "${ARM}_dd256" "${ADAPTER_FLAGS[@]}"
python scripts/27_caption_llava.py --eval-config "${DOCCI}" --tag "${ARM}_docci" "${ADAPTER_FLAGS[@]}"

# ---------------- reference-free (dd256) ----------------
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
    --config "${DESC}" --condition "${ARM}_dd256" --clip-model "${CLIP_MODEL}" \
    || echo "[eval] ${ARM}_dd256 CLIPScore FAILED (non-fatal)"
python scripts/15b_siglip_score.py --config "${DESC}" --condition "${ARM}_dd256" \
    --siglip-model "${SIGLIP_MODEL}" || echo "[eval] ${ARM}_dd256 SigLIP FAILED (non-fatal)"

# ---------------- reference-free (DOCCI) ----------------
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
    --config "${DOCCI}" --condition "${ARM}_docci" --clip-model "${CLIP_MODEL}" \
    || echo "[eval] ${ARM}_docci CLIPScore FAILED (non-fatal)"
python scripts/15b_siglip_score.py --config "${DOCCI}" --condition "${ARM}_docci" \
    --siglip-model "${SIGLIP_MODEL}" || echo "[eval] ${ARM}_docci SigLIP FAILED (non-fatal)"

# ---------------- reference-based ----------------
# dd256 vs COCO val2017 5-refs (real multi-ref). NOTE: COCO refs are SHORT, so BLEU/CIDEr
# penalise long detail captions -- read SPICE/METEOR; the ranking is fair across arms.
python scripts/09_score_captions.py --config "${DESC}" --condition "${ARM}_dd256" \
    || echo "[eval] ${ARM}_dd256 COCO-ref scoring FAILED (non-fatal; needs Java)"
# DOCCI-test vs single gold description (SPICE/METEOR).
python scripts/28_score_docci_refs.py --config "${DOCCI}" --tag "${ARM}_docci" \
    || echo "[eval] ${ARM}_docci DOCCI-ref scoring FAILED (non-fatal; needs Java)"

# ---------------- hallucination (CHAIR) — once all four dd256 caption files exist ----------------
DDG=outputs/predictions
if [ -f "${DDG}/captions_arm0_dd256.json" ] && [ -f "${DDG}/captions_vanilla_dd256.json" ] \
   && [ -f "${DDG}/captions_pinned_dd256.json" ] && [ -f "${DDG}/captions_location_dd256.json" ]; then
    echo "[eval] all four dd256 caption sets present -> running CHAIR across arms"
    python scripts/19_chair.py --config "${DESC}" \
        --conditions arm0_dd256 vanilla_dd256 pinned_dd256 location_dd256 \
        --baseline "${CHAIR_BASELINE:-pinned_dd256}" \
        --out outputs/metrics/chair_llava_3arm.json \
        || echo "[eval] CHAIR FAILED (non-fatal)"
else
    echo "[eval] not all four dd256 caption sets present yet -> CHAIR deferred to the last arm's eval job"
fi

echo "[slurm] llava-eval ${ARM} complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
