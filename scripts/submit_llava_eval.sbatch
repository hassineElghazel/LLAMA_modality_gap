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

# Caption + score ONE arm on DOCCI-test ONLY. Launch per arm (QOS=1 serial):
#     sbatch --export=ALL,ARM=arm0     scripts/submit_llava_eval.sbatch   # pretrained reference
#     sbatch --export=ALL,ARM=vanilla  scripts/submit_llava_eval.sbatch
#     sbatch --export=ALL,ARM=pinned   scripts/submit_llava_eval.sbatch
#     sbatch --export=ALL,ARM=location scripts/submit_llava_eval.sbatch
#
# Metric families (DOCCI-test, n=5000):
#   reference-free : CLIPScore (15) + SigLIP (15b; needs sentencepiece+protobuf)
#   reference-based: single-gold BLEU/CIDEr/METEOR vs DOCCI description (28, --no-spice).
#   SPICE dropped on DOCCI: its Stanford parser OOMs per-sentence on DOCCI's long
#   captions (135-354 tok) and NPE-crashes; METEOR is the single-ref metric here.
# NOTE: COCO/dd256 eval is intentionally dropped (DOCCI-only study). That also
# drops hallucination (CHAIR): CHAIR needs COCO-80 instance annotations and
# cannot be computed on DOCCI. If hallucination is wanted later, run 19_chair.py
# on dd256 separately (no training dependency).
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

DOCCI=configs/docci_eval.yaml                   # DOCCI-test (n=5000)
CLIP_MODEL="${CLIP_MODEL:-$HOME/clip_b32}"      # local ViT-B/32 for offline CLIPScore
SIGLIP_MODEL="${SIGLIP_MODEL:-google/siglip-base-patch16-224}"

# --- adapter / connector flags: empty for arm0 (pretrained reference) ---
ADAPTER_FLAGS=()
if [ "${ARM}" != "arm0" ]; then
    CKPT="outputs/checkpoints/llava_docci_${ARM}"
    test -f "${CKPT}/connector.pt" || { echo "[eval] FATAL: ${CKPT}/connector.pt missing (train arm first)" >&2; exit 1; }
    ADAPTER_FLAGS=(--adapter-dir "${CKPT}" --connector-path "${CKPT}/connector.pt")
fi

# ---------------- caption DOCCI-test ----------------
python scripts/27_caption_llava.py --eval-config "${DOCCI}" --tag "${ARM}_docci" --batch-size 8 "${ADAPTER_FLAGS[@]}"

# ---------------- reference-free (DOCCI) ----------------
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
    --config "${DOCCI}" --condition "${ARM}_docci" --clip-model "${CLIP_MODEL}" \
    || echo "[eval] ${ARM}_docci CLIPScore FAILED (non-fatal)"
python scripts/15b_siglip_score.py --config "${DOCCI}" --condition "${ARM}_docci" \
    --siglip-model "${SIGLIP_MODEL}" || echo "[eval] ${ARM}_docci SigLIP FAILED (non-fatal)"

# ---------------- reference-based (DOCCI single gold: SPICE/METEOR) ----------------
# Circular vs Arm0 (arms 1-3 trained on DOCCI style) but FAIR between the geometry arms.
python scripts/28_score_docci_refs.py --config "${DOCCI}" --tag "${ARM}_docci" --no-spice \
    || echo "[eval] ${ARM}_docci DOCCI-ref scoring FAILED (non-fatal; needs Java)"

echo "[slurm] llava-eval ${ARM} complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
