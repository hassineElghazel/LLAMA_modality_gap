#!/bin/bash
#SBATCH --job-name=mllm-pooled
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=47:00:00
#SBATCH --output=outputs/logs/pooled_%j.out
#SBATCH --error=outputs/logs/pooled_%j.err

# Pooled-257 control==measurement family. The geometry term (distance for C5p,
# InfoNCE for C4p) acts on the MEAN of all 257 projected tokens -- the same
# pooled vector G_mu / subspace_overlap are measured on and the decoder ingests
# -- instead of the CLS token. This closes the control != measurement gap.
#
# Begin with C5p (priority). C4p is staged below (commented) for the second run.

set -Eeuo pipefail

echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Environment
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
    tg_notify "❌ *POOLED FAILED* (job \`${SLURM_JOB_ID}\`)
Line: \`${line_no}\`  Exit: \`${exit_code}\`
Node: \`$(hostname)\`
Log: \`outputs/logs/pooled_${SLURM_JOB_ID}.err\`"
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/checkpoints outputs/predictions outputs/metrics outputs/embeddings

# --- Preflight: GPU ---
echo "[slurm] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
python -c "
import sys, torch
ok = torch.cuda.is_available() and torch.cuda.device_count() >= 1
print(f'[torch] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')
sys.exit(0 if ok else 1)
" || { echo "[slurm] CUDA init failed" >&2; exit 1; }

# --- Preflight: C2 connector (pooled runs init from it, like C3/C4/C5) ---
if [ ! -f outputs/checkpoints/stage1_connector_C2.pt ]; then
    echo "[pooled] FATAL: outputs/checkpoints/stage1_connector_C2.pt not found" >&2
    exit 1
fi

# --- Preflight: C3 diagnostic embeddings (frozen mu_y / trace_x sources) ---
MU_Y="outputs/embeddings/projected_C3_stage2_text_pooled.pt"
TRX="outputs/embeddings/projected_C3_stage2_image_pooled.pt"
if [ ! -f "$MU_Y" ]; then
    echo "[pooled] FATAL: $MU_Y not found (run 07_extract_projected.py --condition C3_stage2)" >&2
    exit 1
fi
test -f "$TRX" || echo "[pooled] WARN: $TRX missing; will fall back to config scalar trace_x"

# --- Preflight: COCO train2017 captions (replicates C4's exact image stream) ---
CAP_JSON="data/coco/annotations/captions_train2017.json"
if [ ! -f "$CAP_JSON" ]; then
    echo "[pooled] extracting captions_train2017.json from annotations zip"
    unzip -o -j data/coco/annotations_trainval2017.zip \
        annotations/captions_train2017.json -d data/coco/annotations/
fi
test -f "$CAP_JSON" || { echo "[pooled] FATAL: $CAP_JSON missing" >&2; exit 1; }

# --- CLIPScore setup (folded into run_one so it's never skipped) ---
CLIP_MODEL="$HOME/clip_b32"
[ -d "$CLIP_MODEL" ] || echo "[pooled] WARN: $CLIP_MODEL missing -> clipscore step will fail (non-fatal)"
grep -q use_safetensors scripts/15_clipscore.py || \
    sed -i 's/CLIPModel\.from_pretrained(args\.clip_model)/CLIPModel.from_pretrained(args.clip_model, use_safetensors=True)/' \
        scripts/15_clipscore.py

tg_notify "🚀 *POOLED started* (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
Geometry term on pooled-257 (control==measurement), init from C2, 450 steps.
Running: C5p_lam0p1 (distance lambda_d=0.1 on the mean of all 257 tokens)."

# train (450 steps) -> extract -> gap -> caption -> score -> clipscore.
# train_script ($3) differs per arm (C5p uses 17_..., C4p uses 14_...).
run_one() {
    local tag="$1"           # condition tag, e.g. C5p_lam0p1 (must be in CONDITIONS)
    local ckname="$2"        # checkpoint filename, e.g. stage2_vlm_C5p_lam0p1.pt
    local train_script="$3"  # 17_train_c5_distance.py | 14_train_c4_joint.py
    shift 3
    local train_flags=("$@") # e.g. --pool all257 --lambda-d 0.1
    local ckpt="outputs/checkpoints/${ckname}"

    tg_notify "POOLED ${tag}: training (${train_flags[*]})"
    python "scripts/${train_script}" "${train_flags[@]}" \
        --output-name "${ckname}" --max-steps 450

    tg_notify "POOLED ${tag}: extract + gap"
    python scripts/07_extract_projected.py --condition "${tag}" --connector-override "${ckpt}"
    python scripts/03_compute_gap.py --condition "${tag}"

    tg_notify "POOLED ${tag}: caption + score"
    python scripts/08_run_captioning.py --condition "${tag}" --vlm-checkpoint "${ckpt}"
    python scripts/09_score_captions.py --condition "${tag}"

    tg_notify "POOLED ${tag}: clipscore"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
        --condition "${tag}" --clip-model "$CLIP_MODEL" \
        || echo "[pooled] ${tag} clipscore FAILED (non-fatal; rerun submit_clipscore_c5.sbatch with CONDITIONS=${tag})"

    tg_notify "✅ POOLED ${tag} done"
}

# --- C5p (PRIORITY): location closed on the MEASURED pooled-257 vector. ---
# Matched to C5_lam0p1 (the C5-CLS peak); no scale pin needed (pooled batch-mean
# is a clean translation -> closes G_mu without compressing trace_image).
run_one C5p_lam0p1  stage2_vlm_C5p_lam0p1.pt  17_train_c5_distance.py  --pool all257 --lambda-d 0.1

# --- C4p (second run): orientation on the MEASURED pooled-257 vector. ---
# Uncomment to run after C5p (matched to C4_lam0p9, the orientation best point).
# run_one C4p_lam0p9  stage2_vlm_C4p_lam0p9.pt  14_train_c4_joint.py     --pool all257 --lambda 0.9

tg_notify "✅ *POOLED complete* (job \`${SLURM_JOB_ID}\`)
Checkpoint: \`stage2_vlm_C5p_lam0p1.pt\`
Gap: \`outputs/metrics/gap_C5p_lam0p1.json\`  Captions + CLIPScore done.
Verdict: C5p still ~0.58 => location weak even when closing the MEASURED G_mu
(the R@1<->captioning reconciliation is task-driven, not a CLS artifact);
clearly higher => closing the measured pooled location DOES help captioning."

echo "[slurm] POOLED complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
