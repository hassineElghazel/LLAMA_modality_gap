#!/bin/bash
#SBATCH --job-name=mllm-c5
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=47:00:00
#SBATCH --output=outputs/logs/c5_%j.out
#SBATCH --error=outputs/logs/c5_%j.err

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
    tg_notify "❌ *C5 FAILED* (job \`${SLURM_JOB_ID}\`)
Line: \`${line_no}\`  Exit: \`${exit_code}\`
Node: \`$(hostname)\`
Log: \`outputs/logs/c5_${SLURM_JOB_ID}.err\`"
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

# --- Preflight: C2 connector (C5 inits from it; no Stage-1 rerun) ---
if [ ! -f outputs/checkpoints/stage1_connector_C2.pt ]; then
    echo "[c5] FATAL: outputs/checkpoints/stage1_connector_C2.pt not found" >&2
    exit 1
fi

# --- Preflight: C3 diagnostic embeddings (frozen mu_y / trace_x sources) ---
MU_Y="outputs/embeddings/projected_C3_stage2_text_pooled.pt"
TRX="outputs/embeddings/projected_C3_stage2_image_pooled.pt"
if [ ! -f "$MU_Y" ]; then
    echo "[c5] FATAL: $MU_Y not found (run 07_extract_projected.py --condition C3_stage2)" >&2
    exit 1
fi
test -f "$TRX" || echo "[c5] WARN: $TRX missing; will fall back to config scalar trace_x"

# --- Preflight: COCO train2017 captions (replicates C4's exact image stream) ---
CAP_JSON="data/coco/annotations/captions_train2017.json"
if [ ! -f "$CAP_JSON" ]; then
    echo "[c5] extracting captions_train2017.json from annotations zip"
    unzip -o -j data/coco/annotations_trainval2017.zip \
        annotations/captions_train2017.json -d data/coco/annotations/
fi
test -f "$CAP_JSON" || { echo "[c5] FATAL: $CAP_JSON missing" >&2; exit 1; }

tg_notify "🚀 *C5 started* (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
Joint AR+distance (location), init from C2 connector, 450 steps.
Running: lambda_d=0.1 (low end of the sweep; 0.5 already done)."

# --- One config: train (450 steps) -> extract -> gap -> caption -> score ---
# To run another lambda_d as its own job, add/uncomment a run_one line.
run_one() {
    local tag="$1"        # condition tag, e.g. C5_lam0p5 (must be in CONDITIONS)
    local ckname="$2"     # checkpoint filename, e.g. stage2_vlm_C5_lam0p5.pt
    shift 2
    local train_flags=("$@")   # e.g. --lambda-d 0.5
    local ckpt="outputs/checkpoints/${ckname}"

    tg_notify "C5 ${tag}: training (${train_flags[*]})"
    python scripts/17_train_c5_distance.py "${train_flags[@]}" \
        --output-name "${ckname}" --max-steps 450

    tg_notify "C5 ${tag}: extract + gap"
    python scripts/07_extract_projected.py --condition "${tag}" --connector-override "${ckpt}"
    python scripts/03_compute_gap.py --condition "${tag}"

    tg_notify "C5 ${tag}: caption + score"
    python scripts/08_run_captioning.py --condition "${tag}" --vlm-checkpoint "${ckpt}"
    python scripts/09_score_captions.py --condition "${tag}"

    tg_notify "✅ C5 ${tag} done"
}

# run_one C5_lam0p5  stage2_vlm_C5_lam0p5.pt  --lambda-d 0.5   # done
run_one C5_lam0p1  stage2_vlm_C5_lam0p1.pt  --lambda-d 0.1
# run_one C5_lam0p3  stage2_vlm_C5_lam0p3.pt  --lambda-d 0.3
# run_one C5_lam0p7  stage2_vlm_C5_lam0p7.pt  --lambda-d 0.7
# run_one C5_lam0p9  stage2_vlm_C5_lam0p9.pt  --lambda-d 0.9

tg_notify "✅ *C5 complete* (job \`${SLURM_JOB_ID}\`)
Checkpoint: \`stage2_vlm_C5_lam0p1.pt\`
Gap: \`outputs/metrics/gap_C5_lam0p1.json\`  Captions scored.
Compare vs C5_lam0p5: less closing -> less compression? read trace_image + CLIPScore."

echo "[slurm] C5 complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
