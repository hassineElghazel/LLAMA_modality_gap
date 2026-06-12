#!/bin/bash
#SBATCH --job-name=mllm-c4
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/logs/c4_%x_%j.out
#SBATCH --error=outputs/logs/c4_%x_%j.err

set -Eeuo pipefail

# One C4 config per job (smaller, faster-scheduling, lower blast radius than the
# all-in-one submit_c4.sbatch). Pass the condition tag, the checkpoint filename,
# and the train flags as positional args; set a distinct --job-name so the logs
# and Telegram messages stay readable:
#
#   sbatch --job-name=c4-lam0p1  scripts/submit_c4_one.sbatch C4_lam0p1  stage2_vlm_C4_lam0p1.pt  --lambda 0.1
#   sbatch --job-name=c4-lam0p5  scripts/submit_c4_one.sbatch C4_lam0p5  stage2_vlm_C4_lam0p5.pt  --lambda 0.5
#   sbatch --job-name=c4-kendall scripts/submit_c4_one.sbatch C4_kendall stage2_vlm_C4_kendall.pt --kendall
#
# lambda=0 is C3 (reuse the existing checkpoint); there is no job for it.

if [ "$#" -lt 3 ]; then
    echo "usage: $0 <condition-tag> <checkpoint-filename> <train-flags...>" >&2
    echo "  e.g. $0 C4_lam0p1 stage2_vlm_C4_lam0p1.pt --lambda 0.1" >&2
    exit 2
fi
TAG="$1"
CKNAME="$2"
shift 2
TRAIN_FLAGS=("$@")
CKPT="outputs/checkpoints/${CKNAME}"

echo "[slurm] job=${SLURM_JOB_ID} name=${SLURM_JOB_NAME:-?} node=$(hostname) tag=${TAG} started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

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
    tg_notify "❌ *C4 ${TAG} FAILED* (job \`${SLURM_JOB_ID}\`)
Line: \`${line_no}\`  Exit: \`${exit_code}\`
Node: \`$(hostname)\`
Log: \`outputs/logs/c4_${SLURM_JOB_NAME:-job}_${SLURM_JOB_ID}.err\`"
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

# --- Preflight: C2 connector (C4 inits from it; no Stage-1 rerun) ---
if [ ! -f outputs/checkpoints/stage1_connector_C2.pt ]; then
    echo "[c4] FATAL: outputs/checkpoints/stage1_connector_C2.pt not found" >&2
    exit 1
fi

# --- Preflight: COCO train2017 captions (contrastive target) ---
CAP_JSON="data/coco/annotations/captions_train2017.json"
if [ ! -f "$CAP_JSON" ]; then
    echo "[c4] extracting captions_train2017.json from annotations zip"
    unzip -o -j data/coco/annotations_trainval2017.zip \
        annotations/captions_train2017.json -d data/coco/annotations/
fi
test -f "$CAP_JSON" || { echo "[c4] FATAL: $CAP_JSON missing" >&2; exit 1; }

tg_notify "🚀 *C4 ${TAG} started* (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
Joint AR+contrastive, init from C2 connector, 450 steps.
Flags: \`${TRAIN_FLAGS[*]}\`"

# --- train (450 steps) -> extract -> gap -> caption -> score ---
tg_notify "C4 ${TAG}: training (${TRAIN_FLAGS[*]})"
python scripts/14_train_c4_joint.py "${TRAIN_FLAGS[@]}" \
    --output-name "${CKNAME}" --max-steps 450

tg_notify "C4 ${TAG}: extract + gap"
python scripts/07_extract_projected.py --condition "${TAG}" --connector-override "${CKPT}"
python scripts/03_compute_gap.py --condition "${TAG}"

tg_notify "C4 ${TAG}: caption + score"
python scripts/08_run_captioning.py --condition "${TAG}" --vlm-checkpoint "${CKPT}"
python scripts/09_score_captions.py --condition "${TAG}"

tg_notify "✅ *C4 ${TAG} done* (job \`${SLURM_JOB_ID}\`)
Checkpoint: \`${CKNAME}\`
Gap: \`outputs/metrics/gap_${TAG}.json\`  Captions scored."

echo "[slurm] C4 ${TAG} complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
