#!/bin/bash
#SBATCH --job-name=mllm-c4
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=47:00:00
#SBATCH --output=outputs/logs/c4_%j.out
#SBATCH --error=outputs/logs/c4_%j.err

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
    tg_notify "❌ *C4 FAILED* (job \`${SLURM_JOB_ID}\`)
Line: \`${line_no}\`  Exit: \`${exit_code}\`
Node: \`$(hostname)\`
Log: \`outputs/logs/c4_${SLURM_JOB_ID}.err\`"
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

tg_notify "🚀 *C4 started* (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
Joint AR+contrastive, init from C2 connector, 450 steps each.
Sweep: lambda 0.1, lambda 0.5, Kendall (learned). lambda=0 is C3 (reuse)."

# --- One config: train (450 steps) -> extract -> gap -> caption -> score ---
# To run a single config as its own job, comment out the other run_one lines.
run_one() {
    local tag="$1"        # condition tag, e.g. C4_lam0p1 (must be in CONDITIONS)
    local ckname="$2"     # checkpoint filename, e.g. stage2_vlm_C4_lam0p1.pt
    shift 2
    local train_flags=("$@")   # e.g. --lambda 0.1   or   --kendall
    local ckpt="outputs/checkpoints/${ckname}"

    tg_notify "C4 ${tag}: training (${train_flags[*]})"
    python scripts/14_train_c4_joint.py "${train_flags[@]}" \
        --output-name "${ckname}" --max-steps 450

    tg_notify "C4 ${tag}: extract + gap"
    python scripts/07_extract_projected.py --condition "${tag}" --connector-override "${ckpt}"
    python scripts/03_compute_gap.py --condition "${tag}"

    tg_notify "C4 ${tag}: caption + score"
    python scripts/08_run_captioning.py --condition "${tag}" --vlm-checkpoint "${ckpt}"
    python scripts/09_score_captions.py --condition "${tag}"

    tg_notify "✅ C4 ${tag} done"
}

run_one C4_lam0p1  stage2_vlm_C4_lam0p1.pt  --lambda 0.1
run_one C4_lam0p5  stage2_vlm_C4_lam0p5.pt  --lambda 0.5
run_one C4_kendall stage2_vlm_C4_kendall.pt --kendall

tg_notify "✅ *C4 complete* (job \`${SLURM_JOB_ID}\`)
Checkpoints: \`stage2_vlm_C4_{lam0p1,lam0p5,kendall}.pt\`
Gap: \`outputs/metrics/gap_C4_*.json\`  Captions scored.
lambda=0 anchor = existing C3. Build the dose-response plot from the gap+caption JSONs."

echo "[slurm] C4 complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
