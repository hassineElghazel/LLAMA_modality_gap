#!/bin/bash
#SBATCH --job-name=mllm-cap
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/logs/cap_%j.out
#SBATCH --error=outputs/logs/cap_%j.err

# COCO val2017 captioning + pycocoevalcap scoring for ALL FOUR conditions
# (C0_random, C1_stage2, C2_stage1, C3_stage2) in one job.
#
# Idempotent: 08_run_captioning.py reloads any existing captions_<cond>.json and
# only generates the missing image_ids, so re-running this job is safe.
#
# --no-requeue: explicit guard against SLURM auto-restart that silently
# overwrote a finished C3 checkpoint earlier in this project.

set -Eeuo pipefail

echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

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
    tg_notify "❌ *captioning eval FAILED* (job \`${SLURM_JOB_ID}\`)
Line: \`${line_no}\`  Exit: \`${exit_code}\`
Last condition attempted: \`${CURRENT_COND:-unknown}\`
Log: \`outputs/logs/cap_${SLURM_JOB_ID}.err\`"
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/predictions outputs/metrics

# --- GPU sanity check ---
echo "[slurm] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
python -c "
import sys, torch
ok = torch.cuda.is_available() and torch.cuda.device_count() >= 1
print(f'[torch] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')
sys.exit(0 if ok else 1)
" || { echo "[slurm] CUDA init failed" >&2; exit 1; }

# --- Preflight: required checkpoints exist ---
for f in \
    outputs/checkpoints/stage2_vlm_C1.pt \
    outputs/checkpoints/stage1_connector_C2.pt \
    outputs/checkpoints/stage2_vlm_C3.pt; do
    if [[ ! -e "$f" ]]; then
        echo "[preflight] missing required checkpoint: $f" >&2
        exit 1
    fi
done
echo "[preflight] all required checkpoints present"

tg_notify "🚀 *captioning eval started* (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
Conditions: C0_random, C1_stage2, C2_stage1, C3_stage2
Eval set: COCO val2017 (5000 images)"

# --- Per-condition captioning + scoring ---
# Each call streams predictions to outputs/predictions/captions_<cond>.json with
# resume support; scoring writes outputs/metrics/captioning_<cond>.json.
# SPICE skipped (--no-spice) to avoid the SPICE Java jar; CIDEr / BLEU / METEOR
# already cover what the thesis needs.

run_cond() {
    local cond="$1"
    local ckpt="$2"
    export CURRENT_COND="$cond"

    echo "[eval] === ${cond}  ckpt=${ckpt} ==="
    tg_notify "▶️ captioning *${cond}* (job \`${SLURM_JOB_ID}\`)"

    python scripts/08_run_captioning.py \
        --condition "$cond" \
        --vlm-checkpoint "$ckpt"

    python scripts/09_score_captions.py --condition "$cond" --no-spice

    # Tiny progress ping with the headline scores from the json we just wrote.
    local score_json="outputs/metrics/captioning_${cond}.json"
    if [[ -f "$score_json" ]]; then
        local summary
        summary=$(python -c "
import json
with open('${score_json}') as f:
    d = json.load(f)
s = d.get('scores', {})
keys = ['CIDEr', 'BLEU-4', 'METEOR']
print(' '.join(f\"{k}={s[k]:.4f}\" for k in keys if k in s))
" 2>/dev/null || echo "scores parse failed")
        tg_notify "✅ *${cond}* done — ${summary}"
    fi
}

# Order: C0 (cheapest sanity) → C1 → C2 → C3. Same LLM is hot-loaded across all
# four (only the connector / LoRA changes); HF cache is reused.
run_cond C0_random   random
run_cond C1_stage2   outputs/checkpoints/stage2_vlm_C1.pt
run_cond C2_stage1   outputs/checkpoints/stage1_connector_C2.pt
run_cond C3_stage2   outputs/checkpoints/stage2_vlm_C3.pt

unset CURRENT_COND
tg_notify "🎉 *captioning eval complete* (job \`${SLURM_JOB_ID}\`)
All 4 conditions scored. Files: \`outputs/metrics/captioning_C{0,1,2,3}_*.json\`"

echo "[slurm] all four conditions complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
