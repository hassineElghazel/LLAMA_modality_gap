#!/bin/bash
#SBATCH --job-name=llava-3arm
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=47:00:00
#SBATCH --output=outputs/logs/llava_3arm_%j.out
#SBATCH --error=outputs/logs/llava_3arm_%j.err

# ONE arm per job (QOS=1 serial). Launch with:
#     sbatch --export=ALL,ARM=vanilla  scripts/submit_llava_3arm.sbatch
#     sbatch --export=ALL,ARM=pinned   scripts/submit_llava_3arm.sbatch
#     sbatch --export=ALL,ARM=location scripts/submit_llava_3arm.sbatch
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
trap 'tg_notify "❌ *LLaVA 3-arm ${ARM} FAILED* (job \`${SLURM_JOB_ID}\`, line \`${LINENO}\`)"' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/checkpoints

nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
test -f outputs/anchors_docci/mu_y.pt || { echo "[3arm] FATAL: run submit_llava_anchors.sbatch first" >&2; exit 1; }
test -f data/docci/train_manifest.json || { echo "[3arm] FATAL: run submit_docci_prep.sbatch first" >&2; exit 1; }

tg_notify "🚀 *LLaVA 3-arm ${ARM}* started (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)"

HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} \
python scripts/26_finetune_llava_3arm.py \
    --config configs/training_llava_docci.yaml --arm "${ARM}"

CKPT="outputs/checkpoints/llava_docci_${ARM}"
tg_notify "✅ *LLaVA 3-arm ${ARM} done* -> \`${CKPT}\`
$(cat ${CKPT}/train_sidecar.json 2>/dev/null | head -c 400)"
echo "[slurm] llava-3arm ${ARM} complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
