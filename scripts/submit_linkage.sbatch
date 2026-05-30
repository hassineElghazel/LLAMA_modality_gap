#!/bin/bash
#SBATCH --job-name=mllm-linkage
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/logs/linkage_%j.out
#SBATCH --error=outputs/logs/linkage_%j.err

# Per-image gap<->caption linkage for the four caption-producing conditions
# (C0_random, C1_stage2, C2_stage1, C3_stage2). For each image it computes
# mean-centered pairwise alignment cos(z_img-mu, z_txt-mu) and correlates it
# against (a) CLIPScore grounding and (b) gzip/char-4gram degeneracy. CIDEr is
# included only for C1/C3 (it is floored noise for the C0/C2 gibberish, and
# left off there with --no-cider).
#
# GPU is needed for CLIPScore (loads the full CLIP ViT-L/14 and re-reads images).

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
    tg_notify "❌ *linkage FAILED* (job \`${SLURM_JOB_ID}\`)
Line: \`${line_no}\`  Exit: \`${exit_code}\`
Last condition: \`${CURRENT_COND:-unknown}\`
Log: \`outputs/logs/linkage_${SLURM_JOB_ID}.err\`"
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/metrics outputs/figures/linkage

echo "[slurm] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
python -c "
import sys, torch
ok = torch.cuda.is_available() and torch.cuda.device_count() >= 1
print(f'[torch] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')
sys.exit(0 if ok else 1)
" || { echo "[slurm] CUDA init failed" >&2; exit 1; }

# --- Preflight: embeddings + captions present for every condition ---
for cond in C0_random C1_stage2 C2_stage1 C3_stage2; do
    for f in \
        "outputs/embeddings/projected_${cond}_image_pooled.pt" \
        "outputs/embeddings/projected_${cond}_text_pooled.pt" \
        "outputs/predictions/captions_${cond}.json"; do
        if [[ ! -e "$f" ]]; then
            echo "[preflight] missing required file: $f" >&2
            exit 1
        fi
    done
done
echo "[preflight] all embeddings + caption files present"

tg_notify "🚀 *linkage* started (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
Conditions: C0, C1, C2, C3 — centered alignment vs CLIPScore + degeneracy."

# --- Per-condition linkage + Telegram summary ---
run_link() {
    local cond="$1"; shift
    local extra="$*"
    export CURRENT_COND="$cond"

    echo "[link] === ${cond}  (flags: --clipscore ${extra}) ==="
    tg_notify "▶️ linkage *${cond}* (job \`${SLURM_JOB_ID}\`)"

    python scripts/11_link_gap_to_captions.py --condition "$cond" --clipscore ${extra}

    local j="outputs/metrics/linkage_${cond}.json"
    local line
    line=$(python -c "
import json
d = json.load(open('${j}'))['summary']
uni = d.get('univariate', {})
mv = d.get('multivariate', {})
def us(feat, tgt):
    r = uni.get(feat, {}).get(tgt, {})
    return f\"{r.get('spearman', float('nan')):.3f}\"
primary = 'clipscore' if 'clipscore' in mv else 'comp_ratio'
parts = [f\"n={d['n_joined']}\"]
cs = d.get('clipscore_mean')
if cs is not None:
    parts.append(f\"clip_mean={cs:.3f}\")
if primary in mv:
    parts.append(f\"R2({primary})={mv[primary]['r2']:.3f}\")
parts.append(f\"S[align~{primary}]={us('align_centered', primary)}\")
parts.append(f\"S[subsp~{primary}]={us('subspace_energy', primary)}\")
print(' | '.join(parts))
" 2>/dev/null || echo "parse failed")
    tg_notify "✅ *${cond}* — ${line}"
    echo "[link] ${cond}: ${line}"
}

run_link C0_random  --no-cider
run_link C1_stage2
run_link C2_stage1  --no-cider
run_link C3_stage2

unset CURRENT_COND
tg_notify "🎉 *linkage complete* (job \`${SLURM_JOB_ID}\`)
Tables: \`outputs/metrics/linkage_C*.json\`  Figures: \`outputs/figures/linkage/\`"

echo "[slurm] linkage complete for all four conditions $(date -u +%Y-%m-%dT%H:%M:%SZ)"
