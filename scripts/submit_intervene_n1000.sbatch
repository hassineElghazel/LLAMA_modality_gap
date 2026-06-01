#!/bin/bash
#SBATCH --job-name=mllm-intervene-n1000
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/logs/intervene_n1000_%j.out
#SBATCH --error=outputs/logs/intervene_n1000_%j.err

# Confirmation run: N=1000, beta=1.5 ONLY — tighten the one soft spot from the
# N=200 sweep (beta=1.5 was +0.003 over beta=1.0, ~1 SE = noise).
#
# beta=1.0 is the identity transform, so it just reproduces the original C3
# captions (captions_C3_stage2.json), whose per-image CLIPScore is already on
# disk in outputs/metrics/linkage_C3_stage2.json. So we do NOT re-caption it —
# we only generate beta=1.5 here, then compare against the beta=1.0 baseline
# pulled from the linkage table on the SAME 1000 image ids (see the printed
# command at the end). At N=1000 the SE drops to ~0.0013.

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
    tg_notify "❌ *intervene-n1000 FAILED* (job \`${SLURM_JOB_ID}\`)
Line: \`${line_no}\`  Exit: \`${exit_code}\`  Node: \`$(hostname)\`
Resubmit \`sbatch scripts/submit_intervene_n1000.sbatch\` to resume per-beta.
Log: \`outputs/logs/intervene_n1000_${SLURM_JOB_ID}.err\`"
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs \
    outputs/predictions/probe_n1000 outputs/metrics/probe_n1000 outputs/figures/probe_n1000

echo "[slurm] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
python -c "
import sys, torch
ok = torch.cuda.is_available() and torch.cuda.device_count() >= 1
print(f'[torch] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')
sys.exit(0 if ok else 1)
" || { echo "[slurm] CUDA init failed" >&2; exit 1; }

for f in \
    outputs/checkpoints/stage2_vlm_C3.pt \
    outputs/embeddings/projected_C3_stage2_text_pooled.pt \
    outputs/embeddings/projected_C3_stage2_image_pooled.pt; do
    if [[ ! -e "$f" ]]; then
        echo "[preflight] missing required input: $f" >&2
        tg_notify "❌ *intervene-n1000* — missing input \`$f\`."
        exit 1
    fi
done
echo "[preflight] inputs present"

tg_notify "🚀 *intervene-n1000* started (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
N=1000, β=1.5 only (β=1.0 baseline reused from linkage table). SE≈0.0013.
Isolated outputs under \`outputs/*/probe_n1000/\`."

python scripts/12_intervene_subspace.py \
    --subset-size 1000 \
    --betas 1.5 \
    --pred-dir outputs/predictions/probe_n1000 \
    --metrics-dir outputs/metrics/probe_n1000 \
    --fig-dir outputs/figures/probe_n1000

# --- beta=1.0 baseline on the SAME 1000 ids, free, from the linkage table ---
echo "[probe-n1000] beta=1.5 vs beta=1.0 baseline (same 1000 images):"
python - <<'PYEOF'
import json
b15 = json.load(open("outputs/predictions/probe_n1000/captions_C3_beta1.50.json"))
ids = {int(r["image_id"]) for r in b15}
link = json.load(open("outputs/metrics/linkage_C3_stage2.json"))["rows"]
cs = [r["clipscore"] for r in link
      if int(r["image_id"]) in ids and r.get("clipscore") is not None]
b15_cs = json.load(open("outputs/metrics/probe_n1000/clipscore_vs_beta.json"))["results"]["beta1.50"]["clipscore_mean"]
base = sum(cs) / len(cs) if cs else float("nan")
print(f"  beta=1.0 (C3 identity, from linkage) CLIPScore = {base:.4f}  (n={len(cs)})")
print(f"  beta=1.5 (energy down)               CLIPScore = {b15_cs:.4f}")
print(f"  delta (1.5 - 1.0) = {b15_cs - base:+.4f}   (SE~0.0013; >~0.003 = real gain)")
PYEOF

tg_notify "✅ *intervene-n1000 COMPLETE* (job \`${SLURM_JOB_ID}\`)
β=1.5: \`outputs/metrics/probe_n1000/clipscore_vs_beta.json\`
β=1.0 baseline reused from linkage; delta printed in the log."
echo "[slurm] intervene-n1000 complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
