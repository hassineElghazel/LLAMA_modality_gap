#!/bin/bash
#SBATCH --job-name=mllm-cloc-long
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=92:00:00
#SBATCH --output=outputs/logs/cloc_long_%j.out
#SBATCH --error=outputs/logs/cloc_long_%j.err

# ============================================================================
# THE LONG RUN: the method (Cloc) trained to a COMPLETED cosine schedule.
#
#   sbatch scripts/submit_cloc_long.sbatch
#
# WHY --subset-size AND NOT --max-steps.
#   17_train_c5_distance.py:224 computes total_steps from len(dataset), NOT from
#   --max-steps. Stopping early with --max-steps leaves the LR near peak and the
#   model unannealed (which is what every 450-step run in the thesis is).
#   --subset-size N makes the cosine complete over exactly N/32 steps.
#
#   SUBSET=72000  ->  2250 steps  ->  ~72 h at the measured 115 s/step
#   (measured from c5_85498, c5_85335, pooled_85537, pooled_85673: 113-117 s/step)
#
# CHECK AT 30 MINUTES: you should be near step 16. If you are at step 12 or
# fewer the node is slow -- cancel and resubmit with SUBSET=60000, because
# overshooting truncates the anneal and costs more than the missing steps.
#
# RERUN EVAL ONLY (checkpoint already exists):  SKIP_TRAIN=1 sbatch ...
# ============================================================================

set -Eeuo pipefail

SUBSET="${SUBSET:-72000}"
TAG="${TAG:-Cloc_long}"
CK="stage2_vlm_${TAG}.pt"
CKPT="outputs/checkpoints/${CK}"
STEPS=$(( SUBSET / 32 ))

echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[long]  subset=${SUBSET} -> ${STEPS} steps (annealed cosine), tag=${TAG}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.telegram_env 2>/dev/null || true

CLIP_MODEL="${CLIP_MODEL:-$HOME/clip_b32}"

tg_notify() {
    local msg="$1"
    [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]] && return 0
    curl -fsS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=${msg}" --data-urlencode "parse_mode=Markdown" >/dev/null || true
}
on_error() {
    local ec=$?
    tg_notify "❌ *CLOC-LONG FAILED* (job \`${SLURM_JOB_ID}\`)
Line: \`$1\`  Exit: \`${ec}\`  Node: \`$(hostname)\`
Checkpoint (if any) survives at \`${CKPT}\`; rerun eval with SKIP_TRAIN=1.
Log: \`outputs/logs/cloc_long_${SLURM_JOB_ID}.err\`"
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/checkpoints outputs/predictions outputs/metrics outputs/embeddings

nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
python -c "import torch; assert torch.cuda.is_available(); print('[preflight] gpu:', torch.cuda.get_device_name(0))"

# --- Train -----------------------------------------------------------------
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    if [[ -f "${CKPT}" ]]; then
        echo "[long] REFUSING to overwrite ${CKPT}. Use SKIP_TRAIN=1 to re-run eval, or pick another TAG." >&2
        exit 3
    fi
    tg_notify "🚀 *CLOC-LONG started* (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
${STEPS} steps, annealed cosine (subset=${SUBSET}), ~$(( STEPS * 115 / 3600 )) h.
This is the method: location drive, scale and rank pinned, pooled-257."

    python scripts/17_train_c5_distance.py \
        --pool all257 --lambda-d 0.1 --lambda-s 1.0 --lambda-rank 1.0 \
        --subset-size "${SUBSET}" \
        --output-name "${CK}"

    tg_notify "CLOC-LONG: training finished (${STEPS} steps). Starting evaluation."
else
    echo "[long] SKIP_TRAIN=1: evaluating the existing ${CKPT}"
    [[ -f "${CKPT}" ]] || { echo "[long] ${CKPT} not found" >&2; exit 4; }
fi

# --- Geometry + captioning + scoring ---------------------------------------
tg_notify "CLOC-LONG ${TAG}: extract + gap"
python scripts/07_extract_projected.py --condition "${TAG}" --connector-override "${CKPT}"
python scripts/03_compute_gap.py --condition "${TAG}"

tg_notify "CLOC-LONG ${TAG}: captioning (brief, n=5000) + score"
python scripts/08_run_captioning.py --condition "${TAG}" --vlm-checkpoint "${CKPT}"
python scripts/09_score_captions.py --condition "${TAG}"

tg_notify "CLOC-LONG ${TAG}: clipscore"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
    --condition "${TAG}" --clip-model "${CLIP_MODEL}" \
    || echo "[long] clipscore FAILED (non-fatal; rerun submit_clipscore_c5.sbatch with CONDITIONS=${TAG})"

# --- Report against the 450-step run and the reference ----------------------
python - <<PYEOF
import json, pathlib
def g(p,*k):
    f=pathlib.Path(p)
    if not f.exists(): return None
    d=json.loads(f.read_text())
    for x in k:
        d=d.get(x) if isinstance(d,dict) else None
        if d is None: return None
    return d
rows=[("Cloc (450 steps)","Cloc"),("${TAG} (${STEPS} steps)","${TAG}")]
print()
print(f"{'condition':28s} {'CLIP':>8s} {'SPICE':>8s} {'G_mu':>9s}")
for label,tag in rows:
    print(f"{label:28s} {str(g(f'outputs/metrics/clipscore_{tag}.json','CLIPScore'))[:8]:>8s} "
          f"{str(g(f'outputs/metrics/captioning_{tag}.json','scores','SPICE'))[:8]:>8s} "
          f"{str(g(f'outputs/metrics/gap_{tag}.json','spec_metrics','G_mu'))[:9]:>9s}")
print("\nreference ceiling (dd256): LLaVA-1.5-7B CLIPScore 0.7909")
PYEOF

tg_notify "✅ *CLOC-LONG done* \`${TAG}\` (job \`${SLURM_JOB_ID}\`)"
echo "[slurm] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
