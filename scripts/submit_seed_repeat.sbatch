#!/bin/bash
#SBATCH --job-name=mllm-seed
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=30:00:00
#SBATCH --output=outputs/logs/seed_%j.out
#SBATCH --error=outputs/logs/seed_%j.err

# ============================================================================
# SEED REPEAT of the attribution pair.  ONE (arm, seed) per submission.
#
#   ARM=pinned SEED=1 sbatch scripts/submit_seed_repeat.sbatch
#   ARM=loc    SEED=1 sbatch scripts/submit_seed_repeat.sbatch
#   ARM=pinned SEED=2 sbatch scripts/submit_seed_repeat.sbatch
#   ARM=loc    SEED=2 sbatch scripts/submit_seed_repeat.sbatch
#
# Four submissions total. Seed 42 already exists as C3pinr / Cloc, so this
# gives n=3 per condition. Each job is one ~20 h run and fits the 30 h wall.
#
# WHAT THE SEED VARIES: LoRA init, data order, and all sampling
# (17_train_c5_distance.py:161 / 18_train_c6_orientation_pinned.py, then the
# loader at :211 and :228). It does NOT vary the connector's starting point:
# that is loaded from a FIXED Stage-1 checkpoint shared by every condition
# (17_train_c5_distance.py:189). The repeats are therefore PAIRED across the
# two arms, which is what the claim needs bounded: the variance of the Stage-2
# procedure, not of Stage 1.
#
# FLAGS ARE IDENTICAL to the original runs (submit_pooled.sbatch:191 and :202).
# Only --seed and the output name change. Do not "improve" them here.
# ============================================================================

set -Eeuo pipefail

ARM="${ARM:?set ARM=pinned|loc}"
SEED="${SEED:?set SEED=1|2 (42 is the run already in the thesis)}"

case "$ARM" in
  pinned)
    TAG="C3pinr_s${SEED}"
    CK="stage2_vlm_C3pinr_s${SEED}.pt"
    TRAIN_SCRIPT="18_train_c6_orientation_pinned.py"
    TRAIN_FLAGS=(--pool all257 --lambda 0.0 --lambda-p 10 --lambda-s 1.0 --lambda-rank 1.0)
    ;;
  loc)
    TAG="Cloc_s${SEED}"
    CK="stage2_vlm_Cloc_s${SEED}.pt"
    TRAIN_SCRIPT="17_train_c5_distance.py"
    TRAIN_FLAGS=(--pool all257 --lambda-d 0.1 --lambda-s 1.0 --lambda-rank 1.0)
    ;;
  *) echo "ARM must be 'pinned' or 'loc', got '${ARM}'" >&2; exit 2 ;;
esac

echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[seed]  arm=${ARM} seed=${SEED} tag=${TAG} ckpt=${CK}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.telegram_env 2>/dev/null || true

CLIP_MODEL="${CLIP_MODEL:-$HOME/clip_b32}"

tg_notify() {
    local msg="$1"
    [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]] && return 0
    curl -fsS -X POST \
        "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=${msg}" \
        --data-urlencode "parse_mode=Markdown" > /dev/null || true
}

on_error() {
    local exit_code=$?
    tg_notify "❌ *SEED REPEAT FAILED* (job \`${SLURM_JOB_ID}\`)
Arm: \`${ARM}\`  Seed: \`${SEED}\`  Tag: \`${TAG}\`
Line: \`$1\`  Exit: \`${exit_code}\`  Node: \`$(hostname)\`
Log: \`outputs/logs/seed_${SLURM_JOB_ID}.err\`"
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/checkpoints outputs/predictions outputs/metrics outputs/embeddings

# --- Preflight -------------------------------------------------------------
nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
python -c "
import sys, torch
assert torch.cuda.is_available(), 'no CUDA device'
print('[preflight] gpu:', torch.cuda.get_device_name(0))
"
if [[ -f "outputs/checkpoints/${CK}" ]]; then
    echo "[seed] REFUSING to overwrite existing outputs/checkpoints/${CK}" >&2
    exit 3
fi

tg_notify "🌱 *SEED REPEAT started* (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
Arm: \`${ARM}\`  Seed: \`${SEED}\`  Tag: \`${TAG}\`
Flags: \`${TRAIN_FLAGS[*]}\`
Same recipe as the thesis run; only the seed differs."

# --- Train -----------------------------------------------------------------
tg_notify "SEED ${TAG}: training (450 steps)"
python "scripts/${TRAIN_SCRIPT}" "${TRAIN_FLAGS[@]}" \
    --seed "${SEED}" --output-name "${CK}" --max-steps 450

CKPT="outputs/checkpoints/${CK}"

# --- Geometry --------------------------------------------------------------
tg_notify "SEED ${TAG}: extract + gap"
python scripts/07_extract_projected.py --condition "${TAG}" --connector-override "${CKPT}"
python scripts/03_compute_gap.py --condition "${TAG}"

# --- Captioning + scoring --------------------------------------------------
tg_notify "SEED ${TAG}: caption + score"
python scripts/08_run_captioning.py --condition "${TAG}" --vlm-checkpoint "${CKPT}"
python scripts/09_score_captions.py --condition "${TAG}"

tg_notify "SEED ${TAG}: clipscore"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
    --condition "${TAG}" --clip-model "${CLIP_MODEL}" \
    || echo "[seed] ${TAG} clipscore FAILED (non-fatal; rerun submit_clipscore_c5.sbatch with CONDITIONS=${TAG})"

# --- Report ----------------------------------------------------------------
python - <<PYEOF
import json, pathlib
tag = "${TAG}"
def read(p, *keys):
    f = pathlib.Path(p)
    if not f.exists(): return None
    d = json.loads(f.read_text())
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
        if d is None: return None
    return d
clip = read(f"outputs/metrics/clipscore_{tag}.json", "CLIPScore")
gm   = read(f"outputs/metrics/gap_{tag}.json", "spec_metrics", "G_mu")
tr   = read(f"outputs/metrics/gap_{tag}.json", "spec_metrics", "trace_image")
r    = read(f"outputs/metrics/gap_{tag}.json", "spec_metrics", "eff_rank_image")
sp   = read(f"outputs/metrics/captioning_{tag}.json", "scores", "SPICE")
fmt  = lambda v, p=4: f"{v:.{p}f}" if isinstance(v, float) else "n/a"
print(f"[seed] {tag}: CLIP={fmt(clip)} SPICE={fmt(sp)} "
      f"G_mu={fmt(gm,2)} trace={fmt(tr,1)} r={fmt(r,2)}")
pathlib.Path("outputs/metrics/seed_runs.jsonl").open("a").write(
    json.dumps({"tag": tag, "arm": "${ARM}", "seed": ${SEED},
                "CLIPScore": clip, "SPICE": sp,
                "G_mu": gm, "trace": tr, "eff_rank": r}) + "\n")
PYEOF

tg_notify "✅ *SEED REPEAT done* \`${TAG}\` (job \`${SLURM_JOB_ID}\`)
$(tail -1 outputs/metrics/seed_runs.jsonl)"

echo "[slurm] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
