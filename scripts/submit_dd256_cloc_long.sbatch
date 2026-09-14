#!/bin/bash
#SBATCH --job-name=mllm-dd256-long
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --output=outputs/logs/dd256_long_%j.out
#SBATCH --error=outputs/logs/dd256_long_%j.err

# ============================================================================
# Task-matched (dd256) evaluation of the long Cloc run.
#
#   sbatch scripts/submit_dd256_cloc_long.sbatch
#
# Only Cloc_long is GENERATED. 08_run_captioning.py has no skip logic, so
# re-running the eight existing conditions would regenerate 1300 captions each
# for no reason; 19_chair.py reads predictions/captions_<tag>.json off disk, so
# the full table is rebuilt from what is already there plus the new row.
#
# No BLEU/CIDEr here, matching submit_description_eval.sbatch: the references
# are ~13 tokens and these captions are ~95, so n-gram metrics measure length.
# ============================================================================

set -Eeuo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
source ~/.telegram_env 2>/dev/null || true

CFG=configs/description_eval.yaml
TAG=Cloc_long_dd256
CKPT=outputs/checkpoints/stage2_vlm_Cloc_long.pt
CLIP_MODEL="${CLIP_MODEL:-$HOME/clip_b32}"
BASE=C3pinr_dd256
ALL="C3pinr_dd256 Cscale1500_dd256 Crank15_dd256 Corient_dd256 Cloc_dd256 Clocorient_dd256 C5bp_lam0p1_dd256 C4bp_lam0p9_dd256 ${TAG}"

tg() { [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]] && return 0
       curl -fsS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
         --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" --data-urlencode "text=$1" >/dev/null || true; }
trap 'tg "dd256 Cloc_long FAILED (job ${SLURM_JOB_ID}) line $1"' ERR

cd ~/LLAMA_modality_gap
[[ -f "${CKPT}" ]] || { echo "[dd256] ${CKPT} not found" >&2; exit 3; }
nvidia-smi --query-gpu=name --format=csv,noheader || true
tg "dd256 eval of Cloc_long started (job ${SLURM_JOB_ID})"

# keep the eight-condition table recoverable before 19_chair overwrites it
S=outputs/metrics/chair_summary_dd256.json
[[ -f "$S" ]] && cp -p "$S" "${S%.json}.pre_long.json"

echo "==================== DESCRIBE ${TAG} (1300 images, 256 tokens) ===================="
python scripts/08_run_captioning.py --config "${CFG}" --condition "${TAG}" --vlm-checkpoint "${CKPT}"

echo "==================== CLIPScore ${TAG} ===================="
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
    --config "${CFG}" --condition "${TAG}" --clip-model "${CLIP_MODEL}"

echo "==================== CHAIR, all conditions + ${TAG} ===================="
KEEP=""
for t in ${ALL}; do
  [[ -f "outputs/predictions/captions_${t}.json" ]] && KEEP="${KEEP} ${t}" || echo "[skip] ${t} (no predictions)"
done
python scripts/19_chair.py --conditions ${KEEP} --baseline "${BASE}" \
    --config "${CFG}" --out "${S}"

python - <<'PYEOF'
import json, pathlib
M = pathlib.Path("outputs/metrics")
ch = json.loads((M/"chair_summary_dd256.json").read_text()).get("per_condition", {})
def clip(t):
    p = M/f"clipscore_{t}.json"
    return json.loads(p.read_text())["CLIPScore"] if p.exists() else None
ROWS = [("all-pinned","C3pinr_dd256"), ("Cloc","Cloc_dd256"),
        ("Clocorient","Clocorient_dd256"), ("Cloc_long","Cloc_long_dd256")]
print(f"\n{'condition':14s} {'CLIP':>8s} {'CHAIR_s':>9s} {'CHAIR_i':>9s} {'recall':>8s} {'words':>7s}")
print("-"*60)
for lab, t in ROWS:
    c = clip(t); r = ch.get(t, {})
    f = lambda k: f"{r[k]:.4f}" if k in r else "n/a"
    print(f"{lab:14s} {(f'{c:.4f}' if c else 'n/a'):>8s} {f('chair_s'):>9s} {f('chair_i'):>9s} "
          f"{f('recall'):>8s} {f('avg_len'):>7s}")
print(f"{'LLaVA-1.5-7B':14s} {0.7910:8.4f} {0.5069:9.4f} {0.1518:9.4f} {0.7738:8.4f} {82.4:7.1f}   <- reference")
b, l = clip("C3pinr_dd256"), clip("Cloc_long_dd256")
if b and l:
    print(f"\nspan baseline -> reference covered on CLIPScore: {100*(l-b)/(0.7910-b):.0f}%")
PYEOF

tg "dd256 eval of Cloc_long done (job ${SLURM_JOB_ID})"
echo "[dd256] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
