#!/bin/bash
#SBATCH --job-name=mllm-score-dd256
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/logs/score_dd256_%j.out
#SBATCH --error=outputs/logs/score_dd256_%j.err

# ============================================================================
# n-gram scores (BLEU, METEOR, CIDEr, SPICE) on the dd256 predictions.
#
#   sbatch scripts/submit_score_dd256.sbatch
#   CONDITIONS="Cloc_dd256 Corient_dd256" sbatch scripts/submit_score_dd256.sbatch
#
# configs/description_eval.yaml deliberately omits these metrics, because the
# dd256 captions run ~101 words against COCO references of ~11 and precision
# based scores collapse under that mismatch. They are computed here anyway so
# the omission can be justified with numbers rather than asserted: "we ran them,
# here they are, here is the length ratio that makes them uninformative".
#
# READ THE OUTPUT AS A PAIRED COMPARISON, NOT A LEADERBOARD. Cloc_long and
# Cloc_80k produce 101.6 and 103.9 words, a 2% difference, so they carry almost
# the same length penalty and the gap between them is meaningful. Against
# C3pinr (94.9 words) it is less so, and against LLaVA-1.5 (82.4) it is not:
# that model will look good on BLEU largely for being shorter.
#
# No GPU work happens here; the allocation exists only because the partition
# schedules that way. SPICE is a Java process and is the bulk of the runtime:
# budget ~1.5 h PER CONDITION at n=1300, not minutes. At ~100 words a caption
# exceeds the length SPICE can use as a cache filename, so it logs 'Caption may
# be too long' / 'Could not cache item', silently disables its cache, and
# reparses every caption through the Stanford pipeline. Those lines are noise,
# but the lost caching is why job 91603 died on a 2 h wall after one condition.
# Each condition writes its own metrics file as it finishes, so a rerun only
# needs the conditions that are missing.
# ============================================================================

set -Eeuo pipefail

CFG="${CFG:-configs/description_eval.yaml}"
CONDITIONS="${CONDITIONS:-Cloc_long_dd256 Cloc_80k_dd256}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
source ~/.telegram_env 2>/dev/null || true
cd ~/LLAMA_modality_gap

tg() { [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]] && return 0
       curl -fsS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
         --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" --data-urlencode "text=$1" >/dev/null || true; }
trap 'tg "dd256 scoring FAILED (job ${SLURM_JOB_ID}) line $1"' ERR

echo "[score] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[score] config=${CFG}"

# Fail before the first SPICE run rather than after it.
for c in ${CONDITIONS}; do
  f="outputs/predictions/captions_${c}.json"
  [[ -f "${f}" ]] || { echo "[score] missing ${f}" >&2; exit 3; }
  n=$(python -c "import json;print(len(json.load(open('${f}'))))")
  echo "[score] ${c}: ${n} captions"
done

tg "dd256 scoring started (job ${SLURM_JOB_ID}): ${CONDITIONS}"

for c in ${CONDITIONS}; do
  echo "==================== scoring ${c} ===================="
  python scripts/09_score_captions.py --config "${CFG}" --condition "${c}"
done

CONDITIONS="${CONDITIONS}" python - <<'PYEOF'
import json, os, pathlib
M = pathlib.Path("outputs/metrics")
P = pathlib.Path("outputs/predictions")
conds = os.environ["CONDITIONS"].split()
keys = ["BLEU-1", "BLEU-4", "METEOR", "CIDEr", "SPICE"]

def words(c):
    f = P / f"captions_{c}.json"
    if not f.exists(): return None
    rows = json.loads(f.read_text())
    return sum(len(r["caption"].split()) for r in rows) / max(1, len(rows))

print(f"\n{'condition':24}" + "".join(f"{k:>10}" for k in keys) + f"{'words':>9}")
print("-" * (24 + 10*len(keys) + 9))
for c in conds:
    f = M / f"captioning_{c}.json"
    s = json.loads(f.read_text()).get("scores", {}) if f.exists() else {}
    w = words(c)
    print(f"{c:24}" + "".join(f"{s[k]:>10.4f}" if isinstance(s.get(k), (int, float))
                              else f"{'-':>10}" for k in keys)
          + (f"{w:>9.1f}" if w else f"{'-':>9}"))
print("\nCOCO val2017 references average ~11 words. A caption of ~100 words cannot")
print("score well on precision-based metrics regardless of how good it is; compare")
print("the two runs to each other, where the length penalty is nearly equal.")
PYEOF

tg "dd256 scoring done (job ${SLURM_JOB_ID})"
echo "[score] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
