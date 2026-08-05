#!/bin/bash
#SBATCH --job-name=matched
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --output=outputs/logs/matched_%j.out
#SBATCH --error=outputs/logs/matched_%j.err

# Re-score every arm on the MATCHED SUBSET: images where ALL arms terminated
# their caption instead of running past the 256-token cap.
#     sbatch scripts/submit_matched_rescore.sbatch
#     sbatch --export=ALL,TAGS="pinned_docci location_docci" scripts/submit_matched_rescore.sbatch
#
# WHY. The fine-tuned arms run past the cap at different rates (location 45.8%,
# vanilla 48.6%, pinned 57.2%) and those run-on captions are degenerate
# (distinct_4 ~0.34, 100% looping). A whole-set score therefore partly measures
# which arm degenerated more often. Conditional on terminating, the arms write
# equally well, so the matched subset compares like with like.
#
# READ IT AS: a rescue analysis on a subset SELECTED BY OUTCOME. Absolute values
# are optimistic (it is the easy quarter of the eval set) and arm0 remains
# non-comparable to the fine-tuned arms (62.8 words vs ~92, so CLIP's 77-token
# window sits differently for each). It supports "no C-sized location effect on
# comparable captions"; it cannot carry a headline grounding claim.
#
# No training, no captioning -- scores cached predictions only, so it is cheap
# (~20 min for 4 arms x 3 scorers) and safely re-runnable.
set -Eeuo pipefail
START_TS=$(date +%s)
echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || true)
source ~/.telegram_env 2>/dev/null || true

tg_notify() {
    [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]] && return 0
    curl -fsS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=$1" --data-urlencode "parse_mode=Markdown" >/dev/null || true
}
elapsed() { local s=$(( $(date +%s) - START_TS )); printf '%dh%02dm' $((s/3600)) $(((s%3600)/60)); }

STAGE="startup"
on_err() {
    local rc=$1
    tg_notify "❌ *matched re-score FAILED*
stage: \`${STAGE}\`
cmd: \`${FAILED_CMD}\`
exit \`${rc}\` after $(elapsed)
job \`${SLURM_JOB_ID}\` on \`$(hostname)\`
log: \`outputs/logs/matched_${SLURM_JOB_ID}.err\`"
}
trap 'rc=$?; FAILED_CMD="${BASH_COMMAND}"; on_err "${rc}"' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/metrics outputs/predictions

DOCCI=configs/docci_eval.yaml
TAGS="${TAGS:-arm0_docci vanilla_docci pinned_docci location_docci}"
CLIP_MODEL="${CLIP_MODEL:-$HOME/clip_b32}"
SIGLIP_MODEL="${SIGLIP_MODEL:-google/siglip-base-patch16-224}"

tg_notify "🔎 *matched re-score* started
job \`${SLURM_JOB_ID}\` on \`$(hostname)\`
tags: \`${TAGS}\`"

# ---------------- 1. build the matched subset (cheap, idempotent) ----------------
STAGE="subset"
python scripts/32_matched_subset.py --tags ${TAGS}
SUBSET_N=$(python -c "import json;print(json.load(open('outputs/metrics/matched_subset.json'))['n_matched'])")
echo "[matched] subset n=${SUBSET_N}"

# ---------------- 2. score each arm on the subset (all non-fatal) ----------------
for T in ${TAGS}; do
    STAGE="score:${T}"
    echo "[matched] ===== ${T}_matched ====="
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
        --config "${DOCCI}" --condition "${T}_matched" --clip-model "${CLIP_MODEL}" \
        || echo "[matched] ${T}_matched CLIPScore FAILED (non-fatal)"
    python scripts/15b_siglip_score.py --config "${DOCCI}" --condition "${T}_matched" \
        --siglip-model "${SIGLIP_MODEL}" || echo "[matched] ${T}_matched SigLIP FAILED (non-fatal)"
    python scripts/28_score_docci_refs.py --config "${DOCCI}" --tag "${T}_matched" --no-spice \
        || echo "[matched] ${T}_matched DOCCI-ref scoring FAILED (non-fatal)"
done

# ---------------- 3. one table + the causal contrast ----------------
STAGE="table"
TABLE=$(python - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob("outputs/metrics/clipscore_*_matched.json")):
    tag = os.path.basename(f)[len("clipscore_"):-len(".json")]
    d = json.load(open(f))
    sg = {}
    p = f"outputs/metrics/siglip_{tag}.json"
    if os.path.exists(p):
        sg = json.load(open(p))
    rows.append((tag.replace("_docci_matched", ""), d.get("CLIPScore"),
                 d.get("CLIPScore_std"), d.get("n_evaluated"), sg.get("SigLIPScore")))
if rows:
    print(f"{'arm':10} {'CLIPScore':>10} {'SE':>8} {'SigLIP':>8} {'n':>6}")
    for name, cs, sd, n, sg in sorted(rows, key=lambda r: -(r[1] or 0)):
        se = (sd / n ** 0.5) if (sd and n) else None
        f = lambda v, k=4: f"{v:.{k}f}" if isinstance(v, (int, float)) else "--"
        print(f"{name:10} {f(cs):>10} {f(se):>8} {f(sg):>8} {str(n):>6}")
    ref = {r[0]: r for r in rows}
    if "pinned" in ref and "location" in ref:
        a, b = ref["location"], ref["pinned"]
        se = ((a[2]**2)/a[3] + (b[2]**2)/b[3]) ** 0.5
        z = (a[1] - b[1]) / se
        print(f"\nlocation - pinned = {a[1]-b[1]:+.4f}  (diff-SE {se:.4f}, z={z:+.1f}) "
              f"-> {'SIGNIFICANT' if abs(z) >= 2 else 'null'}")
        print(f"2-sigma detection floor = {2*se:.4f}; the diagnostic-C location effect was +0.049")
PY
)
echo "${TABLE}"
tg_notify "🏁 *matched re-score DONE* ($(elapsed))
subset n=\`${SUBSET_N}\` (all arms terminated)
\`\`\`
${TABLE}
\`\`\`"
echo "[slurm] matched re-score complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
