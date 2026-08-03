#!/bin/bash
#SBATCH --job-name=llava-geo
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/llava_geo_%j.out
#SBATCH --error=outputs/logs/llava_geo_%j.err

# Measure the FULL gap suite for every available arm on DOCCI-test, in ONE job.
#     sbatch scripts/submit_llava_geometry.sbatch
#     sbatch --export=ALL,ARMS="arm0 location" scripts/submit_llava_geometry.sbatch
#
# Cheap: vision + connector forward only, no generation (~15 min/arm at n=5000),
# so the whole table fits in a single QOS slot. Arms without a checkpoint are
# skipped with a note rather than failing the job.
#
# Writes outputs/metrics/gap_<arm>_docci.json in the SAME schema as the diagnostic
# gap_C*.json -- G_mu, trace, eff_rank, subspace_overlap, anisotropy, kNN mixing,
# PMB/COB, JS-angular -- so the DOCCI arms drop straight into the existing table.
#
# Measures the CONNECTOR only, deliberately: the image cloud stops at the connector
# output and the text cloud is embed_tokens, while LoRA sits on the LLM's q/v/o_proj.
# Neither cloud traverses the adapter, so loading it would change no number, and the
# text cloud Y is therefore identical across arms.
#
# Two caveats worth carrying into the write-up:
#   * n=5000 here vs n=1300 for the diagnostic gap_C*.json rows.
#   * LLaVA's pooled cloud is eff_rank ~3.4 vs ~37 for the diagnostic models, so the
#     scale-normalised G_mu reads high by construction. Compare ARM-vs-ARM.
set -Eeuo pipefail
START_TS=$(date +%s)
echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

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
elapsed() { local s=$(( $(date +%s) - START_TS )); printf '%dh%02dm' $((s/3600)) $(((s%3600)/60)); }

STAGE="startup"
on_err() {
    local rc=$1
    tg_notify "❌ *LLaVA geometry FAILED*
stage: \`${STAGE}\`
cmd: \`${FAILED_CMD}\`
exit \`${rc}\` after $(elapsed)
job \`${SLURM_JOB_ID}\` on \`$(hostname)\`
log: \`outputs/logs/llava_geo_${SLURM_JOB_ID}.err\`"
}
trap 'rc=$?; FAILED_CMD="${BASH_COMMAND}"; on_err "${rc}"' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/metrics outputs/embeddings_docci

nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
test -f data/docci/test_manifest.json || { echo "[geo] FATAL: run submit_docci_prep.sbatch first" >&2; exit 1; }

ARMS="${ARMS:-arm0 vanilla pinned location}"
MANIFEST="${MANIFEST:-data/docci/test_manifest.json}"
MAX_IMAGES="${MAX_IMAGES:-}"          # set e.g. 200 for a smoke test

tg_notify "📐 *LLaVA geometry* started
job \`${SLURM_JOB_ID}\` on \`$(hostname)\`
arms: \`${ARMS}\`  set: \`${MANIFEST}\`"

DONE=""
for ARM in ${ARMS}; do
    STAGE="gap:${ARM}"
    TAG="${ARM}_docci"
    EXTRA=()
    [ -n "${MAX_IMAGES}" ] && EXTRA+=(--max-images "${MAX_IMAGES}")
    if [ "${ARM}" != "arm0" ]; then
        CKPT="outputs/checkpoints/llava_docci_${ARM}/connector.pt"
        if [ ! -f "${CKPT}" ]; then
            echo "[geo] SKIP ${ARM}: ${CKPT} not found (arm not trained yet)"
            continue
        fi
        EXTRA+=(--connector-path "${CKPT}")
    fi
    echo "[geo] ===== ${ARM} ====="
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/23_extract_llava15_geometry.py \
        --manifest "${MANIFEST}" \
        --condition "${TAG}" \
        --embeddings-dir outputs/embeddings_docci \
        "${EXTRA[@]}" \
        || { echo "[geo] ${ARM} FAILED (non-fatal, continuing)"; continue; }
    DONE="${DONE} ${ARM}"
done

# ---------------- one comparison table across whatever landed ----------------
STAGE="table"
TABLE=$(python - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob("outputs/metrics/gap_*_docci.json")):
    d = json.load(open(f))
    sm = d.get("spec_metrics", {})
    src = d.get("source", {})
    g, tr = sm.get("G_mu"), sm.get("trace_image")
    ng = src.get("G_mu_norm")
    if ng is None and g is not None and tr:
        ng = g / tr ** 0.5
    rows.append((os.path.basename(f)[4:-5].replace("_docci", ""),
                 g, ng, tr, sm.get("eff_rank_image"), src.get("n_pairs", d.get("n"))))
if rows:
    print(f"{'arm':10} {'G_mu':>8} {'normG':>7} {'trace':>9} {'effR':>7} {'n':>6}")
    for r in sorted(rows, key=lambda x: (x[2] is None, x[2])):
        f = lambda v, n=2: f"{v:.{n}f}" if isinstance(v, (int, float)) else "--"
        print(f"{r[0]:10} {f(r[1]):>8} {f(r[2],3):>7} {f(r[3],1):>9} {f(r[4]):>7} {str(r[5]):>6}")
else:
    print("no gap_*_docci.json found")
PY
)
echo "${TABLE}"
tg_notify "🏁 *LLaVA geometry DONE* ($(elapsed))
measured:\`${DONE:- none}\`
\`\`\`
${TABLE}
\`\`\`"
echo "[slurm] llava-geo complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
