#!/bin/bash
#SBATCH --job-name=mllm-finalize-step1
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/logs/finalize_step1_%j.out
#SBATCH --error=outputs/logs/finalize_step1_%j.err

# Finalize Step 1 of the C4 sweep, in ONE GPU job (run after lambda 0.3 training
# has finished and baldur is free):
#   1) lambda 0.9 geometry  : extract embeddings (07) + gap (03)   [only one missing]
#   2) ensure captions exist for C3 + every C4 lambda (08, resumable)
#   3) SPICE-inclusive scoring (09) for C3 + every C4 lambda
#
# SPICE fix: the crash was Java 17 module encapsulation, not a code bug. Install
# openjdk 8 in the env first (see runbook). This job ALSO auto-adds --add-opens
# if it detects a non-8 JVM, so it cannot crash at the SPICE step either way.
#
# Idempotent: skips embeddings/gap/captions that already exist; re-scoring simply
# overwrites the metrics JSON (adds the SPICE column to the --no-spice ones).

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
    tg_notify "❌ *finalize-step1 FAILED* (job \`${SLURM_JOB_ID}\`)
Line: \`${line_no}\`  Exit: \`${exit_code}\`  Cond: \`${CURRENT_COND:-?}\`
Log: \`outputs/logs/finalize_step1_${SLURM_JOB_ID}.err\`"
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/embeddings outputs/metrics outputs/predictions outputs/checkpoints

# --- GPU sanity check ---
echo "[slurm] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
python -c "
import sys, torch
ok = torch.cuda.is_available() and torch.cuda.device_count() >= 1
print(f'[torch] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')
sys.exit(0 if ok else 1)
" || { echo "[slurm] CUDA init failed" >&2; exit 1; }

# --- SPICE/Java guard: prefer Java 8; if 9+, open the modules SPICE reflects on ---
JV=$(java -version 2>&1 | head -1 || echo "no-java")
echo "[finalize] java: ${JV}"
if ! echo "${JV}" | grep -q '1\.8'; then
    export JAVA_TOOL_OPTIONS="--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED ${JAVA_TOOL_OPTIONS:-}"
    echo "[finalize] non-8 JVM -> JAVA_TOOL_OPTIONS=${JAVA_TOOL_OPTIONS}"
fi

CKDIR=outputs/checkpoints
EMB=outputs/embeddings
PRED=outputs/predictions

# --- Preflight: lambda 0.9 checkpoint is the core of this job and must exist.
#     Other conditions are scored only if their checkpoint is present, so this
#     job can run BEFORE lambda 0.3 has been (re)trained — it just skips it. ---
if [ ! -e ${CKDIR}/stage2_vlm_C4_lam0p9.pt ]; then
    echo "[preflight] missing required ${CKDIR}/stage2_vlm_C4_lam0p9.pt" >&2
    tg_notify "❌ *finalize-step1* missing λ0.9 checkpoint."
    exit 1
fi
echo "[preflight] lambda 0.9 checkpoint present; others scored only if their checkpoint exists"

tg_notify "🚀 *finalize-step1* started (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
λ0.9 geometry + captions, then SPICE scoring for C3 + all C4 λ."

# --- 1) lambda 0.9 geometry (the only condition still lacking embeddings/gap) ---
export CURRENT_COND="C4_lam0p9 (geometry)"
if [ ! -f "${EMB}/projected_C4_lam0p9_image_pooled.pt" ]; then
    tg_notify "C4_lam0p9: extract embeddings (07)"
    python scripts/07_extract_projected.py --condition C4_lam0p9 \
        --connector-override ${CKDIR}/stage2_vlm_C4_lam0p9.pt
fi
if [ ! -f outputs/metrics/gap_C4_lam0p9.json ]; then
    tg_notify "C4_lam0p9: gap (03)"
    python scripts/03_compute_gap.py --condition C4_lam0p9
fi

# --- 2)+3) ensure captions, then SPICE-inclusive scoring, per condition ---
score_one() {
    local cond="$1"; local ckpt="$2"
    if [ ! -e "$ckpt" ]; then
        echo "[finalize] SKIP ${cond}: checkpoint not present yet (${ckpt})"
        tg_notify "⏭️ skip *${cond}* — no checkpoint yet (train it separately)"
        return 0
    fi
    export CURRENT_COND="$cond"
    # 08 is resumable: it skips already-captioned image_ids and no-ops when a
    # condition is complete, so calling it unconditionally also finishes any
    # partial caption file (e.g. a lambda 0.3 run that timed out mid-caption).
    tg_notify "▶️ captioning *${cond}* (08, resumable)"
    python scripts/08_run_captioning.py --condition "$cond" --vlm-checkpoint "$ckpt"
    tg_notify "▶️ scoring (with SPICE) *${cond}* (09)"
    python scripts/09_score_captions.py --condition "$cond"
    local sj="outputs/metrics/captioning_${cond}.json"
    if [[ -f "$sj" ]]; then
        local summary
        summary=$(python -c "
import json
d=json.load(open('${sj}')); s=d.get('scores',{})
print(' '.join(f'{k}={s[k]:.4f}' for k in ['CIDEr','BLEU-4','METEOR','SPICE'] if k in s))
" 2>/dev/null || echo "parse failed")
        tg_notify "✅ *${cond}*: ${summary}"
    fi
}

score_one C3_stage2  ${CKDIR}/stage2_vlm_C3.pt
score_one C4_lam0p1  ${CKDIR}/stage2_vlm_C4_lam0p1.pt
score_one C4_lam0p3  ${CKDIR}/stage2_vlm_C4_lam0p3.pt
score_one C4_lam0p5  ${CKDIR}/stage2_vlm_C4_lam0p5.pt
score_one C4_lam0p7  ${CKDIR}/stage2_vlm_C4_lam0p7.pt
score_one C4_lam0p9  ${CKDIR}/stage2_vlm_C4_lam0p9.pt

unset CURRENT_COND
tg_notify "🎉 *finalize-step1 COMPLETE* (job \`${SLURM_JOB_ID}\`)
Gap: \`gap_C4_lam0p9.json\`. SPICE scores written for C3 + all C4 λ.
Pick the best λ (highest CLIPScore/CIDEr), then run intervene-geom for Step 2."
echo "[slurm] finalize-step1 complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
