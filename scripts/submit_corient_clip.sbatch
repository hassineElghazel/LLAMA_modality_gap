#!/bin/bash
#SBATCH --job-name=mllm-corient-clip
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=47:00:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/corientclip_%j.out
#SBATCH --error=outputs/logs/corientclip_%j.err

# ---------------------------------------------------------------------------
# Corient_clip RERUN — resilient to baldur's intermittent GPU fault.
#
# Cloc_clip already landed (job 87396). Corient_clip crashed on its first backward
# with "CUDA driver error: unknown error" (a transient Xid on baldur, NOT our code:
# Cloc did a full train + 2h20m caption on the same GPU first). Since baldur is the
# only GPU available, we don't exclude it — instead we survive the fault:
#   * RETRY loop: up to MAX_TRIES attempts at the 450-step training.
#   * AUTO-RESUME: the trainer saves every 50 steps; each retry resumes from the
#     last checkpoint (--resume), so progress accumulates across GPU hiccups.
#   * --requeue: if the whole node dies, SLURM requeues the job.
# The CLIP-anchor lift is loaded from the shared frozen artifact each time
# (deterministic W), so resumed/retried runs stay on one identical anchor.
#
# Run from repo root: sbatch scripts/submit_corient_clip.sbatch
# ---------------------------------------------------------------------------

set -Eeuo pipefail

echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || true)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/.telegram_env 2>/dev/null || true
tg_notify() {
    local msg="$1"
    [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]] && return 0
    curl -fsS -X POST \
        "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=${msg}" --data-urlencode "parse_mode=Markdown" \
        > /dev/null 2>&1 || true
}
on_error() {
    local exit_code=$?; local line_no=$1
    tg_notify "❌ *CORIENT-CLIP FAILED* (job \`${SLURM_JOB_ID}\`) line \`${line_no}\` exit \`${exit_code}\` node \`$(hostname)\`"
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/checkpoints outputs/predictions outputs/metrics \
         outputs/embeddings_1300 outputs/metrics_1300 outputs/anchors

# --- Preflight: GPU responsive (fail fast if baldur handed us a dead device) ---
python -c "
import sys, torch
if not torch.cuda.is_available(): print('[preflight] no CUDA'); sys.exit(1)
x = torch.randn(2048, 2048, device='cuda'); _ = (x @ x).sum().item()
torch.cuda.synchronize(); print('[preflight] GPU op OK on', torch.cuda.get_device_name(0))
" || { echo '[corient] GPU preflight FAILED (bad allocation); exiting for resubmit' >&2; exit 1; }

# --- Preflight: required artifacts ---
test -f outputs/checkpoints/stage1_connector_C2.pt || { echo '[corient] FATAL: C2 connector missing' >&2; exit 1; }
test -f outputs/anchors/clip_lift.pt              || { echo '[corient] FATAL: outputs/anchors/clip_lift.pt missing (run 21_extract_clip_text_anchor.py)' >&2; exit 1; }
CAP_JSON="data/coco/annotations/captions_train2017.json"
if [ ! -f "$CAP_JSON" ]; then
    unzip -o -j data/coco/annotations_trainval2017.zip annotations/captions_train2017.json -d data/coco/annotations/
fi
test -f "$CAP_JSON" || { echo "[corient] FATAL: $CAP_JSON missing" >&2; exit 1; }

DESC_CFG=configs/description_eval.yaml
CLIP_MODEL="${CLIP_MODEL:-$HOME/clip_b32}"
[ -d "$CLIP_MODEL" ] || echo "[corient] WARN: $CLIP_MODEL missing -> clipscore step will fail (non-fatal)"
grep -q use_safetensors scripts/15_clipscore.py || \
    sed -i 's/CLIPModel\.from_pretrained(args\.clip_model)/CLIPModel.from_pretrained(args.clip_model, use_safetensors=True)/' scripts/15_clipscore.py

CK="outputs/checkpoints/stage2_vlm_Corient_clip.pt"
MAX_TRIES=6

tg_notify "🚀 *CORIENT-CLIP rerun* (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
Resilient retry+resume on baldur (Cloc_clip already done). Up to ${MAX_TRIES} attempts."

# --- Retry loop: train 450 steps, resuming from the last 50-step checkpoint ---
attempt=1
success=0
while [ "$attempt" -le "$MAX_TRIES" ]; do
    RESUME_FLAG=()
    if [ -f "$CK" ]; then
        RESUME_FLAG=(--resume "$CK")
        echo "[corient] attempt ${attempt}/${MAX_TRIES}: RESUME from $CK"
        tg_notify "CORIENT attempt ${attempt}: resuming from checkpoint"
    else
        echo "[corient] attempt ${attempt}/${MAX_TRIES}: fresh start"
        tg_notify "CORIENT attempt ${attempt}: fresh start"
    fi

    set +e
    trap - ERR
    python scripts/18_train_c6_orientation_pinned.py \
        --pool all257 --lambda 0.5 --lambda-p 10 --lambda-s 1.0 --lambda-rank 1.0 \
        --clip-text-anchor \
        --output-name stage2_vlm_Corient_clip.pt --max-steps 450 "${RESUME_FLAG[@]}"
    rc=$?
    set -e
    trap 'on_error ${LINENO}' ERR

    if [ "$rc" -eq 0 ]; then success=1; break; fi
    echo "[corient] attempt ${attempt} failed (rc=${rc}); GPU hiccup — sleep 60s then retry/resume"
    tg_notify "CORIENT attempt ${attempt} failed (rc=${rc}); retrying with resume"
    sleep 60
    attempt=$((attempt + 1))
done

if [ "$success" -ne 1 ]; then
    echo "[corient] all ${MAX_TRIES} attempts failed" >&2
    tg_notify "❌ CORIENT-CLIP: all ${MAX_TRIES} attempts failed on baldur"
    exit 1
fi

tg_notify "✅ CORIENT-CLIP training done — measuring gap(1300) + dd256 grounding"

# --- gap on 1300 (LLaMA anchor) ---
python scripts/07_extract_projected.py --condition Corient_clip --connector-override "$CK" \
    --data-config configs/data_1300.yaml --out-dir outputs/embeddings_1300
python scripts/03_compute_gap.py --condition Corient_clip \
    --embeddings-dir outputs/embeddings_1300 --out-dir outputs/metrics_1300

# --- dd256 grounding (detail prompt, 256 tok) + CLIPScore ---
python scripts/08_run_captioning.py --config "$DESC_CFG" --condition Corient_clip_dd256 --vlm-checkpoint "$CK"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
    --config "$DESC_CFG" --condition Corient_clip_dd256 --clip-model "$CLIP_MODEL" \
    || echo "[corient] clipscore FAILED (non-fatal; rerun 15_clipscore.py --config $DESC_CFG --condition Corient_clip_dd256)"

# --- CHAIR across BOTH CLIP-anchor arms vs baseline (Cloc_clip already done) ---
python scripts/19_chair.py --conditions Cloc_clip_native_dd256 Corient_clip_dd256 \
    --baseline C3pinr_dd256 --config "$DESC_CFG" \
    --out outputs/metrics/chair_summary_clip_anchor.json \
    || echo "[corient] CHAIR FAILED (non-fatal; ensure C3pinr_dd256 + Cloc_clip_native_dd256 predictions exist)"

tg_notify "✅ *CORIENT-CLIP complete* (job \`${SLURM_JOB_ID}\`)
gap: outputs/metrics_1300/gap_Corient_clip.json
grounding: clipscore_Corient_clip_dd256.json + chair_summary_clip_anchor.json"

echo "[corient-clip] done"
