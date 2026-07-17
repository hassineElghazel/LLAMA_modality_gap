#!/bin/bash
#SBATCH --job-name=mllm-cloc-clip-native
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=47:00:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/clocclipnative_%j.out
#SBATCH --error=outputs/logs/clocclipnative_%j.err

# ---------------------------------------------------------------------------
# Cloc_clip_NATIVE — location drive toward the CLIP text centroid at its NATURAL
# scale (~7.9, near origin), NOT the hand-imposed norm-178 target used in Cloc_clip.
# This is the faithful drop-in: use the CLIP text tower as-is, no scaling. Because
# the natural CLIP centroid sits near the origin (like LLaMA text), this should
# CLOSE the location gap and score ~ the original Cloc — confirming CLIP-location
# behaves like LLaMA-location and that location is about proximity-to-origin, not
# semantics. Resilient to baldur's intermittent GPU fault (retry + auto-resume).
#
# Prereq: the native anchor cloud
#   python scripts/21_extract_clip_text_anchor.py --device cpu \
#       --out outputs/embeddings_1300/clipanchor_text_1300_native.pt   # NO --match-norm
#
# Run from repo root: sbatch scripts/submit_cloc_clip_native.sbatch
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
    curl -fsS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=${msg}" --data-urlencode "parse_mode=Markdown" > /dev/null 2>&1 || true
}
on_error() { local ec=$?; tg_notify "❌ *CLOC-CLIP-NATIVE FAILED* (job \`${SLURM_JOB_ID}\`) line \`$1\` exit \`${ec}\` node \`$(hostname)\`"; }
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/checkpoints outputs/predictions outputs/metrics \
         outputs/embeddings_1300 outputs/metrics_1300

# --- Preflight: GPU responsive (fail fast on a dead baldur allocation) ---
python -c "
import sys, torch
if not torch.cuda.is_available(): print('[preflight] no CUDA'); sys.exit(1)
x = torch.randn(2048, 2048, device='cuda'); _ = (x @ x).sum().item(); torch.cuda.synchronize()
print('[preflight] GPU op OK on', torch.cuda.get_device_name(0))
" || { echo '[cloc-native] GPU preflight FAILED; exiting for resubmit' >&2; exit 1; }

# --- Preflight: artifacts ---
NATIVE_ANCHOR="outputs/embeddings_1300/clipanchor_text_1300_native.pt"
test -f outputs/checkpoints/stage1_connector_C2.pt || { echo '[cloc-native] FATAL: C2 connector missing' >&2; exit 1; }
test -f "$NATIVE_ANCHOR" || { echo "[cloc-native] FATAL: $NATIVE_ANCHOR missing -- run 21_extract_clip_text_anchor.py WITHOUT --match-norm" >&2; exit 1; }
test -f data/dd256/gap_manifest_1300.json || { echo '[cloc-native] FATAL: 1300 manifest missing' >&2; exit 1; }
CAP_JSON="data/coco/annotations/captions_train2017.json"
[ -f "$CAP_JSON" ] || unzip -o -j data/coco/annotations_trainval2017.zip annotations/captions_train2017.json -d data/coco/annotations/
test -f "$CAP_JSON" || { echo "[cloc-native] FATAL: $CAP_JSON missing" >&2; exit 1; }

DESC_CFG=configs/description_eval.yaml
CLIP_MODEL="${CLIP_MODEL:-$HOME/clip_b32}"
[ -d "$CLIP_MODEL" ] || echo "[cloc-native] WARN: $CLIP_MODEL missing -> clipscore will fail (non-fatal)"
grep -q use_safetensors scripts/15_clipscore.py || \
    sed -i 's/CLIPModel\.from_pretrained(args\.clip_model)/CLIPModel.from_pretrained(args.clip_model, use_safetensors=True)/' scripts/15_clipscore.py

CK="outputs/checkpoints/stage2_vlm_Cloc_clip_native.pt"
MAX_TRIES=6
tg_notify "🚀 *CLOC-CLIP-NATIVE* (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`) — location drive to natural CLIP centroid (~7.9). Retry+resume, up to ${MAX_TRIES} attempts."

# --- Retry loop: train 450 steps, resuming from the last 50-step checkpoint ---
attempt=1; success=0
while [ "$attempt" -le "$MAX_TRIES" ]; do
    RESUME_FLAG=()
    if [ -f "$CK" ]; then RESUME_FLAG=(--resume "$CK"); echo "[cloc-native] attempt ${attempt}: RESUME"; else echo "[cloc-native] attempt ${attempt}: fresh"; fi
    set +e; trap - ERR
    python scripts/17_train_c5_distance.py \
        --pool all257 --lambda-d 0.1 --lambda-s 1.0 --lambda-rank 1.0 \
        --mu-y-source "$NATIVE_ANCHOR" \
        --output-name stage2_vlm_Cloc_clip_native.pt --max-steps 450 "${RESUME_FLAG[@]}"
    rc=$?
    set -e; trap 'on_error ${LINENO}' ERR
    if [ "$rc" -eq 0 ]; then success=1; break; fi
    echo "[cloc-native] attempt ${attempt} failed (rc=${rc}); sleep 60s then retry/resume"
    tg_notify "CLOC-NATIVE attempt ${attempt} failed (rc=${rc}); retrying"
    sleep 60; attempt=$((attempt + 1))
done
[ "$success" -eq 1 ] || { echo "[cloc-native] all ${MAX_TRIES} attempts failed" >&2; tg_notify "❌ CLOC-CLIP-NATIVE: all attempts failed"; exit 1; }

tg_notify "✅ CLOC-CLIP-NATIVE trained — measuring gap(1300) + dd256 grounding"

# --- gap on 1300 (LLaMA anchor: should CLOSE, unlike the 178 version) ---
python scripts/07_extract_projected.py --condition Cloc_clip_native --connector-override "$CK" \
    --data-config configs/data_1300.yaml --out-dir outputs/embeddings_1300
python scripts/03_compute_gap.py --condition Cloc_clip_native \
    --embeddings-dir outputs/embeddings_1300 --out-dir outputs/metrics_1300
# real location gap = vs the CLIP flag it was trained toward (natural-scale anchor)
python scripts/03_compute_gap.py --condition Cloc_clip_native \
    --embeddings-dir outputs/embeddings_1300 --out-dir outputs/metrics_1300 \
    --text-embeddings "$NATIVE_ANCHOR" --out-suffix _vsCLIP

# --- dd256 grounding ---
python scripts/08_run_captioning.py --config "$DESC_CFG" --condition Cloc_clip_native_dd256 --vlm-checkpoint "$CK"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
    --config "$DESC_CFG" --condition Cloc_clip_native_dd256 --clip-model "$CLIP_MODEL" \
    || echo "[cloc-native] clipscore FAILED (non-fatal; rerun 15_clipscore.py --config $DESC_CFG --condition Cloc_clip_native_dd256)"

tg_notify "✅ *CLOC-CLIP-NATIVE complete* (job \`${SLURM_JOB_ID}\`)
gap: outputs/metrics_1300/gap_Cloc_clip_native.json
grounding: clipscore_Cloc_clip_native_dd256.json"
echo "[cloc-clip-native] done"
