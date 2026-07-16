#!/bin/bash
#SBATCH --job-name=mllm-clip-anchor
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=47:00:00
#SBATCH --output=outputs/logs/clipanchor_%j.out
#SBATCH --error=outputs/logs/clipanchor_%j.err

# ---------------------------------------------------------------------------
# CLIP-TEXT-ANCHOR arm. Retrain ONLY the two text-relative models toward a FROZEN
# CLIP text tower + frozen shared lift (outputs/anchors/clip_lift.pt), everything
# else identical to the originals:
#   Cloc_clip    : distance drive -> CLIP text centroid mu_y (--mu-y-source), scale
#                  + rank pinned. Anchor pre-scaled to image-cloud norm (~178) so
#                  the CLIP DIRECTION drives location (see scripts/21_extract...).
#   Corient_clip : InfoNCE positive from the CLIP tower (--clip-text-anchor); loc +
#                  scale + rank pinned. InfoNCE normalises -> aligns image DIRECTIONS
#                  to CLIP text directions (anchor scale irrelevant here).
#
# Eval is UNIFIED on n=1300 (the dd256 set): gap geometry (LLaMA anchor, via
# data_1300.yaml) + dd256 grounding (CLIPScore + CHAIR, reference-free). No brief
# captioning / CIDEr (train/eval mismatch). Outputs are non-destructive: _clip
# checkpoints, embeddings_1300 / metrics_1300 dirs, *_dd256 predictions.
#
# Run from repo root: sbatch scripts/submit_clip_anchor.sbatch
# ---------------------------------------------------------------------------

set -Eeuo pipefail

echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Environment
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
        --data-urlencode "text=${msg}" \
        --data-urlencode "parse_mode=Markdown" \
        > /dev/null || true
}

on_error() {
    local exit_code=$?
    local line_no=$1
    tg_notify "❌ *CLIP-ANCHOR FAILED* (job \`${SLURM_JOB_ID}\`)
Line: \`${line_no}\`  Exit: \`${exit_code}\`
Node: \`$(hostname)\`
Log: \`outputs/logs/clipanchor_${SLURM_JOB_ID}.err\`"
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/checkpoints outputs/predictions outputs/metrics \
         outputs/embeddings outputs/embeddings_1300 outputs/metrics_1300 outputs/anchors

# --- Preflight: GPU ---
echo "[slurm] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
python -c "
import sys, torch
ok = torch.cuda.is_available() and torch.cuda.device_count() >= 1
print(f'[torch] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')
sys.exit(0 if ok else 1)
" || { echo "[slurm] CUDA init failed" >&2; exit 1; }

# --- Preflight: C2 connector (both arms init from it, like C3/C4/C5) ---
test -f outputs/checkpoints/stage1_connector_C2.pt \
    || { echo "[clip] FATAL: stage1_connector_C2.pt not found" >&2; exit 1; }

# --- Preflight: CLIP anchor artifacts (built by scripts/21_extract_clip_text_anchor.py) ---
ANCHOR="outputs/embeddings_1300/clipanchor_text_1300.pt"   # Cloc_clip mu_y source (scaled ~178)
LIFT="outputs/anchors/clip_lift.pt"                        # shared frozen lift (Corient_clip)
test -f "$ANCHOR" || { echo "[clip] FATAL: $ANCHOR missing -- run scripts/21_extract_clip_text_anchor.py --match-norm 177.85" >&2; exit 1; }
test -f "$LIFT"   || { echo "[clip] FATAL: $LIFT missing -- run scripts/21_extract_clip_text_anchor.py first (materialises the shared lift)" >&2; exit 1; }

# --- Preflight: 1300 gap manifest (built by scripts/prep_dd256_gap_manifest.py) ---
test -f data/dd256/gap_manifest_1300.json \
    || { echo "[clip] FATAL: data/dd256/gap_manifest_1300.json missing -- run scripts/prep_dd256_gap_manifest.py" >&2; exit 1; }

# --- Preflight: trace_x source (frozen normaliser for the distance/pin terms) ---
test -f outputs/embeddings/projected_C3_stage2_image_pooled.pt \
    || echo "[clip] WARN: C3 image embeddings missing; trainers fall back to config scalar trace_x"

# --- Preflight: COCO train2017 captions (the contrastive/distance image stream) ---
CAP_JSON="data/coco/annotations/captions_train2017.json"
if [ ! -f "$CAP_JSON" ]; then
    echo "[clip] extracting captions_train2017.json from annotations zip"
    unzip -o -j data/coco/annotations_trainval2017.zip \
        annotations/captions_train2017.json -d data/coco/annotations/
fi
test -f "$CAP_JSON" || { echo "[clip] FATAL: $CAP_JSON missing" >&2; exit 1; }

# --- CLIPScore setup (grounding metric; folded into run_one so it's never skipped) ---
DESC_CFG=configs/description_eval.yaml
CLIP_MODEL="${CLIP_MODEL:-$HOME/clip_b32}"
[ -d "$CLIP_MODEL" ] || echo "[clip] WARN: $CLIP_MODEL missing -> clipscore step will fail (non-fatal)"
grep -q use_safetensors scripts/15_clipscore.py || \
    sed -i 's/CLIPModel\.from_pretrained(args\.clip_model)/CLIPModel.from_pretrained(args.clip_model, use_safetensors=True)/' \
        scripts/15_clipscore.py

tg_notify "🚀 *CLIP-ANCHOR started* (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
Retraining Cloc_clip + Corient_clip (450 steps) toward the frozen CLIP text anchor.
Eval unified on n=1300: gap (LLaMA anchor) + dd256 grounding (CLIPScore + CHAIR)."

# train (450 steps) -> gap on 1300 (LLaMA anchor) -> dd256 grounding (CLIPScore).
# CHAIR is run once at the end across both arms.
run_one() {
    local tag="$1"           # condition tag (must be in CONDITIONS): Cloc_clip | Corient_clip
    local ckname="$2"        # checkpoint filename, e.g. stage2_vlm_Cloc_clip.pt
    local train_script="$3"  # 17_train_c5_distance.py | 18_train_c6_orientation_pinned.py
    shift 3
    local train_flags=("$@")
    local ckpt="outputs/checkpoints/${ckname}"

    tg_notify "CLIP ${tag}: training (${train_flags[*]})"
    python "scripts/${train_script}" "${train_flags[@]}" \
        --output-name "${ckname}" --max-steps 450

    tg_notify "CLIP ${tag}: gap on 1300 (LLaMA anchor)"
    python scripts/07_extract_projected.py --condition "${tag}" --connector-override "${ckpt}" \
        --data-config configs/data_1300.yaml --out-dir outputs/embeddings_1300
    python scripts/03_compute_gap.py --condition "${tag}" \
        --embeddings-dir outputs/embeddings_1300 --out-dir outputs/metrics_1300

    tg_notify "CLIP ${tag}: dd256 grounding (detail prompt, 256 tok) + CLIPScore"
    python scripts/08_run_captioning.py --config "$DESC_CFG" \
        --condition "${tag}_dd256" --vlm-checkpoint "${ckpt}"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
        --config "$DESC_CFG" --condition "${tag}_dd256" --clip-model "$CLIP_MODEL" \
        || echo "[clip] ${tag}_dd256 clipscore FAILED (non-fatal; rerun 15_clipscore.py --config $DESC_CFG --condition ${tag}_dd256)"

    tg_notify "✅ CLIP ${tag} done (checkpoint + gap_1300 + dd256 CLIPScore)"
}

# Cloc_clip  (original Cloc: --pool all257 --lambda-d 0.1 --lambda-s 1.0 --lambda-rank 1.0)
run_one Cloc_clip    stage2_vlm_Cloc_clip.pt    17_train_c5_distance.py \
    --pool all257 --lambda-d 0.1 --lambda-s 1.0 --lambda-rank 1.0 \
    --mu-y-source outputs/embeddings_1300/clipanchor_text_1300.pt

# Corient_clip  (original Corient: --pool all257 --lambda 0.5 --lambda-p 10 --lambda-s 1.0 --lambda-rank 1.0)
run_one Corient_clip stage2_vlm_Corient_clip.pt 18_train_c6_orientation_pinned.py \
    --pool all257 --lambda 0.5 --lambda-p 10 --lambda-s 1.0 --lambda-rank 1.0 \
    --clip-text-anchor

# --- CHAIR across the two CLIP-anchor arms (reference-free hallucination) ---
# Baseline C3pinr_dd256 must already exist from the prior dd256 run; non-fatal otherwise.
tg_notify "CLIP: CHAIR (Cloc_clip / Corient_clip vs C3pinr_dd256)"
python scripts/19_chair.py --conditions Cloc_clip_dd256 Corient_clip_dd256 \
    --baseline C3pinr_dd256 --config "$DESC_CFG" \
    --out outputs/metrics/chair_summary_clip_anchor.json \
    || echo "[clip] CHAIR FAILED (non-fatal; ensure C3pinr_dd256 predictions exist, then rerun 19_chair.py)"

tg_notify "✅ *CLIP-ANCHOR complete* (job \`${SLURM_JOB_ID}\`)
Checkpoints: stage2_vlm_Cloc_clip.pt, stage2_vlm_Corient_clip.pt
Gap: outputs/metrics_1300/gap_{Cloc_clip,Corient_clip}.json
Grounding: outputs/metrics/clipscore_*_dd256.json + chair_summary_clip_anchor.json"

echo "[clip-anchor] done"
