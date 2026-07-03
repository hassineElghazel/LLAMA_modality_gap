#!/bin/bash
#SBATCH --job-name=mllm-pooled
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=47:00:00
#SBATCH --output=outputs/logs/pooled_%j.out
#SBATCH --error=outputs/logs/pooled_%j.err

# Pooled-257 control==measurement family. The geometry term (distance for C5p,
# InfoNCE for C4p) acts on the MEAN of all 257 projected tokens -- the same
# pooled vector G_mu / subspace_overlap are measured on and the decoder ingests
# -- instead of the CLS token. This closes the control != measurement gap.
#
# Begin with C5p (priority). C4p is staged below (commented) for the second run.

set -Eeuo pipefail

echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Environment
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
    tg_notify "❌ *POOLED FAILED* (job \`${SLURM_JOB_ID}\`)
Line: \`${line_no}\`  Exit: \`${exit_code}\`
Node: \`$(hostname)\`
Log: \`outputs/logs/pooled_${SLURM_JOB_ID}.err\`"
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/checkpoints outputs/predictions outputs/metrics outputs/embeddings

# --- Preflight: GPU ---
echo "[slurm] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[slurm] nvidia-smi failed"
python -c "
import sys, torch
ok = torch.cuda.is_available() and torch.cuda.device_count() >= 1
print(f'[torch] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')
sys.exit(0 if ok else 1)
" || { echo "[slurm] CUDA init failed" >&2; exit 1; }

# --- Preflight: C2 connector (pooled runs init from it, like C3/C4/C5) ---
if [ ! -f outputs/checkpoints/stage1_connector_C2.pt ]; then
    echo "[pooled] FATAL: outputs/checkpoints/stage1_connector_C2.pt not found" >&2
    exit 1
fi

# --- Preflight: C3 diagnostic embeddings (frozen mu_y / trace_x sources) ---
MU_Y="outputs/embeddings/projected_C3_stage2_text_pooled.pt"
TRX="outputs/embeddings/projected_C3_stage2_image_pooled.pt"
if [ ! -f "$MU_Y" ]; then
    echo "[pooled] FATAL: $MU_Y not found (run 07_extract_projected.py --condition C3_stage2)" >&2
    exit 1
fi
test -f "$TRX" || echo "[pooled] WARN: $TRX missing; will fall back to config scalar trace_x"

# --- Preflight: COCO train2017 captions (replicates C4's exact image stream) ---
CAP_JSON="data/coco/annotations/captions_train2017.json"
if [ ! -f "$CAP_JSON" ]; then
    echo "[pooled] extracting captions_train2017.json from annotations zip"
    unzip -o -j data/coco/annotations_trainval2017.zip \
        annotations/captions_train2017.json -d data/coco/annotations/
fi
test -f "$CAP_JSON" || { echo "[pooled] FATAL: $CAP_JSON missing" >&2; exit 1; }

# --- CLIPScore setup (folded into run_one so it's never skipped) ---
CLIP_MODEL="$HOME/clip_b32"
[ -d "$CLIP_MODEL" ] || echo "[pooled] WARN: $CLIP_MODEL missing -> clipscore step will fail (non-fatal)"
grep -q use_safetensors scripts/15_clipscore.py || \
    sed -i 's/CLIPModel\.from_pretrained(args\.clip_model)/CLIPModel.from_pretrained(args.clip_model, use_safetensors=True)/' \
        scripts/15_clipscore.py

tg_notify "🚀 *POOLED started* (job \`${SLURM_JOB_ID}\` on \`$(hostname)\`)
Geometry term on pooled-257 (control==measurement), init from C2, 450 steps.
Running: C5bp_lam0p3 (trace-hold THRESHOLD probe: distance lambda_d=0.3 + scale-pin lambda_s=1.0; location saturates ~G_mu 25-30, WATCH whether btrace holds ~4767 or compresses)."

# train (450 steps) -> extract -> gap -> caption -> score -> clipscore.
# train_script ($3) differs per arm (C5p uses 17_..., C4p uses 14_...).
run_one() {
    local tag="$1"           # condition tag, e.g. C5p_lam0p1 (must be in CONDITIONS)
    local ckname="$2"        # checkpoint filename, e.g. stage2_vlm_C5p_lam0p1.pt
    local train_script="$3"  # 17_train_c5_distance.py | 14_train_c4_joint.py
    shift 3
    local train_flags=("$@") # e.g. --pool all257 --lambda-d 0.1
    local ckpt="outputs/checkpoints/${ckname}"

    tg_notify "POOLED ${tag}: training (${train_flags[*]})"
    python "scripts/${train_script}" "${train_flags[@]}" \
        --output-name "${ckname}" --max-steps 450

    tg_notify "POOLED ${tag}: extract + gap"
    python scripts/07_extract_projected.py --condition "${tag}" --connector-override "${ckpt}"
    python scripts/03_compute_gap.py --condition "${tag}"

    tg_notify "POOLED ${tag}: caption + score"
    python scripts/08_run_captioning.py --condition "${tag}" --vlm-checkpoint "${ckpt}"
    python scripts/09_score_captions.py --condition "${tag}"

    tg_notify "POOLED ${tag}: clipscore"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
        --condition "${tag}" --clip-model "$CLIP_MODEL" \
        || echo "[pooled] ${tag} clipscore FAILED (non-fatal; rerun submit_clipscore_c5.sbatch with CONDITIONS=${tag})"

    tg_notify "✅ POOLED ${tag} done"
}

# --- C5p: DONE (CLIPScore 0.648, SPICE 0.128 -- best in study, but trace
# collapsed 62% so location vs compression is confounded). ---
# run_one C5p_lam0p1  stage2_vlm_C5p_lam0p1.pt  17_train_c5_distance.py  --pool all257 --lambda-d 0.1

# --- C5bp: DONE (CLIPScore 0.6337, SPICE 0.1217; trace HELD 4792, G_mu 29.6).
# Location-alone-with-scale-held = +0.090 over C3 (86% of C5p's gain) => location
# is the driver; the 62% compression in C5p added only +0.0144. ---
# run_one C5bp_lam0p1  stage2_vlm_C5bp_lam0p1.pt  17_train_c5_distance.py  --pool all257 --lambda-d 0.1 --lambda-s 1.0

# --- C4bp_lam0p1: ABANDONED (job 85983, killed step ~50). Two failures in the
# opening steps: (1) lambda_o=0.1 too weak for InfoNCE -- L_nce stuck ~3.85 (chance
# ln64=4.16), orientation barely moved; (2) lambda_p=0.5 too weak -- loc_drift
# climbed to ~70 (image centroid leaked ~28% of the 246 gap toward text), so it was
# NOT isolating orientation. Lesson: lambda=0.1 is a strong dose for the DIRECT
# distance loss (C5bp) but near-inert for INDIRECT InfoNCE -- matching the lambda
# number != matching the effect. Scale-pin held fine (btrace ~4767). ---
# run_one C4bp_lam0p1  stage2_vlm_C4bp_lam0p1.pt  18_train_c6_orientation_pinned.py  --pool all257 --lambda 0.1 --lambda-p 0.5 --lambda-s 1.0

# --- C4bp_lam0p9: DONE (job 86013). Clean pure-orientation run: location HELD at
# C2's 177 (drift ~12), scale held (trace 5031), orientation moved HUGE q16 0.051->0.168
# (> CLS C4's 0.137). CLIPScore 0.5886, SPICE 0.1053 -- 0.045 (~28 SE) BELOW C5bp's
# 0.634 (pure location). => LOCATION DOMINATES; orientation strongly manipulable but
# weakly effective. (+0.017 over the location-177 interpolation ~0.572 = possible small
# orientation effect OR curvature; C3pin below resolves it.) ---
# run_one C4bp_lam0p9  stage2_vlm_C4bp_lam0p9.pt  18_train_c6_orientation_pinned.py  --pool all257 --lambda 0.9 --lambda-p 10 --lambda-s 1.0

# --- C3pin (ACTIVE): pins-only control = C3 baseline (full AR, lambda_o=0) with the
# location-pin (lambda_p=10) + scale-pin (lambda_s=1.0) on pooled-257, NO InfoNCE.
# Holds location at C2's 177 and scale at baseline while captioning; orientation stays
# at baseline (nothing pushes it). PURPOSE: the true location-177 anchor.
#   * C3pin vs C3 (both AR weight 1.0) = CLEAN "hold location at 177" effect -> 3rd point
#     on the location dose-response (C3@246->0.543, C3pin@177->?, C5bp@30->0.634).
#   * C4bp vs C3pin = orientation effect (AR-weight caveat: C4bp AR=0.1 vs C3pin AR=1.0,
#     but the C4 sweep showed CLIPScore flat across AR 0.1-0.9, so negligible).
# Verdict on the +0.017 excess: C3pin ~0.572 => orientation IS a small real lever;
# C3pin ~0.589 (== C4bp) => orientation fully inert, the whole gain was held-location.
# WATCH: loc_drift ~0 (pin holds), btrace ~baseline, L_nce IGNORED (weighted 0).
# --- C3pin: DONE (the real captioning baseline: G_mu 177.8, CLIPScore 0.580,
# CHAIR_i 0.438, recall 0.391; verbose obj/cap 5.04). Controls AR-drift + verbosity. ---
# run_one C3pin  stage2_vlm_C3pin.pt  18_train_c6_orientation_pinned.py  --pool all257 --lambda 0.0 --lambda-p 10 --lambda-s 1.0

# --- C5bp LOCATION DOSE-RESPONSE SWEEP: TRACE HELD (ACTIVE) ---
# Goal: sample location (G_mu) at DIFFERENT points while trace stays at baseline (no
# compression). Sweep --lambda-d DOWNWARD, holding scale-pin --lambda-s 1.0.
# WHY downward: HIGH lambda_d (0.3-0.9) does NOT close location further -- it saturates
# at G_mu ~30 by lambda_d=0.1 -- it only COMPRESSES trace (the distance pull, realized
# via the shared connector toward the near-origin mu_y, shrinks output; the soft pin
# loses the tug-of-war). Verified on C5bp_lam0p9 (ABANDONED job 86227): G_mu->30 but
# btrace 4767->3006 (-37%). So trace-held closure lives at LOW lambda_d. At low lambda_d
# the pin wins and trace holds (lambda_d=0.1 held btrace 4792). Weaker lambda_d = less
# closure = higher G_mu with trace held. Run ONE per job; PLOT vs MEASURED G_mu.
# DONE: lambda_d=0.01 -> G_mu 81, btrace HELD 4788, CLIP 0.624 (trace-held intermediate point).
# run_one C5bp_lam0p01  stage2_vlm_C5bp_lam0p01.pt  17_train_c5_distance.py  --pool all257 --lambda-d 0.01 --lambda-s 1.0
# Now: lambda_d=0.3 -- maps the trace-hold THRESHOLD (0.1 held 4792, 0.9 compressed 3006 -37%).
# Location expected to SATURATE ~G_mu 25-30 (no new location point); WATCH btrace: held =>
# matched-location trace contrast vs C5bp_lam0p1; compressed => mid-compression point at fixed loc.
run_one C5bp_lam0p3   stage2_vlm_C5bp_lam0p3.pt   17_train_c5_distance.py  --pool all257 --lambda-d 0.3  --lambda-s 1.0
# staged trace-held fill (uncomment one per job): 0.02 (~G_mu 110), 0.05 (~G_mu 55):
# run_one C5bp_lam0p02  stage2_vlm_C5bp_lam0p02.pt  17_train_c5_distance.py  --pool all257 --lambda-d 0.02 --lambda-s 1.0
# run_one C5bp_lam0p05  stage2_vlm_C5bp_lam0p05.pt  17_train_c5_distance.py  --pool all257 --lambda-d 0.05 --lambda-s 1.0

tg_notify "✅ *POOLED complete* (job \`${SLURM_JOB_ID}\`)
Checkpoint: \`stage2_vlm_C5bp_lam0p3.pt\`
Gap: \`outputs/metrics/gap_C5bp_lam0p3.json\`  Captions + CLIPScore done.
CHECK: G_mu (expect saturated ~25-30) and btrace -- threshold probe: HELD ~4767 or COMPRESSED (<4000)?
Reference: lambda_d 0.1 held 4792, 0.9 compressed to 3006. 0.3 locates the trace-hold cliff."

echo "[slurm] POOLED complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
