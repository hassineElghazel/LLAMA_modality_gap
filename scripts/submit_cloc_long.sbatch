#!/bin/bash
#SBATCH --job-name=mllm-cloc-long
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=92:00:00
#SBATCH --output=outputs/logs/cloc_long_%j.out
#SBATCH --error=outputs/logs/cloc_long_%j.err

# ============================================================================
# THE LONG RUN, self-calibrating.
#
#   sbatch scripts/submit_cloc_long.sbatch
#
# WHY IT CALIBRATES.  17_train_c5_distance.py:224 fixes the cosine horizon from
# len(dataset) at launch, so the step count must be chosen BEFORE training and
# cannot be changed later. Too few steps wastes time; too many gets the job
# killed mid-schedule and leaves the model unannealed, which is worse.
# baldur is shared and its speed is not constant: the same recipe has run at
# 113 s/step (c5_85498, pooled_85673) and at 176 s/step (job 91318). So the job
# MEASURES itself first, then sizes the real run to the wall clock it has.
#
# PHASE A  ~10 steps on a throwaway checkpoint -> median s/step.
# PHASE B  steps = (WALL - EVAL - elapsed) * SAFETY / s_per_step, then train.
#
# Overrides: WALL_H, EVAL_H, SAFETY, CAL_STEPS, SPS (skip phase A), TAG.
#   SPS=113 sbatch ...        # trust a known rate, skip calibration
#   SKIP_TRAIN=1 sbatch ...   # re-run evaluation on an existing checkpoint
#   RESUME=1 SUBSET=48928 sbatch ...   # continue an interrupted run
#
# RESUME skips both phases and picks the checkpoint up where it stopped. SUBSET
# must repeat the original value, since that is what fixed total_steps.
# Training is retried TRAIN_ATTEMPTS times, RETRY_SLEEP apart, resuming each time.
# ============================================================================

set -Eeuo pipefail

WALL_H="${WALL_H:-92}"          # MUST match #SBATCH --time above
EVAL_H="${EVAL_H:-7}"           # extract + gap + caption 5000 + score + clipscore
SAFETY="${SAFETY:-0.88}"        # margin for startup, checkpoint writes, slowdown
CAL_STEPS="${CAL_STEPS:-10}"
TRAIN_ATTEMPTS="${TRAIN_ATTEMPTS:-3}"   # a crash resumes from the last checkpoint
RETRY_SLEEP="${RETRY_SLEEP:-1200}"      # 20 min, to let a stalled mount come back
TAG="${TAG:-Cloc_long}"
CK="stage2_vlm_${TAG}.pt"
CKPT="outputs/checkpoints/${CK}"
FLAGS=(--pool all257 --lambda-d 0.1 --lambda-s 1.0 --lambda-rank 1.0)

JOB_START=$(date +%s)
echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.telegram_env 2>/dev/null || true
CLIP_MODEL="${CLIP_MODEL:-$HOME/clip_b32}"

tg_notify() {
    [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]] && return 0
    curl -fsS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=$1" >/dev/null || true
}
on_error() {
    tg_notify "CLOC-LONG FAILED (job ${SLURM_JOB_ID}) line $1 exit $?
Checkpoint if any: ${CKPT} — re-run eval with SKIP_TRAIN=1, or resume with RESUME=1 SUBSET=<n>."
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/checkpoints outputs/predictions outputs/metrics outputs/embeddings
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader || true
python -c "import torch; assert torch.cuda.is_available(); print('[preflight]', torch.cuda.get_device_name(0))"

median_step_seconds() {   # $1 = log file -> median seconds between step lines
    grep -oE "^\[[0-9]{2}:[0-9]{2}:[0-9]{2}\].*step=[0-9]+/" "$1" \
      | sed -E 's/^\[([0-9]{2}):([0-9]{2}):([0-9]{2})\].*/\1 \2 \3/' \
      | awk '{t=$1*3600+$2*60+$3; if(p!=""){d=t-p; if(d<0)d+=86400; print d} p=t}' \
      | sort -n | awk '{a[NR]=$1} END{if(NR>0) print a[int((NR+1)/2)]; else print ""}'
}

# train_with_retry -- run training, and on failure resume from the checkpoint.
#
# Job 91319 died to a 3.5 h NFS stall; the checkpoint at step 100 was intact and
# the wall clock had 80 h left, but the script had already exited. Sleeping and
# resuming costs at most save_every_steps (50) plus the outage.
train_with_retry() {           # $1 = --subset-size value, $2 = resume? (0/1)
    local subset="$1" resume_flags=()
    [[ "${2:-0}" == "1" ]] && resume_flags=(--resume "${CKPT}")
    local attempt rc
    for attempt in $(seq 1 "${TRAIN_ATTEMPTS}"); do
        set +e
        python scripts/17_train_c5_distance.py "${FLAGS[@]}" \
            --subset-size "${subset}" --output-name "${CK}" ${resume_flags[@]+"${resume_flags[@]}"}
        rc=$?
        set -e
        [[ ${rc} -eq 0 ]] && return 0
        if [[ ${attempt} -eq ${TRAIN_ATTEMPTS} ]]; then
            echo "[long] attempt ${attempt} failed (rc=${rc}); giving up" >&2
            return ${rc}
        fi
        echo "[long] attempt ${attempt} failed (rc=${rc}); sleeping ${RETRY_SLEEP}s, then resuming from ${CKPT}" >&2
        tg_notify "CLOC-LONG attempt ${attempt} failed (rc=${rc}). Sleeping ${RETRY_SLEEP}s, then resuming from ${CKPT}."
        sleep "${RETRY_SLEEP}"
        [[ -f "${CKPT}" ]] && resume_flags=(--resume "${CKPT}")
    done
    return 1                   # only reachable if TRAIN_ATTEMPTS < 1
}

if [[ "${RESUME:-0}" == "1" ]]; then
  # Resume an interrupted run. SUBSET must be the ORIGINAL --subset-size: it is
  # what set total_steps, and the restored scheduler state is only meaningful
  # against the same horizon.
  [[ -f "${CKPT}" ]] || { echo "[long] RESUME=1 but ${CKPT} not found" >&2; exit 5; }
  : "${SUBSET:?RESUME=1 requires SUBSET (the original --subset-size, e.g. 48928)}"

  # The scale and rank pins are captured on the FIRST distance step of a process
  # (stage2_distance_sft.py:239,253), when the connector is still at C2-init. A
  # resumed process would recapture them from the already-moved connector and
  # silently re-anchor the scale pin to a COMPRESSED cloud -- the one thing the
  # pin exists to prevent. Recover the original values from the interrupted
  # run's log, or take them from the environment.
  if [[ -z "${BTRACE0:-}" || -z "${EFFRANK0:-}" ]]; then
      PINS_LOG="${PINS_LOG:-$(grep -l 'captured btrace0=' outputs/logs/cloc_long_*.out 2>/dev/null | tail -1)}"
      [[ -n "${PINS_LOG}" ]] || { echo "[long] no log with a pin capture; pass BTRACE0= and EFFRANK0=" >&2; exit 6; }
      BTRACE0="${BTRACE0:-$(grep -o 'btrace0=[0-9.]*' "${PINS_LOG}" | head -1 | cut -d= -f2)}"
      EFFRANK0="${EFFRANK0:-$(grep -o 'effrank0=[0-9.]*' "${PINS_LOG}" | head -1 | cut -d= -f2)}"
      echo "[long] pins recovered from ${PINS_LOG}"
  fi
  [[ -n "${BTRACE0}" && -n "${EFFRANK0}" ]] \
      || { echo "[long] could not determine btrace0/effrank0; pass them explicitly" >&2; exit 6; }
  FLAGS+=(--btrace0 "${BTRACE0}" --effrank0 "${EFFRANK0}")
  echo "[long] pins held at btrace0=${BTRACE0} effrank0=${EFFRANK0} (C2-init, not recaptured)"

  echo "[long] RESUME: continuing ${CKPT} with subset=${SUBSET}"
  tg_notify "CLOC-LONG resuming from ${CKPT} (subset ${SUBSET}) on $(hostname)."
  train_with_retry "${SUBSET}" 1
  tg_notify "CLOC-LONG: training done. Evaluating."

elif [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  [[ -f "${CKPT}" ]] && { echo "[long] refusing to overwrite ${CKPT}; use SKIP_TRAIN=1, RESUME=1, or a new TAG" >&2; exit 3; }

  # ---------------- PHASE A: calibrate --------------------------------------
  if [[ -z "${SPS:-}" ]]; then
      CAL_LOG="outputs/logs/cloc_calib_${SLURM_JOB_ID}.log"
      tg_notify "CLOC-LONG calibrating (${CAL_STEPS} steps) on $(hostname)…"
      echo "[long] PHASE A: ${CAL_STEPS} calibration steps"
      python scripts/17_train_c5_distance.py "${FLAGS[@]}" \
          --subset-size $(( CAL_STEPS * 32 )) \
          --output-name "_calib_${SLURM_JOB_ID}.pt" 2>&1 | tee "${CAL_LOG}"
      rm -f "outputs/checkpoints/_calib_${SLURM_JOB_ID}.pt"
      SPS=$(median_step_seconds "${CAL_LOG}")
      [[ -z "${SPS}" || "${SPS}" -lt 30 ]] && { echo "[long] calibration unusable, defaulting to 180"; SPS=180; }
      echo "[long] measured ${SPS} s/step"
  else
      echo "[long] using supplied SPS=${SPS} s/step (calibration skipped)"
  fi

  # ---------------- PHASE B: size the run, then train -----------------------
  ELAPSED_H=$(python3 -c "import sys;print(($(date +%s)-${JOB_START})/3600.0)")
  STEPS=$(python3 -c "
avail=(${WALL_H}-${EVAL_H}-${ELAPSED_H})*3600*${SAFETY}
print(max(200, int(avail/${SPS})))")
  SUBSET=$(( STEPS * 32 ))
  EST_H=$(python3 -c "print(round(${STEPS}*${SPS}/3600,1))")
  echo "[long] PHASE B: ${STEPS} steps (subset=${SUBSET}) ~${EST_H} h at ${SPS} s/step"
  tg_notify "CLOC-LONG sized: ${STEPS} steps (subset ${SUBSET}) ~${EST_H} h
Measured ${SPS} s/step on $(hostname). Annealed cosine, wall ${WALL_H} h."

  train_with_retry "${SUBSET}" 0
  tg_notify "CLOC-LONG: training done (${STEPS} steps). Evaluating."
else
  echo "[long] SKIP_TRAIN=1: evaluating existing ${CKPT}"
  [[ -f "${CKPT}" ]] || { echo "[long] ${CKPT} not found" >&2; exit 4; }
fi

# ---------------- Evaluation ----------------------------------------------
python scripts/07_extract_projected.py --condition "${TAG}" --connector-override "${CKPT}"
python scripts/03_compute_gap.py --condition "${TAG}"
python scripts/08_run_captioning.py --condition "${TAG}" --vlm-checkpoint "${CKPT}"
python scripts/09_score_captions.py --condition "${TAG}"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
    --condition "${TAG}" --clip-model "${CLIP_MODEL}" || echo "[long] clipscore failed (non-fatal)"

python - <<PYEOF
import json, pathlib
def g(p,*k):
    f=pathlib.Path(p)
    if not f.exists(): return "n/a"
    d=json.loads(f.read_text())
    for x in k:
        d=d.get(x) if isinstance(d,dict) else None
        if d is None: return "n/a"
    return round(d,4) if isinstance(d,float) else d
print()
print(f"{'condition':22s} {'CLIP':>8s} {'SPICE':>8s} {'G_mu':>9s}")
for lab,t in [("Cloc (450 steps)","Cloc"),("${TAG}","${TAG}")]:
    print(f"{lab:22s} {str(g(f'outputs/metrics/clipscore_{t}.json','CLIPScore')):>8s} "
          f"{str(g(f'outputs/metrics/captioning_{t}.json','scores','SPICE')):>8s} "
          f"{str(g(f'outputs/metrics/gap_{t}.json','spec_metrics','G_mu')):>9s}")
print("\nreference ceiling (dd256): LLaVA-1.5-7B CLIPScore 0.7909")
PYEOF

tg_notify "CLOC-LONG done: ${TAG} (job ${SLURM_JOB_ID})"
echo "[slurm] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
