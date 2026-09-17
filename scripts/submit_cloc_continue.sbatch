#!/bin/bash
#SBATCH --job-name=mllm-cloc-cont
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=92:00:00
#SBATCH --output=outputs/logs/cloc_cont_%j.out
#SBATCH --error=outputs/logs/cloc_cont_%j.err

# ============================================================================
# Continue the finished Cloc_long run onto a larger horizon, in ONE job.
#
#   sbatch scripts/submit_cloc_continue.sbatch              # -> 80,000 items
#   SUBSET=100000 TAG=Cloc_100k sbatch scripts/...          # -> 100,000 items
#
# WHY CONTINUE RATHER THAN RETRAIN.  At the measured 160-175 s/step a FRESH
# 80,000-item run is 2,500 steps = ~115 h of training against a 92 h wall, and a
# fresh 100,000 is ~143 h. Neither fits one allocation. Cloc_long has already
# annealed 1,529 of those steps, so only the remainder is left:
#
#     items     horizon   steps left   train      +eval   restart lr
#    64,000       2,000          471    ~22 h     ~26 h   14% of peak
#    80,000       2,500          971    ~45 h     ~49 h   35% of peak   <- default
#   100,000       3,125        1,596    ~73 h     ~77 h   54% of peak
#
# 80,000 is the default because it leaves ~40 h of slack in the 92 h wall. At
# 100,000 the margin is ~10 h, which one slow node or one NFS retry sleep
# (job 91319 lost 3.5 h to a stalled mount) is enough to eat, forcing the split
# this script exists to avoid.
#
# NO DATA IS REPLAYED.  LLaVAInstruct150KDataset shuffles on a fixed seed and
# slices rows[:limit], so a larger limit is a SUPERSET of what was already seen.
# 17_train_c5_distance.py:231 advances the iterator past step*eff_batch items,
# so the run consumes rows[48928:SUBSET] and nothing twice. This holds ONLY at
# the original seed, so --seed is deliberately never passed here.
#
# THE LEARNING RATE IS A WARM RESTART (SGDR).  Cloc_long annealed to lr=0 over a
# 1,529-step cosine. On a 2,500-step horizon step 1,529 sits at 60% of the new
# cosine, so training resumes at ~35% of the 2.0e-4 peak and anneals to 0 again
# over the remaining 971 steps. Two annealed segments, not one unbroken
# schedule. The preflight prints the lr the first resumed step should show, so
# the log can be checked against it. Expect L_ar to spike briefly, and expect
# btrace/effrank to wander for a few tens of steps before the pins pull them
# back; L_scale and L_rank sat near zero at the end of Cloc_long precisely
# because it had converged.
#
# THE PINS ARE NOT RECAPTURED.  The scale and rank anchors are captured on the
# first distance step of a PROCESS (stage2_distance_sft.py:239,253), when the
# connector is still at C2-init. A resumed process would re-anchor the scale pin
# to the already-moved connector, which is the one thing the pin exists to
# prevent: that is how job 91362 got 4451.2/21.60 instead of 4767.3/23.31. They
# are passed explicitly and cross-checked against the sidecar Cloc_long wrote.
#
# Overrides: SUBSET, TAG, SRC_TAG, BTRACE0, EFFRANK0, EVAL_H, TRAIN_ATTEMPTS.
#   RESUME=1 sbatch ...       # the wall cut it short; continue in place
#   SKIP_TRAIN=1 sbatch ...   # re-run evaluation only
# ============================================================================

set -Eeuo pipefail

SUBSET="${SUBSET:-80000}"       # fixes the cosine horizon: SUBSET/eff_batch steps
SRC_TAG="${SRC_TAG:-Cloc_long}" # the finished run this continues
TAG="${TAG:-Cloc_80k}"
BTRACE0="${BTRACE0:-4767.3}"    # C2-init anchors, captured fresh by jobs 91318/91319
EFFRANK0="${EFFRANK0:-23.31}"
EVAL_H="${EVAL_H:-5}"           # job 91363's eval took 3 h 34 m; 5 is the safe read
WALL_H="${WALL_H:-92}"          # MUST match #SBATCH --time above
TRAIN_ATTEMPTS="${TRAIN_ATTEMPTS:-3}"
RETRY_SLEEP="${RETRY_SLEEP:-1200}"

CK="stage2_vlm_${TAG}.pt"
CKPT="outputs/checkpoints/${CK}"
SRC_CKPT="outputs/checkpoints/stage2_vlm_${SRC_TAG}.pt"
SRC_SIDE="outputs/checkpoints/stage2_vlm_${SRC_TAG}_distance.json"
FLAGS=(--pool all257 --lambda-d 0.1 --lambda-s 1.0 --lambda-rank 1.0
       --btrace0 "${BTRACE0}" --effrank0 "${EFFRANK0}")

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
    tg_notify "CLOC-CONT FAILED (job ${SLURM_JOB_ID}) line $1 exit $?
Checkpoint if any: ${CKPT} — resume with RESUME=1, or evaluate with SKIP_TRAIN=1."
}
trap 'on_error ${LINENO}' ERR

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/checkpoints outputs/predictions outputs/metrics outputs/embeddings
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader || true
python -c "import torch; assert torch.cuda.is_available(); print('[preflight]', torch.cuda.get_device_name(0))"

# ---------------- preflight ------------------------------------------------
# Every number this run depends on is checked against the artefacts on disk
# rather than against the comments above. A mismatch aborts in the first minute
# instead of surfacing after two days of training on a wrong premise.
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  RESUME_SRC="${SRC_CKPT}"
  [[ "${RESUME:-0}" == "1" ]] && RESUME_SRC="${CKPT}"
  [[ -f "${RESUME_SRC}" ]] || { echo "[cont] checkpoint ${RESUME_SRC} not found" >&2; exit 5; }
  [[ -f "${SRC_SIDE}" ]] || { echo "[cont] sidecar ${SRC_SIDE} not found" >&2; exit 5; }

  SUBSET="${SUBSET}" BTRACE0="${BTRACE0}" RESUME_SRC="${RESUME_SRC}" SRC_SIDE="${SRC_SIDE}" \
  TAG="${TAG}" EVAL_H="${EVAL_H}" \
  python - <<'PYEOF'
import json, math, os, sys
from pathlib import Path
import yaml, torch

subset  = int(os.environ["SUBSET"])
btrace0 = float(os.environ["BTRACE0"])

# 1. the anchor about to be passed must be the one the ORIGINAL run captured.
#    Re-anchoring the scale pin to a moved connector is the job-91362 bug.
side = json.loads(Path(os.environ["SRC_SIDE"]).read_text())
if side.get("btrace0") is None:
    sys.exit("[cont] sidecar records no btrace0; the source run had no scale pin.")
if abs(float(side["btrace0"]) - btrace0) > 0.05:
    sys.exit(f"[cont] btrace0 mismatch: passing {btrace0}, sidecar says {side['btrace0']}. "
             "Aborting rather than re-anchoring the scale pin.")
print(f"[cont] btrace0 {btrace0} agrees with the {side['steps']}-step sidecar")

# 2. horizon, resume point, and exactly which rows are new
cfg   = yaml.safe_load(Path("configs/training_c5.yaml").read_text())
eff   = cfg["batch"]["per_device_batch_size"] * max(
            1, int(cfg["batch"].get("gradient_accumulation_steps", 1)))
blob  = torch.load(os.environ["RESUME_SRC"], map_location="cpu", weights_only=False)
done  = int(blob.get("step", 0))
if "optimizer" not in blob or "scheduler" not in blob:
    sys.exit("[cont] checkpoint carries no optimizer/scheduler state; "
             "a resume would restart the schedule from zero.")
total = math.ceil(subset / eff)
if done >= total:
    sys.exit(f"[cont] nothing to do: checkpoint is at step {done:,}, "
             f"horizon is {total:,}. Raise SUBSET.")
print(f"[cont] eff_batch={eff}  horizon={total:,} steps  resume at {done:,}  "
      f"remaining {total-done:,}")
print(f"[cont] items: {done*eff:,} already consumed, {subset-done*eff:,} new "
      f"(rows[{done*eff}:{subset}]), 0 replayed")

# 3. the learning rate the first resumed step should print, so the log can be
#    checked against it (cosine_with_warmup, trainer_utils.py:14)
wr   = cfg["schedule"]["warmup_ratio"]
base = float(cfg["optimizer"]["lr"])
w    = int(total * wr)
frac = (0.5 * (1 + math.cos(math.pi * (done - w) / max(1, total - w)))
        if done >= w else done / max(1, w))
print(f"[cont] warm restart: step {done:,} resumes at lr={frac*base:.2e} "
      f"({100*frac:.0f}% of the {base:.1e} peak), annealing to 0 by step {total:,}")

# 4. the manifest must really hold `subset` rows, at the SAME seed that made
#    the superset property true
sys.path.insert(0, ".")
from src.data.llava_instruct_loader import LLaVAInstruct150KDataset
d = yaml.safe_load(Path("configs/data.yaml").read_text())["llava_instruct_150k"]
n = len(LLaVAInstruct150KDataset(root=d["local_path"], image_root=d["image_root"],
                                 limit=None, shuffle=True, seed=cfg["seed"]))
print(f"[cont] manifest holds {n:,} rows at seed {cfg['seed']}")
if n < subset:
    sys.exit(f"[cont] asked for {subset:,} items but only {n:,} exist. Lower SUBSET.")

# 5. the evaluation scripts validate --condition against a hardcoded CONDITIONS
#    list. An unregistered tag aborts the eval on its first line, AFTER training
#    has finished: that is how job 91561 lost its evaluation after 33 h. Parse the
#    lists rather than importing the scripts, which would pull in torch.
import ast
tag = os.environ["TAG"]
for script in ("scripts/07_extract_projected.py", "scripts/03_compute_gap.py"):
    known = None
    for node in ast.walk(ast.parse(Path(script).read_text())):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "CONDITIONS" for t in node.targets):
            known = ast.literal_eval(node.value)
            break
    if known is None:
        sys.exit(f"[cont] could not read CONDITIONS from {script}; aborting rather "
                 "than training into an eval that may reject the tag.")
    if tag not in known:
        sys.exit(f"[cont] {script} does not know condition '{tag}'. Add it to its "
                 "CONDITIONS before training, or the evaluation will fail after "
                 "the run completes.")
print(f"[cont] condition '{tag}' is registered in both evaluation scripts")

est = (total - done) * 168.6 / 3600
print(f"[cont] estimate: {est:.1f} h training at 168.6 s/step, "
      f"+{os.environ.get('EVAL_H','5')} h evaluation")
PYEOF
fi

# train_with_retry -- job 91319 died to a 3.5 h NFS stall with 80 h of wall left
# and an intact checkpoint, but the script had already exited. Sleeping and
# resuming costs at most save_every_steps (50) plus the outage. CKPT always
# exists here (it was copied, or RESUME=1 required it), so --resume is
# unconditional and correct on every attempt.
train_with_retry() {
    local attempt rc
    for attempt in $(seq 1 "${TRAIN_ATTEMPTS}"); do
        set +e
        python scripts/17_train_c5_distance.py "${FLAGS[@]}" \
            --subset-size "${SUBSET}" --output-name "${CK}" --resume "${CKPT}"
        rc=$?
        set -e
        [[ ${rc} -eq 0 ]] && return 0
        if [[ ${attempt} -eq ${TRAIN_ATTEMPTS} ]]; then
            echo "[cont] attempt ${attempt} failed (rc=${rc}); giving up" >&2
            return ${rc}
        fi
        echo "[cont] attempt ${attempt} failed (rc=${rc}); sleeping ${RETRY_SLEEP}s, then resuming" >&2
        tg_notify "CLOC-CONT attempt ${attempt} failed (rc=${rc}). Sleeping ${RETRY_SLEEP}s, then resuming from ${CKPT}."
        sleep "${RETRY_SLEEP}"
    done
    return 1
}

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  echo "[cont] SKIP_TRAIN=1: evaluating existing ${CKPT}"
  [[ -f "${CKPT}" ]] || { echo "[cont] ${CKPT} not found" >&2; exit 4; }
else
  if [[ "${RESUME:-0}" == "1" ]]; then
    echo "[cont] RESUME: continuing ${CKPT} in place"
  else
    # The source checkpoint backs numbers already reported (CLIPScore 0.7022 and
    # the dd256 table). It is copied, never trained in place.
    [[ -f "${CKPT}" ]] && { echo "[cont] ${CKPT} already exists; use RESUME=1, SKIP_TRAIN=1, or a new TAG" >&2; exit 3; }
    cp "${SRC_CKPT}" "${CKPT}"
    echo "[cont] copied ${SRC_CKPT} -> ${CKPT} (source left untouched)"
  fi

  echo "[cont] pins held at btrace0=${BTRACE0} effrank0=${EFFRANK0} (C2-init, not recaptured)"
  tg_notify "CLOC-CONT starting on $(hostname): ${TAG}, horizon ${SUBSET} items."
  train_with_retry
  tg_notify "CLOC-CONT: training done. Evaluating."
fi

# ---------------- Evaluation (brief set, n=5000, 64 tokens) -----------------
python scripts/07_extract_projected.py --condition "${TAG}" --connector-override "${CKPT}"
python scripts/03_compute_gap.py --condition "${TAG}"
python scripts/08_run_captioning.py --condition "${TAG}" --vlm-checkpoint "${CKPT}"
python scripts/09_score_captions.py --condition "${TAG}"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
    --condition "${TAG}" --clip-model "${CLIP_MODEL}" || echo "[cont] clipscore failed (non-fatal)"

TAG="${TAG}" SUBSET="${SUBSET}" python - <<'PYEOF'
import json, os, pathlib
M = pathlib.Path("outputs/metrics")
def g(p, *k):
    f = M / p
    if not f.exists(): return "n/a"
    d = json.loads(f.read_text())
    for x in k:
        d = d.get(x) if isinstance(d, dict) else None
        if d is None: return "n/a"
    return round(d, 4) if isinstance(d, float) else d
tag = os.environ["TAG"]
rows = [("Cloc (450 steps)", "Cloc"),
        ("Cloc_long (48,928)", "Cloc_long"),
        (f"{tag} ({int(os.environ['SUBSET']):,})", tag)]
print()
print(f"{'condition':26s} {'CLIP':>8s} {'SPICE':>8s} {'G_mu':>9s}")
for lab, t in rows:
    print(f"{lab:26s} {str(g(f'clipscore_{t}.json','CLIPScore')):>8s} "
          f"{str(g(f'captioning_{t}.json','scores','SPICE')):>8s} "
          f"{str(g(f'gap_{t}.json','spec_metrics','G_mu')):>9s}")
print("\nreference ceiling (dd256): LLaVA-1.5-7B CLIPScore 0.7909")
PYEOF

tg_notify "CLOC-CONT done: ${TAG} (job ${SLURM_JOB_ID})"
echo "[slurm] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
