#!/bin/bash
#SBATCH --job-name=mllm-desc-eval
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/desceval_%j.out
#SBATCH --error=outputs/logs/desceval_%j.err
set -Eeuo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap

# ---------------------------------------------------------------------------
# TASK-MATCHED re-evaluation of the captioning ablation.
#
# The models were trained on LLaVA visual instruction following (detailed grounded
# responses), so we re-prompt with a VERBATIM LLaVA detail instruction
# (configs/description_eval.yaml) and score with the REFERENCE-FREE grounding family:
#   - CLIPScore : generated text vs image
#   - CHAIR     : mentioned objects vs the image's full object set
#                 (CHAIR_s, CHAIR_i, object recall, objects/caption, length)
# NO BLEU/CIDEr (short-reference metrics = the earlier train/eval mismatch).
#
# Non-destructive: outputs use a _dd (detailed-description) tag, so the old
# brief-prompt captioning results are preserved for comparison.
# Inference only. Run from repo root: sbatch scripts/submit_description_eval.sbatch
# ---------------------------------------------------------------------------

CFG=configs/description_eval.yaml
CKPT=outputs/checkpoints
CLIP_MODEL="${CLIP_MODEL:-/nfs/home/elghazel/clip_b32}"
CONDS="C3pinr C5bp_lam0p1 C4bp_lam0p9 Cscale1500 Crank15 Cloc Corient Clocorient"

DD_TAGS=""
for c in $CONDS; do
  ck="$CKPT/stage2_vlm_${c}.pt"
  if [ ! -f "$ck" ]; then echo "[skip] $c (no checkpoint $ck)"; continue; fi
  tag="${c}_dd256"
  DD_TAGS="$DD_TAGS $tag"
  echo "==================== DESCRIBE  $tag ===================="
  # detailed-description generation (resumable via existing captions_<tag>.json)
  python scripts/08_run_captioning.py --config "$CFG" --condition "$tag" --vlm-checkpoint "$ck"
  # reference-free grounding: CLIPScore (text vs image)
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
      --config "$CFG" --condition "$tag" --clip-model "$CLIP_MODEL" \
      || echo "[desc] $tag clipscore FAILED (non-fatal; rerun 15_clipscore.py --condition $tag)"
done

# ---- CHAIR across the detailed-description ablation (prints its own table) ----
if [ -n "$DD_TAGS" ]; then
  echo "==================== CHAIR (detailed-description ablation) ===================="
  python scripts/19_chair.py --conditions $DD_TAGS --baseline C3pinr_dd256 \
      --config "$CFG" --out outputs/metrics/chair_summary_dd256.json
fi

# ---- CLIPScore summary table (CHAIR table printed above by 19_chair.py) ----
python - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob("outputs/metrics/clipscore_*_dd256.json")):
    cond = os.path.basename(f)[len("clipscore_"):-5]
    rows.append((cond, json.load(open(f)).get("CLIPScore")))
print("\n%-20s %10s" % ("condition (detailed)", "CLIPScore"))
print("-" * 32)
for c, cs in rows:
    print("%-20s %10s" % (c, ("%.4f" % cs) if cs is not None else "--"))
print("\nCHAIR (CHAIR_s / CHAIR_i / recall) -> table above + outputs/metrics/chair_summary_dd256.json")
PY

echo "[desc-eval] done"
