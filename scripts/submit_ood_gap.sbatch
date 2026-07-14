#!/bin/bash
#SBATCH --job-name=mllm-ood-gap
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/oodgap_%j.out
#SBATCH --error=outputs/logs/oodgap_%j.err
set -Eeuo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap

# ---------------------------------------------------------------------------
# MODALITY GAP on the OOD (NoCaps / Open Images) set, for BOTH models.
#
#   baseline : C3pinr  (gap left open)
#   treatment: Cloc    (location closed, scale + rank pinned)
#
# Text side = frozen LLaMA embeddings of one human NoCaps caption per image
# (identical across conditions -> anchors the text centroid). Image side = each
# model's connector on the 300 NoCaps images. So any G_mu difference between the
# two is 100% the image-side geometry the connector produces off-distribution.
#
# Pipeline: prep_ood_refs (recover captions + manifest) -> 07_extract_projected
# (--data-config configs/data_ood.yaml, separate out-dir) -> 03_compute_gap.
# Everything writes to *_ood dirs so in-distribution embeddings/metrics untouched.
#
# Run from repo root: sbatch scripts/submit_ood_gap.sbatch
# ---------------------------------------------------------------------------

DATACFG=configs/data_ood.yaml
EMB=outputs/embeddings_ood
MET=outputs/metrics_ood
CONDS="C3pinr Cloc"        # baseline first
mkdir -p "$EMB" "$MET" outputs/logs

# 1) Recover NoCaps references + build the diagnostic manifest (idempotent).
if [ ! -f data/nocaps/ood_manifest_300.json ]; then
  echo "==================== RECOVER OOD REFS -> MANIFEST ===================="
  python scripts/prep_ood_refs.py --hf-dataset lmms-lab/NoCaps \
      --hf-split validation --out-root data/nocaps --n 300 --seed 42 --also-coco-json
fi
if [ ! -f data/nocaps/ood_manifest_300.json ]; then
  echo "[fatal] manifest still missing -- prep_ood_refs.py failed" >&2; exit 1
fi

# 2) Extract projected embeddings + 3) compute gap, per condition.
for c in $CONDS; do
  ck="outputs/checkpoints/stage2_vlm_${c}.pt"
  if [ ! -f "$ck" ]; then echo "[skip] $c (no checkpoint $ck)"; continue; fi
  echo "==================== EXTRACT (OOD)  $c ===================="
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/07_extract_projected.py \
      --condition "$c" --data-config "$DATACFG" --out-dir "$EMB"
  echo "==================== GAP (OOD)  $c ===================="
  python scripts/03_compute_gap.py --condition "$c" \
      --embeddings-dir "$EMB" --out-dir "$MET"
done

# ---- side-by-side gap decomposition: baseline vs Cloc on the OOD set ----
python - <<'PY'
import json, os
conds = ["C3pinr", "Cloc"]
keys  = [("G_mu","location  G_mu"), ("trace_image","scale     tr(Sig_x)"),
         ("eff_rank_image","rank      effrank_x"), ("alpha_image","alpha_img"),
         ("js_divergence_angular","orient    JS_ang"),
         ("knn_mixing_rate_k20","mix       kNN@20")]
S = {}
for c in conds:
    f = f"outputs/metrics_ood/gap_{c}.json"
    S[c] = json.load(open(f))["spec_metrics"] if os.path.exists(f) else None
print("\n%-22s %14s %14s %12s" % ("metric (OOD)", conds[0], conds[1], "Cloc-base"))
print("-"*66)
for k, label in keys:
    a = S[conds[0]][k] if S[conds[0]] and k in S[conds[0]] else None
    b = S[conds[1]][k] if S[conds[1]] and k in S[conds[1]] else None
    if a is None or b is None:
        print("%-22s %14s %14s %12s" % (label, a, b, "--")); continue
    print("%-22s %14.4f %14.4f %+12.4f" % (label, a, b, b-a))
PY

echo "[ood-gap] done"
