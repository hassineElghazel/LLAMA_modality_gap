#!/bin/bash
#SBATCH --job-name=clipscore-c5
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/logs/clipscore_c5_%j.out
#SBATCH --error=outputs/logs/clipscore_c5_%j.err

# Reference-free CLIPScore (scripts/15_clipscore.py) for the C3/C4/C5 lambda=0.5
# comparison. Re-runs C3 and C4_lam0p5 alongside C5_lam0p5 with the IDENTICAL
# backbone + env so all three CLIPScores come from one consistent run (the C4
# leg should reproduce its prior 0.604 — a sanity check that the invocation
# matches). CLIPScore is reference-free, so it has no caption-length confound
# (unlike BLEU/CIDEr) — the key metric given C5's verbose captions.
#
# NOTE: no `set -e` — a single bad condition must not abort the whole job.
set -uo pipefail

echo "[clipscore-c5] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/metrics

# CLIP ViT-B/32 is cached from the login-node download; compute nodes are offline.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Idempotent safetensors fix: recent transformers refuses torch.load on a .bin
# checkpoint unless torch>=2.6 (CVE-2025-32434); force the .safetensors path.
grep -q use_safetensors scripts/15_clipscore.py || \
    sed -i 's/CLIPModel\.from_pretrained(args\.clip_model)/CLIPModel.from_pretrained(args.clip_model, use_safetensors=True)/' \
        scripts/15_clipscore.py

echo "[clipscore-c5] GPU:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[clipscore-c5] nvidia-smi failed -> CPU fallback"

# Local CLIP ViT-B/32 dir (same backbone used for the intervention CLIPScores),
# loaded offline — avoids the hub-conversion call that breaks under HF_HUB_OFFLINE
# when only a .bin is cached for the HF id.
CLIP_MODEL="$HOME/clip_b32"
if [ ! -d "$CLIP_MODEL" ]; then
    echo "[clipscore-c5] FATAL: local CLIP dir $CLIP_MODEL not found" >&2
    exit 1
fi

# Only the new condition; C3/C4/C5_lam0p5 CLIPScores already exist.
CONDITIONS=(C5_lam0p1)

for c in "${CONDITIONS[@]}"; do
    echo "=== ${c} ==="
    if [ ! -f "outputs/predictions/captions_${c}.json" ]; then
        echo "[clipscore-c5] ${c}: predictions file missing, skipping"
        continue
    fi
    python scripts/15_clipscore.py --condition "${c}" --clip-model "$CLIP_MODEL" \
        || echo "[clipscore-c5] ${c} FAILED (see traceback above)"
done

echo "[clipscore-c5] ===== summary ====="
for c in "${CONDITIONS[@]}"; do
    f="outputs/metrics/clipscore_${c}.json"
    if [ -f "$f" ]; then
        python - "$f" "$c" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{sys.argv[2]}: CLIPScore={d['CLIPScore']:.4f} "
      f"(mean_cos={d['mean_cosine']:.4f}, n={d['n_evaluated']}, {d['clip_model']})")
PY
    fi
done

echo "[clipscore-c5] complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
