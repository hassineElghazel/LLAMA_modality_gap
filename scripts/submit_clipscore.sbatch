#!/bin/bash
#SBATCH --job-name=clipscore
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/logs/clipscore_%j.out
#SBATCH --error=outputs/logs/clipscore_%j.err

# Reference-free CLIPScore (scripts/15_clipscore.py) for every C-series condition.
# NOTE: no `set -e` — a single bad condition (e.g. a corrupted predictions file)
# must not abort the whole job; each condition is run independently and its
# failure is logged.
set -uo pipefail

echo "[clipscore] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

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

echo "[clipscore] GPU:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || echo "[clipscore] nvidia-smi failed -> CPU fallback"

CONDITIONS=(C0_random C1_stage2 C2_stage1 C3_stage2 C4_lam0p1 C4_lam0p5)

for c in "${CONDITIONS[@]}"; do
    echo "=== ${c} ==="
    python scripts/15_clipscore.py --condition "${c}" \
        || echo "[clipscore] ${c} FAILED (see traceback above)"
done

echo "[clipscore] ===== summary ====="
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

echo "[clipscore] complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
