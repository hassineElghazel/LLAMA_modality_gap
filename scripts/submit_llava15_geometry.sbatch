#!/bin/bash
#SBATCH --job-name=llava15-geo
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/llava15geo_%j.out
#SBATCH --error=outputs/logs/llava15geo_%j.err

# ---------------------------------------------------------------------------
# Extract LLaVA-1.5-7B's OWN connector-output gap on the 1300 dd256 pairs.
# Gates the "does location-closure generalise to a competent model" experiment:
#   normalised G_mu near ~2.7  -> location still OPEN  -> fine-tune experiment worth it
#   normalised G_mu near ~0.45 -> full training already CLOSED it (a result in itself)
# No training; ~1300 forward passes -> short. Resilient to baldur's Xid.
#
# Run from repo root: sbatch scripts/submit_llava15_geometry.sbatch
# ---------------------------------------------------------------------------

set -Eeuo pipefail
echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || true)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/metrics outputs/embeddings_1300

LLAVA_MODEL="${LLAVA_MODEL:-llava-hf/llava-1.5-7b-hf}"
MANIFEST="data/dd256/gap_manifest_1300.json"

# --- preflight: GPU responsive ---
python -c "
import sys, torch
if not torch.cuda.is_available(): print('[preflight] no CUDA'); sys.exit(1)
x = torch.randn(2048,2048,device='cuda'); _=(x@x).sum().item(); torch.cuda.synchronize()
print('[preflight] GPU op OK on', torch.cuda.get_device_name(0))
" || { echo '[geo] GPU preflight FAILED; exit for resubmit' >&2; exit 1; }

# --- preflight: transformers has LLaVA + model cached + manifest present ---
test -f "$MANIFEST" || { echo "[geo] FATAL: $MANIFEST missing (run prep_dd256_gap_manifest.py)" >&2; exit 1; }
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "
import sys
try:
    from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoConfig
except Exception as e:
    print('[preflight] transformers lacks LLaVA:', e); sys.exit(2)
try:
    AutoConfig.from_pretrained('${LLAVA_MODEL}')
except Exception as e:
    print('[preflight] model not cached (${LLAVA_MODEL}); pre-download once online:')
    print(\"  python -c \\\"from huggingface_hub import snapshot_download; snapshot_download('llava-hf/llava-1.5-7b-hf')\\\"\")
    print('detail:', e); sys.exit(3)
print('[preflight] transformers+LLaVA OK, model reachable')
" || { echo '[geo] model/transformers preflight FAILED' >&2; exit 1; }

# --- retry-from-scratch on GPU hiccup (fast inference pass) ---
attempt=1
until [ "$attempt" -gt 4 ]; do
    echo "[geo] attempt ${attempt}/4"
    if HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/23_extract_llava15_geometry.py \
        --data-config configs/data_1300.yaml --model-id "$LLAVA_MODEL" --condition llava15; then
        echo "[geo] done -> outputs/metrics/gap_llava15.json"; exit 0
    fi
    echo "[geo] attempt ${attempt} failed; sleep 60 then retry"; sleep 60
    attempt=$((attempt+1))
done
echo "[geo] all attempts failed" >&2; exit 1
