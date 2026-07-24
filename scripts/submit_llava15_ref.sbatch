#!/bin/bash
#SBATCH --job-name=llava15-ref
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/llava15ref_%j.out
#SBATCH --error=outputs/logs/llava15ref_%j.err

# ---------------------------------------------------------------------------
# LLaVA-1.5-7B REFERENCE captioning on the EXACT dd256 set (n=1300), scored with
# the same non-circular ViT-B/32 CLIPScore. Establishes the fully-resourced
# CEILING next to the diagnostic models. Resilient to baldur's intermittent Xid
# (retry loop; inference is a single fast pass so each retry restarts fresh).
#
# Pre-req (run ONCE, online): download the model to the HF cache, e.g.
#   python -c "from huggingface_hub import snapshot_download; \
#              snapshot_download('llava-hf/llava-1.5-7b-hf')"
# or point LLAVA_MODEL at a local snapshot dir.
#
# Run from repo root: sbatch scripts/submit_llava15_ref.sbatch
# ---------------------------------------------------------------------------

set -Eeuo pipefail

echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || true)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/predictions outputs/metrics

DESC_CFG=configs/description_eval.yaml
COND=llava15_dd256
LLAVA_MODEL="${LLAVA_MODEL:-llava-hf/llava-1.5-7b-hf}"
CLIP_MODEL="${CLIP_MODEL:-$HOME/clip_b32}"
MAX_TRIES=4

# --- Preflight: GPU responsive (fail fast if baldur handed us a dead device) ---
python -c "
import sys, torch
if not torch.cuda.is_available(): print('[preflight] no CUDA'); sys.exit(1)
x = torch.randn(2048, 2048, device='cuda'); _ = (x @ x).sum().item()
torch.cuda.synchronize(); print('[preflight] GPU op OK on', torch.cuda.get_device_name(0))
" || { echo '[llava15] GPU preflight FAILED (bad allocation); exiting for resubmit' >&2; exit 1; }

# --- Preflight: transformers has LLaVA + model is available (offline-safe) ---
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "
import sys
try:
    from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoConfig
except Exception as e:
    print('[preflight] transformers lacks LlavaForConditionalGeneration:', e); sys.exit(2)
try:
    AutoConfig.from_pretrained('${LLAVA_MODEL}')
except Exception as e:
    print('[preflight] model not cached locally (${LLAVA_MODEL}). Pre-download it once online:');
    print(\"  python -c \\\"from huggingface_hub import snapshot_download; snapshot_download('llava-hf/llava-1.5-7b-hf')\\\"\")
    print('  or set LLAVA_MODEL=/path/to/local/snapshot'); print('detail:', e); sys.exit(3)
print('[preflight] transformers+LLaVA OK, model reachable: ${LLAVA_MODEL}')
" || { echo '[llava15] model/transformers preflight FAILED' >&2; exit 1; }

[ -d "$CLIP_MODEL" ] || echo "[llava15] WARN: $CLIP_MODEL missing -> clipscore step will fail (non-fatal)"
grep -q use_safetensors scripts/15_clipscore.py || \
    sed -i 's/CLIPModel\.from_pretrained(args\.clip_model)/CLIPModel.from_pretrained(args.clip_model, use_safetensors=True)/' scripts/15_clipscore.py

# --- Captioning: retry-from-scratch on GPU hiccup (single fast inference pass) ---
attempt=1
success=0
while [ "$attempt" -le "$MAX_TRIES" ]; do
    echo "[llava15] caption attempt ${attempt}/${MAX_TRIES}"
    set +e
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/22_run_reference_captioning.py \
        --config "$DESC_CFG" --condition "$COND" --model-id "$LLAVA_MODEL"
    rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then success=1; break; fi
    echo "[llava15] attempt ${attempt} failed (rc=${rc}); sleep 60s then retry"
    sleep 60
    attempt=$((attempt + 1))
done
[ "$success" -eq 1 ] || { echo "[llava15] all ${MAX_TRIES} caption attempts failed" >&2; exit 1; }

# --- CLIPScore (same non-circular ViT-B/32 as every diagnostic run) ---
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
    --config "$DESC_CFG" --condition "$COND" --clip-model "$CLIP_MODEL" \
    || echo "[llava15] clipscore FAILED (non-fatal; rerun 15_clipscore.py --config $DESC_CFG --condition $COND)"

echo "[llava15-ref] done -> outputs/predictions/captions_${COND}.json + outputs/metrics/clipscore_${COND}.json"
echo "[llava15-ref] CHAIR: run locally alongside the diagnostic conditions"
