#!/bin/bash
#SBATCH --job-name=t1-vlmgeo
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/t1_vlmgeo_%j.out
#SBATCH --error=outputs/logs/t1_vlmgeo_%j.err

# ---------------------------------------------------------------------------
# TIER 1 / job 3.  GENERALITY, WITHOUT TRAINING ANYTHING.
#
# WHY.  The four axes are STATISTICS OF A REPRESENTATION, so measuring them
# needs forward passes, not training. That makes cross-model generality cheap:
# characterise the decomposition on other released VLMs on the SAME 1300 pairs.
#
# You already own datapoint one and have not framed it as generality:
#   LLaVA-1.5-7B  normalised G_mu = 3.250   (outputs/metrics/gap_llava15.json)
#   your baseline normalised G_mu = 2.727
#   your closed   normalised G_mu = 0.448
# i.e. a production-scale model, trained on >10x more data, sits FURTHER open
# than your own 450-step baseline. Two more such points turn that from an
# anecdote into: "the location gap is large and persistent in every released
# system measured."  This is generality of the DIAGNOSIS, not of the
# intervention -- the paper must say so explicitly.
#
# MODEL CHOICE IS CONSTRAINED BY CODE, NOT BY TASTE.
# scripts/23_extract_llava15_geometry.py hard-codes LlavaForConditionalGeneration
# (line 95). So:
#   SAFE, no code change : llava-1.5-13b-hf (scale), bakLlava-v1-hf (Mistral LLM)
#   NEEDS a class map    : llava-v1.6-* (LlavaNextForConditionalGeneration),
#                          InstructBLIP, BLIP-2, Qwen-VL
# Do NOT add llava-v1.6 here without first adding the class map, or it fails at
# from_pretrained after paying the download.
#
# Budget: ~1 h per model on 1300 pairs, measured from llava_geo_89241
# (17:05:54 -> 18:06:34). Plus download on first use. 10 h requested because
# llava15geo_88678 burned 4 attempts on model-reachability before succeeding.
#
#   sbatch scripts/submit_t1_vlmgeo.sbatch
#   MODELS="llava-hf/bakLlava-v1-hf" sbatch scripts/submit_t1_vlmgeo.sbatch
# ---------------------------------------------------------------------------

set -Eeuo pipefail
echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || true)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/metrics outputs/embeddings_1300

MODELS="${MODELS:-llava-hf/bakLlava-v1-hf llava-hf/llava-1.5-13b-hf}"
MANIFEST="data/dd256/gap_manifest_1300.json"

test -f "$MANIFEST" || { echo "[vlmgeo] FATAL: $MANIFEST missing" >&2; exit 1; }

python -c "
import sys, torch
if not torch.cuda.is_available(): print('[preflight] no CUDA'); sys.exit(1)
x = torch.randn(2048,2048,device='cuda'); _=(x@x).sum().item(); torch.cuda.synchronize()
print('[preflight] GPU op OK on', torch.cuda.get_device_name(0))
" || { echo '[vlmgeo] GPU preflight FAILED' >&2; exit 1; }

for m in $MODELS; do
  # condition tag from the model id: llava-hf/bakLlava-v1-hf -> bakLlava-v1-hf
  tag="$(echo "$m" | sed 's#.*/##')"
  out="outputs/metrics/gap_${tag}.json"
  if [ -f "$out" ]; then echo "[skip] $tag ($out exists)"; continue; fi

  echo "==================== GEOMETRY  $tag ===================="

  # preflight THIS model before paying for it
  python -c "
import sys
from transformers import AutoConfig, LlavaForConditionalGeneration
try:
    c = AutoConfig.from_pretrained('$m')
except Exception as e:
    print('[preflight] $tag not reachable/cached:', e); sys.exit(3)
at = getattr(c, 'architectures', None) or []
print('[preflight] $tag architectures=', at)
if at and not any('LlavaForConditionalGeneration' == a for a in at):
    print('[preflight] $tag needs a DIFFERENT model class than the script hard-codes.')
    print('[preflight] skipping rather than failing mid-run.'); sys.exit(4)
print('[preflight] $tag OK for LlavaForConditionalGeneration')
" || { echo "[vlmgeo] $tag preflight failed -> skipped" >&2; continue; }

  attempt=1
  until [ "$attempt" -gt 3 ]; do
    echo "[vlmgeo] $tag attempt ${attempt}/3"
    if python scripts/23_extract_llava15_geometry.py \
          --data-config configs/data_1300.yaml \
          --model-id "$m" --condition "$tag"; then
      echo "[vlmgeo] $tag done -> $out"; break
    fi
    echo "[vlmgeo] $tag attempt ${attempt} failed; sleep 60"; sleep 60
    attempt=$((attempt+1))
  done
done

echo "[vlmgeo] complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python - << 'PY'
import json, glob, os
print('[vlmgeo] normalised G_mu across released models (yours: baseline 2.727, closed 0.448)')
for f in sorted(glob.glob('outputs/metrics/gap_*.json')):
    n = os.path.basename(f)[4:-5]
    try: d = json.load(open(f))
    except Exception: continue
    sm = d.get('spec_metrics') or {}
    gm, tr = sm.get('G_mu'), sm.get('trace_image')
    if gm is None or not tr: continue
    print('   %-22s %.3f   (G_mu=%.2f, trace=%.2f)' % (n, gm / tr**0.5, gm, tr))
PY
