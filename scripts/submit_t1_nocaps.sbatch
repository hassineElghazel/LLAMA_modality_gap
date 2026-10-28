#!/bin/bash
#SBATCH --job-name=t1-nocaps
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/t1_nocaps_%j.out
#SBATCH --error=outputs/logs/t1_nocaps_%j.err

# ---------------------------------------------------------------------------
# TIER 1 / job 4.  COMPLETE THE OUT-OF-DISTRIBUTION EVALUATION.
#
# WHY.  NoCaps (Open Images) is disjoint from COCO. You already hold 2 of 8
# conditions and have never reported them: all-pinned 0.5127, Cloc 0.5795, i.e.
# +0.0668 OFF distribution against +0.0526 in it. The effect is LARGER out of
# domain. Completing the other conditions lets the full lever/mirror/null
# ordering be shown to replicate off-distribution, which is a far stronger
# statement than one delta, and it closes "they only tested one dataset".
#
# Budget: 6 conditions x 300 images at 256-token decode. Measured brief rate is
# ~2.96 s/image at 64 tokens, so ~11.8 s/image at 256 -> ~1 h per condition,
# ~6 h total. 12 h requested covers a slow node and the CLIPScore passes.
#
# PREREQ: data/nocaps/ood_eval_300.json must exist (scripts/prep_ood_eval.py).
# The existing Cloc/C3pinr results used the SAME file, so do not regenerate it
# or the new numbers stop being comparable to the two you already have.
#
#   sbatch scripts/submit_t1_nocaps.sbatch
# ---------------------------------------------------------------------------

set -Eeuo pipefail
echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || true)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs outputs/metrics outputs/predictions

CFG="${CFG:-configs/ood_eval.yaml}"
CLIP_MODEL="${CLIP_MODEL:-/nfs/home/elghazel/clip_b32}"
# the 6 NOT yet run; Cloc and C3pinr already exist and are NOT repeated
CONDS="${CONDS:-Cscale1500 Crank15 Corient Clocorient Cloc_long Cloc_80k}"

test -f data/nocaps/ood_eval_300.json || {
  echo "[fatal] data/nocaps/ood_eval_300.json missing -- run scripts/prep_ood_eval.py first" >&2
  echo "[fatal] do NOT rebuild it with a different seed/n: the 2 existing conditions used this exact file" >&2
  exit 1; }

python -c "
import sys, torch
if not torch.cuda.is_available(): print('[preflight] no CUDA'); sys.exit(1)
x = torch.randn(2048,2048,device='cuda'); _=(x@x).sum().item(); torch.cuda.synchronize()
print('[preflight] GPU op OK on', torch.cuda.get_device_name(0))
" || { echo '[nocaps] GPU preflight FAILED' >&2; exit 1; }

for c in $CONDS; do
  ck="outputs/checkpoints/stage2_vlm_${c}.pt"
  if [ ! -f "$ck" ]; then echo "[skip] $c (no checkpoint $ck)"; continue; fi
  tag="${c}_ood"
  if [ -f "outputs/metrics/clipscore_${tag}.json" ]; then
    echo "[skip] $c (clipscore_${tag}.json already exists)"; continue; fi
  echo "==================== DESCRIBE (OOD)  $tag ===================="
  python scripts/08_run_captioning.py --config "$CFG" --condition "$tag" --vlm-checkpoint "$ck" \
    || { echo "[nocaps] $tag captioning FAILED (continuing)" >&2; continue; }
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/15_clipscore.py \
      --config "$CFG" --condition "$tag" --clip-model "$CLIP_MODEL" \
    || echo "[nocaps] $tag clipscore FAILED (rerun 15_clipscore.py --config $CFG --condition $tag)" >&2
done

echo "[nocaps] complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python - << 'PY'
import json, glob, os
rows=[]
for f in sorted(glob.glob('outputs/metrics/clipscore_*_ood.json')):
    try: d=json.load(open(f))
    except Exception: continue
    rows.append((os.path.basename(f).replace('clipscore_','').replace('_ood.json',''),
                 d.get('CLIPScore'), d.get('n_evaluated')))
print('[nocaps] OOD CLIPScore summary')
for n,s,k in rows:
    print('   %-14s %.4f  (n=%s)'%(n,s,k) if isinstance(s,float) else '   %-14s %s'%(n,s))
PY
