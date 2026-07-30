#!/bin/bash
#SBATCH --job-name=docci-prep
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --output=outputs/logs/docci_prep_%j.out
#SBATCH --error=outputs/logs/docci_prep_%j.err

# Materialize DOCCI train (9,647) + test (5,000) to data/docci/ (images + manifests
# + references). Downloads DOCCI's official public files directly from Google Cloud
# Storage (storage.googleapis.com/docci) -- needs outbound network on this node.
# Re-runnable: cached downloads in data/docci/_raw and existing images are skipped.
set -Eeuo pipefail
echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llama_gap
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || true)

cd ~/LLAMA_modality_gap
mkdir -p outputs/logs data/docci

python scripts/24_prep_docci.py --splits train,test --out-root data/docci

echo "[docci] train rows: $(python -c "import json;print(len(json.load(open('data/docci/train_manifest.json'))))")"
echo "[docci] test rows:  $(python -c "import json;print(len(json.load(open('data/docci/test_manifest.json'))))")"
echo "[slurm] docci-prep complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
