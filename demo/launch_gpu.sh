#!/usr/bin/env bash
# Live demo on the GPU node. Run INSIDE an allocation:
#   srun --partition=RTX --gpus=2080ti:1 --pty bash
set -euo pipefail
cd "$(dirname "$0")/.."
source demo/_env.sh
# Offline is OPT-IN, not the default: the CLIP ViT-B/32 safetensors may not be
# cached on the compute node, and transformers >= 4.56 cannot load the .bin on
# torch < 2.6. Set HF_HUB_OFFLINE=1 yourself once the cache is warm.
export GRADIO_ANALYTICS_ENABLED=False
exec "$PY" demo/app.py --live --port "${PORT:-7860}" --host "${HOST:-127.0.0.1}" \
     --timeout "${TIMEOUT:-45}"
