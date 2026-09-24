#!/usr/bin/env bash
# Live demo on the GPU node. Run INSIDE an allocation:
#   srun --partition=RTX --gpus=2080ti:1 --pty bash
set -euo pipefail
cd "$(dirname "$0")/.."
source demo/_env.sh
export GRADIO_ANALYTICS_ENABLED=False HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} \
       TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
exec "$PY" demo/app.py --live --port "${PORT:-7860}" --host "${HOST:-127.0.0.1}" \
     --timeout "${TIMEOUT:-45}"
