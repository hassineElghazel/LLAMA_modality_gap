#!/usr/bin/env bash
# One-line launch on the GPU node. Run INSIDE an interactive allocation:
#   srun --partition=RTX --gpus=2080ti:1 --pty bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python demo/app.py --live --port "${PORT:-7860}" --host 127.0.0.1 \
     --timeout "${TIMEOUT:-45}"
