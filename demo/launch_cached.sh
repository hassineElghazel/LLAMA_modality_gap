#!/usr/bin/env bash
# No-GPU fallback. Works on a laptop or a login node.
set -euo pipefail
cd "$(dirname "$0")/.."
source demo/_env.sh
export GRADIO_ANALYTICS_ENABLED=False
exec "$PY" demo/app.py --cached --port "${PORT:-7860}" --host "${HOST:-127.0.0.1}"
