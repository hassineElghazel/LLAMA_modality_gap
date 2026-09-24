#!/usr/bin/env bash
# One-line launch of the no-GPU fallback (laptop or node).
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python demo/app.py --cached --port "${PORT:-7860}" --host 127.0.0.1
