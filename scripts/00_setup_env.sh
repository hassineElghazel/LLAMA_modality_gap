#!/usr/bin/env bash
# Set up Python environment and install dependencies.
# Per plan §4: Python 3.10, CUDA 12.1, uv preferred over pip.

set -euo pipefail

PYTHON_VERSION="3.10"

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv not found, falling back to pip"
  USE_UV=0
else
  USE_UV=1
fi

if [ ! -d ".venv" ]; then
  if [ "$USE_UV" -eq 1 ]; then
    uv venv --python="$PYTHON_VERSION" .venv
  else
    "python$PYTHON_VERSION" -m venv .venv
  fi
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if [ "$USE_UV" -eq 1 ]; then
  uv pip install -r requirements.txt
  uv pip install -e ".[dev]"
else
  pip install --upgrade pip
  pip install -r requirements.txt
  pip install -e ".[dev]"
fi

# Java check for pycocoevalcap (METEOR + SPICE).
if ! command -v java >/dev/null 2>&1; then
  echo "[setup] WARNING: 'java' not on PATH. METEOR and SPICE will fail."
  echo "[setup] Install JDK 8+ before running scripts/09_score_captions.py"
else
  java -version 2>&1 | head -n 1
fi

echo "[setup] OK. Activate with: source .venv/bin/activate"
