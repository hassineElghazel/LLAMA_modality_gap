# Pick an interpreter that actually has the repo's deps. Sourced by the launchers.
#   PY=/path/to/python ./demo/launch_*.sh    overrides the search
#   CONDA_ENV=name                            conda env to try (default llama_gap)

_demo_has_deps() {                       # $1 = python executable
  [[ -x "$1" ]] || return 1
  "$1" - <<'EOF' >/dev/null 2>&1
import importlib.util as u, sys
sys.exit(0 if all(u.find_spec(m) for m in ("torch", "transformers", "gradio")) else 1)
EOF
}

if [[ -z "${PY:-}" ]]; then
  _demo_cands=()
  [[ -n "${CONDA_PREFIX:-}" ]] && _demo_cands+=("${CONDA_PREFIX}/bin/python")
  if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-llama_gap}" 2>/dev/null \
      && _demo_cands+=("${CONDA_PREFIX}/bin/python")
  fi
  _demo_cands+=(".venv/bin/python" "$(command -v python3 || true)")
  for _c in "${_demo_cands[@]}"; do
    if _demo_has_deps "$_c"; then PY="$_c"; break; fi
  done
fi

if [[ -z "${PY:-}" ]] || ! _demo_has_deps "$PY"; then
  echo "[demo] no interpreter with torch + transformers + gradio." >&2
  echo "[demo] tried: ${_demo_cands[*]:-$PY}" >&2
  echo "[demo] fix: activate your env, or PY=/path/to/python ./demo/launch_*.sh" >&2
  exit 3
fi
echo "[demo] interpreter: $PY"
