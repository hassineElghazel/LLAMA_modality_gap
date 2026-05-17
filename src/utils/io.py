"""I/O helpers — config loading, JSON/Tensor save+load, run-metadata snapshot.

Per §9 of the plan, every script that produces outputs must save alongside them
the full config snapshot, git commit hash, pip freeze output, seeds, and GPU
info — that's what ``snapshot_run_metadata`` does.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj: Any, path: str | Path) -> None:
    if is_dataclass(obj):
        obj = asdict(obj)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def _safe_run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception as e:   # noqa: BLE001
        return f"<unavailable: {e}>"


def snapshot_run_metadata(config: dict, out_dir: str | Path) -> Path:
    """Write a metadata.json next to a run's outputs capturing reproducibility info."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "config": config,
        "git_commit": _safe_run(["git", "rev-parse", "HEAD"]),
        "git_status": _safe_run(["git", "status", "--porcelain"]),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pip_freeze": _safe_run([sys.executable, "-m", "pip", "freeze"]),
    }
    try:
        import torch
        meta["torch_version"] = torch.__version__
        meta["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            meta["gpu_name"] = torch.cuda.get_device_name(0)
            meta["cuda_version"] = torch.version.cuda
    except ImportError:
        meta["torch_version"] = "<not installed>"
    path = out_dir / "metadata.json"
    save_json(meta, path)
    return path
