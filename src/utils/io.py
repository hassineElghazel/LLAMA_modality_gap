"""I/O helpers — config loading, JSON/Tensor save+load, run-metadata snapshot.

Per §9 of the plan, every script that produces outputs must save alongside them
the full config snapshot, git commit hash, pip freeze output, seeds, and GPU
info — that's what ``snapshot_run_metadata`` does.
"""
from __future__ import annotations

import hashlib
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_run_metadata(
    config: dict,
    out_dir: str | Path,
    *,
    config_files: dict[str, str | Path] | None = None,
    extra_files: dict[str, str | Path] | None = None,
) -> Path:
    """Write a metadata.json next to a run's outputs capturing reproducibility info.

    Optional arguments:
      config_files: mapping name -> YAML path. Each file's parsed content AND
                    its SHA256 are embedded so future runs can be byte-compared.
      extra_files:  mapping name -> path. Only the SHA256 is recorded (useful
                    for large artifacts like the diagnostic manifest).
    """
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
    if config_files:
        meta["config_files"] = {}
        for name, p in config_files.items():
            p = Path(p)
            entry: dict[str, Any] = {"path": str(p)}
            if p.exists():
                entry["sha256"] = _sha256_file(p)
                try:
                    entry["content"] = load_yaml(p)
                except Exception as e:  # noqa: BLE001
                    entry["content_error"] = f"<unparseable: {e}>"
            else:
                entry["missing"] = True
            meta["config_files"][name] = entry
    if extra_files:
        meta["extra_files"] = {}
        for name, p in extra_files.items():
            p = Path(p)
            entry = {"path": str(p)}
            if p.exists():
                entry["sha256"] = _sha256_file(p)
                entry["size_bytes"] = p.stat().st_size
            else:
                entry["missing"] = True
            meta["extra_files"][name] = entry
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
