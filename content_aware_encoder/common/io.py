"""I/O helpers: JSON read/write, JSONL append, manifest creation."""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=indent, default=str)


def read_json(path: Path) -> Any:
    """Read and return parsed JSON from *path*."""
    with open(path, "r") as fh:
        return json.load(fh)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append a single JSON record as one line to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read all JSON-line records from *path*."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _git_commit_hash() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def write_manifest(run_dir: Path, cfg: Dict[str, Any], output_files: List[str]) -> Path:
    """Write ``manifest.json`` capturing config, env info, and output paths."""
    manifest = {
        "schema_version": "0.1.0",
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit_hash(),
        "python_version": sys.version,
        "platform": sys.platform,
        "config": cfg,
        "output_files": output_files,
    }
    dest = run_dir / "manifest.json"
    write_json(dest, manifest)
    return dest
