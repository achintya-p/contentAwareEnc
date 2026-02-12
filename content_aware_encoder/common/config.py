"""YAML config loading with defaults and CLI override support."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# YAML loading – fall back to a tiny safe subset if PyYAML is absent
# ---------------------------------------------------------------------------

try:
    import yaml  # type: ignore

    def _load_yaml(path: Path) -> Dict[str, Any]:
        with open(path, "r") as fh:
            return yaml.safe_load(fh) or {}

except ImportError:
    import json as _json
    import re as _re

    def _load_yaml(path: Path) -> Dict[str, Any]:  # type: ignore[misc]
        """Minimal YAML-subset loader (flat key: value, lists via JSON)."""
        data: Dict[str, Any] = {}
        with open(path, "r") as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if not line or line.startswith("---"):
                    continue
                m = _re.match(r"^(\w[\w.]*):\s*(.+)$", line)
                if m:
                    key, val = m.group(1), m.group(2).strip()
                    # Try JSON parse for lists / numbers / booleans
                    try:
                        data[key] = _json.loads(val)
                    except _json.JSONDecodeError:
                        data[key] = val
        return data


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, Any] = {
    "input_video": "sample.mp4",
    "run_name": "default_run",
    "codec": "libx264",
    "crf_list": [18, 23, 28, 33],
    "frame_sample_rate": 1,
    "metrics_enabled": True,
    "detector_enabled": True,
    "detector_backend": "stub",
    "policy_model_type": "heuristic",
    "random_seed": 42,
    "dry_run": False,
    "verbose": False,
}


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load config from YAML, merge with defaults & optional CLI overrides."""
    cfg = copy.deepcopy(DEFAULTS)
    if config_path and Path(config_path).exists():
        cfg.update(_load_yaml(Path(config_path)))
    if overrides:
        cfg.update(overrides)
    return cfg


def resolve_run_dir(cfg: Dict[str, Any], base: str = "results") -> Path:
    """Return ``results/<run_name>`` path, creating it and standard sub-dirs."""
    run_dir = Path(base) / cfg["run_name"]
    for sub in ("encodes", "artifacts", "reports", "logs", "models"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir
