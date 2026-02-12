"""Small general-purpose utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard ``--config / --run_dir / --dry_run / --verbose`` flags."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Explicit run directory (overrides config-based resolution).",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name (results/<run_name>).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Generate placeholder outputs without heavy computation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )


def cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Build a dict of non-None CLI overrides suitable for config merging."""
    overrides: Dict[str, Any] = {}
    if args.dry_run:
        overrides["dry_run"] = True
    if args.verbose:
        overrides["verbose"] = True
    if args.run_name:
        overrides["run_name"] = args.run_name
    return overrides


def file_size_bytes(path: Path) -> int:
    """Return file size in bytes, or 0 if file doesn't exist."""
    try:
        return path.stat().st_size
    except OSError:
        return 0
