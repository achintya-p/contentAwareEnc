"""Structured logging helpers – console + file output."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_CONFIGURED = False


def setup_logging(
    log_file: Optional[Path] = None,
    verbose: bool = False,
) -> logging.Logger:
    """Configure the root ``cae`` logger.

    - Always logs INFO+ to stderr.
    - If *verbose*, logs DEBUG+ to stderr.
    - If *log_file* is provided, logs DEBUG+ to file.
    """
    global _CONFIGURED  # noqa: PLW0603
    logger = logging.getLogger("cae")

    if _CONFIGURED:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _CONFIGURED = True
    return logger


def get_logger(name: str = "cae") -> logging.Logger:
    """Return a child logger under ``cae``."""
    return logging.getLogger(f"cae.{name}")
