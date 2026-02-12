"""Lightweight dataclass schemas for pipeline artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "0.1.0"


# ── Encoding ──────────────────────────────────────────────────────────────

@dataclass
class EncodeSpec:
    """Specification for a single encode variant."""
    codec: str
    crf: int
    input_video: str
    output_path: str


@dataclass
class EncodeResult:
    """Result of a single encode variant."""
    schema_version: str = SCHEMA_VERSION
    codec: str = ""
    crf: int = 0
    path: str = ""
    duration_sec: float = 0.0
    size_bytes: int = 0
    estimated_bitrate_kbps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Features ──────────────────────────────────────────────────────────────

@dataclass
class FeatureResult:
    """Extracted content features for one video."""
    schema_version: str = SCHEMA_VERSION
    input_video: str = ""
    motion_score: float = 0.0
    edge_density: float = 0.0
    brightness_mean: float = 0.0
    brightness_std: float = 0.0
    frame_count_sampled: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Quality / bitrate metrics ────────────────────────────────────────────

@dataclass
class MetricsResult:
    """Quality & bitrate metrics for one encode variant."""
    schema_version: str = SCHEMA_VERSION
    codec: str = ""
    crf: int = 0
    bitrate_kbps: float = 0.0
    perceptual_proxy_psnr: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Detector ──────────────────────────────────────────────────────────────

@dataclass
class DetectorResult:
    """Downstream CV impact metrics for one variant."""
    schema_version: str = SCHEMA_VERSION
    backend: str = "stub"
    codec: str = ""
    crf: int = 0
    detections_per_frame: float = 0.0
    avg_confidence: float = 0.0
    throughput_fps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Merged dataset row ───────────────────────────────────────────────────

@dataclass
class DatasetRow:
    """One row in the merged training dataset."""
    schema_version: str = SCHEMA_VERSION
    input_video: str = ""
    codec: str = ""
    crf: int = 0
    # features
    motion_score: float = 0.0
    edge_density: float = 0.0
    brightness_mean: float = 0.0
    brightness_std: float = 0.0
    # metrics
    bitrate_kbps: float = 0.0
    perceptual_proxy_psnr: float = 0.0
    # detector
    detections_per_frame: float = 0.0
    avg_confidence: float = 0.0
    throughput_fps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
