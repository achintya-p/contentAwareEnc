#!/usr/bin/env python
"""features.py – Extract cheap content features from an input video.

Usage:
    python -m content_aware_encoder.features --config configs/default.yaml --run_dir results/demo --dry_run
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from content_aware_encoder.common.config import load_config, resolve_run_dir
from content_aware_encoder.common.io import write_json
from content_aware_encoder.common.logging import get_logger, setup_logging
from content_aware_encoder.common.schemas import FeatureResult
from content_aware_encoder.common.utils import add_common_args, cli_overrides

log = get_logger("features")


# ── OpenCV feature extraction (best-effort) ──────────────────────────────

def _try_opencv_features(
    video_path: str,
    sample_rate: int = 1,
) -> Optional[FeatureResult]:
    """Attempt real feature extraction via OpenCV.  Returns None on failure."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        log.debug("OpenCV / numpy not installed – skipping real features.")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.debug("Cannot open video %s with OpenCV.", video_path)
        return None

    prev_gray = None
    diffs: list[float] = []
    edges_list: list[float] = []
    brightness_vals: list[float] = []
    frame_idx = 0
    sampled = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % sample_rate != 0:
            continue
        sampled += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        brightness_vals.append(float(gray.mean()))

        # Edge density (Sobel)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges_list.append(float(np.sqrt(sx**2 + sy**2).mean()))

        # Motion score (mean abs frame diff)
        if prev_gray is not None:
            diffs.append(float(np.abs(gray - prev_gray).mean()))
        prev_gray = gray

    cap.release()

    if sampled == 0:
        return None

    import statistics
    return FeatureResult(
        input_video=video_path,
        motion_score=round(statistics.mean(diffs) if diffs else 0.0, 4),
        edge_density=round(statistics.mean(edges_list), 4),
        brightness_mean=round(statistics.mean(brightness_vals), 4),
        brightness_std=round(statistics.pstdev(brightness_vals), 4) if len(brightness_vals) > 1 else 0.0,
        frame_count_sampled=sampled,
    )


# ── Deterministic stub features ──────────────────────────────────────────

def _stub_features(video_path: str, seed: int = 42) -> FeatureResult:
    """Generate realistic but deterministic placeholder features."""
    rng = random.Random(seed)
    return FeatureResult(
        input_video=video_path,
        motion_score=round(rng.uniform(2.0, 25.0), 4),
        edge_density=round(rng.uniform(5.0, 40.0), 4),
        brightness_mean=round(rng.uniform(80.0, 180.0), 4),
        brightness_std=round(rng.uniform(10.0, 50.0), 4),
        frame_count_sampled=rng.randint(30, 300),
    )


# ── Public API ────────────────────────────────────────────────────────────

def extract_features(
    video_path: str,
    sample_rate: int = 1,
    dry_run: bool = False,
    seed: int = 42,
) -> FeatureResult:
    """Extract content features from *video_path*."""
    if not dry_run:
        result = _try_opencv_features(video_path, sample_rate=sample_rate)
        if result is not None:
            log.info("Extracted real features from %s (%d frames).", video_path, result.frame_count_sampled)
            return result
        log.info("Falling back to stub features for %s.", video_path)

    result = _stub_features(video_path, seed=seed)
    log.info("[stub] Generated features for %s", video_path)
    return result


def save_features(result: FeatureResult, artifacts_dir: Path) -> Path:
    dest = artifacts_dir / "features.json"
    write_json(dest, result.to_dict())
    log.info("Features → %s", dest)
    return dest


# ── CLI ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract content features from input video.")
    add_common_args(parser)
    parser.add_argument("--input_video", type=str, help="Override input video path.")
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = cli_overrides(args)
    if args.input_video:
        overrides["input_video"] = args.input_video

    cfg = load_config(args.config, overrides)
    run_dir = Path(args.run_dir) if args.run_dir else resolve_run_dir(cfg)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=run_dir / "logs" / "run.log", verbose=cfg.get("verbose", False))

    result = extract_features(
        video_path=cfg["input_video"],
        sample_rate=cfg.get("frame_sample_rate", 1),
        dry_run=cfg.get("dry_run", False),
        seed=cfg.get("random_seed", 42),
    )
    save_features(result, artifacts_dir)


if __name__ == "__main__":
    main()
