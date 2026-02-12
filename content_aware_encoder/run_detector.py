#!/usr/bin/env python
"""run_detector.py – Evaluate downstream CV impact of each variant.

Backends:
    stub  – always available; produces plausible values correlated with CRF.
    yolo  – requires ``ultralytics``; runs YOLOv8 inference.

Usage:
    python -m content_aware_encoder.run_detector --config configs/default.yaml --run_dir results/demo --dry_run
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List

from content_aware_encoder.common.config import load_config, resolve_run_dir
from content_aware_encoder.common.io import read_json, write_json
from content_aware_encoder.common.logging import get_logger, setup_logging
from content_aware_encoder.common.schemas import DetectorResult, EncodeResult
from content_aware_encoder.common.utils import add_common_args, cli_overrides

log = get_logger("detector")


# ── Stub backend ──────────────────────────────────────────────────────────

def _stub_detect(er: EncodeResult, seed: int) -> DetectorResult:
    """Deterministic stub: higher CRF → slightly worse detection quality."""
    rng = random.Random(seed + er.crf)
    # Base detections per frame ~5; degrades slightly with CRF
    det_per_frame = round(5.0 - (er.crf - 18) * 0.15 + rng.uniform(-0.3, 0.3), 2)
    avg_conf = round(0.85 - (er.crf - 18) * 0.012 + rng.uniform(-0.02, 0.02), 4)
    fps = round(30.0 + rng.uniform(-2, 2), 1)
    return DetectorResult(
        backend="stub",
        codec=er.codec,
        crf=er.crf,
        detections_per_frame=max(det_per_frame, 0.5),
        avg_confidence=max(min(avg_conf, 1.0), 0.1),
        throughput_fps=fps,
    )


# ── YOLO backend (optional) ──────────────────────────────────────────────

def _yolo_detect(er: EncodeResult) -> DetectorResult | None:
    """Run YOLOv8 on encoded variant.  Returns None if ultralytics missing."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        log.debug("ultralytics not installed – cannot use yolo backend.")
        return None

    model = YOLO("yolov8n.pt")
    results = model(er.path, stream=True, verbose=False)

    det_counts: list[int] = []
    confs: list[float] = []
    for r in results:
        n = len(r.boxes)
        det_counts.append(n)
        if n > 0:
            confs.extend(r.boxes.conf.tolist())

    import statistics
    return DetectorResult(
        backend="yolo",
        codec=er.codec,
        crf=er.crf,
        detections_per_frame=round(statistics.mean(det_counts), 2) if det_counts else 0.0,
        avg_confidence=round(statistics.mean(confs), 4) if confs else 0.0,
        throughput_fps=0.0,  # not measured here
    )


# ── Public API ────────────────────────────────────────────────────────────

def run_detector(
    encode_results: List[EncodeResult],
    backend: str = "stub",
    dry_run: bool = False,
    seed: int = 42,
) -> List[DetectorResult]:
    """Run detector evaluation for every encode variant."""
    results: List[DetectorResult] = []

    for er in encode_results:
        if backend == "yolo" and not dry_run:
            det = _yolo_detect(er)
            if det is not None:
                results.append(det)
                log.info("YOLO  crf=%d  det/frame=%.2f  conf=%.4f", er.crf, det.detections_per_frame, det.avg_confidence)
                continue
            log.warning("YOLO unavailable – falling back to stub for crf=%d.", er.crf)

        det = _stub_detect(er, seed)
        results.append(det)
        log.info("[stub]  crf=%d  det/frame=%.2f  conf=%.4f", er.crf, det.detections_per_frame, det.avg_confidence)

    return results


def save_detector(results: List[DetectorResult], artifacts_dir: Path) -> Path:
    dest = artifacts_dir / "detector.json"
    write_json(dest, [d.to_dict() for d in results])
    log.info("Detector → %s", dest)
    return dest


# ── CLI ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate downstream CV impact per variant.")
    add_common_args(parser)
    parser.add_argument("--backend", type=str, choices=["stub", "yolo"], help="Detector backend.")
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = cli_overrides(args)
    if args.backend:
        overrides["detector_backend"] = args.backend

    cfg = load_config(args.config, overrides)
    run_dir = Path(args.run_dir) if args.run_dir else resolve_run_dir(cfg)
    artifacts_dir = run_dir / "artifacts"
    encodes_dir = run_dir / "encodes"

    setup_logging(log_file=run_dir / "logs" / "run.log", verbose=cfg.get("verbose", False))

    manifest_path = encodes_dir / "encode_manifest.json"
    if not manifest_path.exists():
        log.error("Encode manifest not found at %s – run encode first.", manifest_path)
        raise SystemExit(1)

    raw = read_json(manifest_path)
    encode_results = [EncodeResult(**r) for r in raw]

    det_results = run_detector(
        encode_results,
        backend=cfg.get("detector_backend", "stub"),
        dry_run=cfg.get("dry_run", False),
        seed=cfg.get("random_seed", 42),
    )
    save_detector(det_results, artifacts_dir)


if __name__ == "__main__":
    main()
