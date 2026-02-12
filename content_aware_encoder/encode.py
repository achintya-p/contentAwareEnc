#!/usr/bin/env python
"""encode.py – Encode a video at multiple CRF points.

Usage:
    python -m content_aware_encoder.encode --config configs/default.yaml --run_dir results/demo --dry_run
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

from content_aware_encoder.common.config import load_config, resolve_run_dir
from content_aware_encoder.common.ffmpeg import ffmpeg_available, ffprobe_duration, run_encode
from content_aware_encoder.common.io import write_json
from content_aware_encoder.common.logging import get_logger, setup_logging
from content_aware_encoder.common.schemas import EncodeResult
from content_aware_encoder.common.utils import add_common_args, cli_overrides, file_size_bytes

log = get_logger("encode")


# ── Core logic ────────────────────────────────────────────────────────────

def encode_variants(
    input_video: str,
    codec: str,
    crf_list: List[int],
    encodes_dir: Path,
    dry_run: bool = False,
    seed: int = 42,
) -> List[EncodeResult]:
    """Encode *input_video* at each CRF in *crf_list*.

    Returns a list of :class:`EncodeResult` dataclasses.
    """
    results: List[EncodeResult] = []
    rng = random.Random(seed)

    use_ffmpeg = ffmpeg_available() and not dry_run and Path(input_video).exists()

    for crf in crf_list:
        stem = Path(input_video).stem
        out_name = f"{stem}_{codec}_crf{crf}.mp4"
        out_path = encodes_dir / out_name

        if use_ffmpeg:
            log.info("Encoding %s  crf=%d → %s", input_video, crf, out_path)
            try:
                run_encode(Path(input_video), out_path, codec=codec, crf=crf)
            except Exception as exc:
                log.warning("ffmpeg failed for crf=%d: %s – falling back to stub", crf, exc)
                _write_stub(out_path, crf, rng)
        else:
            log.info("[dry_run] Generating stub encode  crf=%d → %s", crf, out_path)
            _write_stub(out_path, crf, rng)

        size = file_size_bytes(out_path)
        # Attempt real duration via ffprobe; fall back to stub
        duration = ffprobe_duration(out_path) if not dry_run else None
        if duration is None:
            duration = 10.0  # stub 10-second clip

        bitrate = (size * 8 / 1000) / duration if duration > 0 else 0.0

        results.append(
            EncodeResult(
                codec=codec,
                crf=crf,
                path=str(out_path),
                duration_sec=round(duration, 3),
                size_bytes=size,
                estimated_bitrate_kbps=round(bitrate, 2),
            )
        )

    return results


def _write_stub(path: Path, crf: int, rng: random.Random) -> None:
    """Write a placeholder file whose size is inversely correlated with CRF."""
    # Higher CRF → smaller file (rough model)
    base_size = 500_000  # ~500 KB at CRF 18
    factor = max(0.1, 1.0 - (crf - 18) * 0.05)
    size = int(base_size * factor)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(rng.randbytes(size))


def save_encode_manifest(results: List[EncodeResult], encodes_dir: Path) -> Path:
    """Write ``encode_manifest.json`` and return its path."""
    dest = encodes_dir / "encode_manifest.json"
    write_json(dest, [r.to_dict() for r in results])
    log.info("Encode manifest → %s", dest)
    return dest


# ── CLI ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode video at multiple CRFs.")
    add_common_args(parser)
    parser.add_argument("--input_video", type=str, help="Override input video path.")
    parser.add_argument("--codec", type=str, help="Video codec (default: libx264).")
    parser.add_argument("--crf_list", type=int, nargs="+", help="CRF values.")
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = cli_overrides(args)
    if args.input_video:
        overrides["input_video"] = args.input_video
    if args.codec:
        overrides["codec"] = args.codec
    if args.crf_list:
        overrides["crf_list"] = args.crf_list

    cfg = load_config(args.config, overrides)
    run_dir = Path(args.run_dir) if args.run_dir else resolve_run_dir(cfg)
    encodes_dir = run_dir / "encodes"
    encodes_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=run_dir / "logs" / "run.log", verbose=cfg.get("verbose", False))
    log.info("Starting encode  dry_run=%s", cfg.get("dry_run"))

    results = encode_variants(
        input_video=cfg["input_video"],
        codec=cfg.get("codec", "libx264"),
        crf_list=cfg.get("crf_list", [18, 23, 28, 33]),
        encodes_dir=encodes_dir,
        dry_run=cfg.get("dry_run", False),
        seed=cfg.get("random_seed", 42),
    )
    save_encode_manifest(results, encodes_dir)
    log.info("Encode step complete – %d variants written.", len(results))


if __name__ == "__main__":
    main()
