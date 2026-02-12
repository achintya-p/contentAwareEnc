#!/usr/bin/env python
"""run_pipeline.py – Orchestrate the full content-aware encoding pipeline.

Runs in order:
    1) encode      – produce encoded variants
    2) features    – extract content features from input video
    3) metrics     – compute bitrate / quality metrics per variant
    4) detector    – evaluate downstream CV impact per variant
    5) merge       – build dataset.jsonl from all artifacts
    6) train       – train (or derive) encoding policy
    7) evaluate    – compare policy vs fixed-CRF baselines
    8) summarise   – write reports/summary.md + manifest.json

Usage:
    python -m content_aware_encoder.scripts.run_pipeline \\
        --config configs/default.yaml --run_name demo --dry_run
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path
from typing import Any, Dict, List

# ── Local imports (all relative to package root) ──────────────────────────
from content_aware_encoder.common.config import load_config, resolve_run_dir
from content_aware_encoder.common.io import (
    append_jsonl,
    read_json,
    read_jsonl,
    write_json,
    write_manifest,
)
from content_aware_encoder.common.logging import get_logger, setup_logging
from content_aware_encoder.common.schemas import DatasetRow
from content_aware_encoder.common.utils import add_common_args, cli_overrides

from content_aware_encoder.encode import encode_variants, save_encode_manifest
from content_aware_encoder.features import extract_features, save_features
from content_aware_encoder.metrics import compute_metrics, save_metrics
from content_aware_encoder.run_detector import run_detector, save_detector
from content_aware_encoder.train_policy import train_policy, save_policy, write_training_report
from content_aware_encoder.evaluate_policy import evaluate, write_eval_report

log = get_logger("pipeline")


# ── Merge step ────────────────────────────────────────────────────────────

def merge_dataset(
    features_path: Path,
    metrics_path: Path,
    detector_path: Path,
    dataset_path: Path,
    input_video: str,
) -> int:
    """Merge artifacts into ``dataset.jsonl``.  Returns number of rows written."""
    features = read_json(features_path)
    metrics_list: List[Dict[str, Any]] = read_json(metrics_path)
    detector_list: List[Dict[str, Any]] = read_json(detector_path)

    # Index detector by CRF for quick lookup
    det_by_crf = {d["crf"]: d for d in detector_list}

    count = 0
    for m in metrics_list:
        crf = m["crf"]
        d = det_by_crf.get(crf, {})
        row = DatasetRow(
            input_video=input_video,
            codec=m.get("codec", ""),
            crf=crf,
            motion_score=features.get("motion_score", 0.0),
            edge_density=features.get("edge_density", 0.0),
            brightness_mean=features.get("brightness_mean", 0.0),
            brightness_std=features.get("brightness_std", 0.0),
            bitrate_kbps=m.get("bitrate_kbps", 0.0),
            perceptual_proxy_psnr=m.get("perceptual_proxy_psnr", 0.0),
            detections_per_frame=d.get("detections_per_frame", 0.0),
            avg_confidence=d.get("avg_confidence", 0.0),
            throughput_fps=d.get("throughput_fps", 0.0),
        )
        append_jsonl(dataset_path, row.to_dict())
        count += 1

    log.info("Merged %d rows → %s", count, dataset_path)
    return count


# ── Summary report ────────────────────────────────────────────────────────

def write_summary(
    run_dir: Path,
    cfg: Dict[str, Any],
    features_path: Path,
    metrics_path: Path,
    detector_path: Path,
) -> Path:
    """Generate ``reports/summary.md``."""
    features = read_json(features_path)
    metrics_list = read_json(metrics_path)
    detector_list = read_json(detector_path)

    lines = [
        "# Pipeline Run Summary",
        "",
        f"**Run name:** {cfg.get('run_name', '?')}",
        f"**Input video:** {cfg.get('input_video', '?')}",
        f"**Codec:** {cfg.get('codec', '?')}",
        f"**CRF list:** {cfg.get('crf_list', [])}",
        f"**Dry run:** {cfg.get('dry_run', False)}",
        f"**Timestamp:** {datetime.datetime.utcnow().isoformat()}Z",
        "",
        "---",
        "",
        "## Content Features",
        "",
        f"| Feature | Value |",
        f"|---------|-------|",
        f"| motion_score | {features.get('motion_score', '?')} |",
        f"| edge_density | {features.get('edge_density', '?')} |",
        f"| brightness_mean | {features.get('brightness_mean', '?')} |",
        f"| brightness_std | {features.get('brightness_std', '?')} |",
        f"| frames_sampled | {features.get('frame_count_sampled', '?')} |",
        "",
        "## Encoding Variants",
        "",
        "| CRF | Size (bytes) | Bitrate (kbps) | PSNR (dB) | Det Conf |",
        "|-----|-------------|---------------|-----------|----------|",
    ]

    det_by_crf = {d["crf"]: d for d in detector_list}
    for m in metrics_list:
        crf = m["crf"]
        d = det_by_crf.get(crf, {})
        # find size from encode manifest
        enc_manifest_path = run_dir / "encodes" / "encode_manifest.json"
        size = "?"
        if enc_manifest_path.exists():
            enc_list = read_json(enc_manifest_path)
            for e in enc_list:
                if e["crf"] == crf:
                    size = e.get("size_bytes", "?")
                    break
        lines.append(
            f"| {crf} | {size} | {m.get('bitrate_kbps', '?'):.1f} "
            f"| {m.get('perceptual_proxy_psnr', '?'):.2f} "
            f"| {d.get('avg_confidence', '?')} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by `content_aware_encoder.scripts.run_pipeline`*")

    dest = run_dir / "reports" / "summary.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")
    log.info("Summary report → %s", dest)
    return dest


# ── Main orchestrator ─────────────────────────────────────────────────────

def run_pipeline(cfg: Dict[str, Any], run_dir: Path) -> None:
    """Execute the full pipeline end-to-end."""
    dry_run = cfg.get("dry_run", False)
    seed = cfg.get("random_seed", 42)

    encodes_dir = run_dir / "encodes"
    artifacts_dir = run_dir / "artifacts"
    reports_dir = run_dir / "reports"

    output_files: List[str] = []

    # 1) Encode
    log.info("═══ Step 1/7: Encode ═══")
    encode_results = encode_variants(
        input_video=cfg["input_video"],
        codec=cfg.get("codec", "libx264"),
        crf_list=cfg.get("crf_list", [18, 23, 28, 33]),
        encodes_dir=encodes_dir,
        dry_run=dry_run,
        seed=seed,
    )
    enc_manifest = save_encode_manifest(encode_results, encodes_dir)
    output_files.append(str(enc_manifest))

    # 2) Features
    log.info("═══ Step 2/7: Features ═══")
    feat_result = extract_features(
        video_path=cfg["input_video"],
        sample_rate=cfg.get("frame_sample_rate", 1),
        dry_run=dry_run,
        seed=seed,
    )
    feat_path = save_features(feat_result, artifacts_dir)
    output_files.append(str(feat_path))

    # 3) Metrics
    log.info("═══ Step 3/7: Metrics ═══")
    metrics_results = compute_metrics(
        encode_results,
        reference_path=cfg.get("input_video"),
        dry_run=dry_run,
        seed=seed,
    )
    metrics_path = save_metrics(metrics_results, artifacts_dir)
    output_files.append(str(metrics_path))

    # 4) Detector
    log.info("═══ Step 4/7: Detector ═══")
    if cfg.get("detector_enabled", True):
        det_results = run_detector(
            encode_results,
            backend=cfg.get("detector_backend", "stub"),
            dry_run=dry_run,
            seed=seed,
        )
        det_path = save_detector(det_results, artifacts_dir)
        output_files.append(str(det_path))
    else:
        log.info("Detector disabled – skipping.")
        # Write empty detector artifact so merge doesn't fail
        det_path = artifacts_dir / "detector.json"
        write_json(det_path, [])

    # 5) Merge
    log.info("═══ Step 5/7: Merge dataset ═══")
    dataset_path = run_dir / "dataset.jsonl"
    merge_dataset(feat_path, metrics_path, det_path, dataset_path, cfg["input_video"])
    output_files.append(str(dataset_path))

    # 6) Train policy
    log.info("═══ Step 6/7: Train policy ═══")
    model = train_policy(
        dataset_path=dataset_path,
        model_type=cfg.get("policy_model_type", "heuristic"),
        seed=seed,
    )
    policy_path = save_policy(model, run_dir / "models")
    write_training_report(model, reports_dir)
    output_files.append(str(policy_path))

    # 7) Evaluate
    log.info("═══ Step 7/7: Evaluate ═══")
    rows = read_jsonl(dataset_path)
    eval_report = evaluate(rows, model, crf_list=cfg.get("crf_list", [18, 23, 28, 33]))
    write_json(artifacts_dir / "eval_results.json", eval_report)
    eval_md = write_eval_report(eval_report, reports_dir)
    output_files.append(str(eval_md))

    # Summary + Manifest
    summary_path = write_summary(run_dir, cfg, feat_path, metrics_path, det_path)
    output_files.append(str(summary_path))
    manifest = write_manifest(run_dir, cfg, output_files)
    output_files.append(str(manifest))

    log.info("═══ Pipeline complete ═══")
    log.info("Run directory: %s", run_dir.resolve())


# ── CLI ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full content-aware encoding pipeline.",
    )
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

    setup_logging(log_file=run_dir / "logs" / "run.log", verbose=cfg.get("verbose", False))
    log.info("Config: %s", cfg)

    run_pipeline(cfg, run_dir)

    # Final banner
    print(f"\n✅  Pipeline complete.  Run directory:\n    {run_dir.resolve()}\n")


if __name__ == "__main__":
    main()
