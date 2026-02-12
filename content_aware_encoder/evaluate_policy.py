#!/usr/bin/env python
"""evaluate_policy.py – Compare fixed-CRF baselines vs learned/heuristic policy.

Usage:
    python -m content_aware_encoder.evaluate_policy --config configs/default.yaml --run_dir results/demo
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from content_aware_encoder.common.config import load_config, resolve_run_dir
from content_aware_encoder.common.io import read_json, read_jsonl, write_json
from content_aware_encoder.common.logging import get_logger, setup_logging
from content_aware_encoder.common.utils import add_common_args, cli_overrides
from content_aware_encoder.train_policy import predict_crf

log = get_logger("evaluate")


# ── Evaluation logic ──────────────────────────────────────────────────────

def evaluate(
    dataset_rows: List[Dict[str, Any]],
    model: Dict[str, Any],
    crf_list: List[int],
) -> Dict[str, Any]:
    """Compare fixed-CRF baselines against the policy.

    Returns a dict with per-baseline stats and policy stats.
    """
    report: Dict[str, Any] = {"baselines": {}, "policy": {}}

    # ── Fixed-CRF baselines ───────────────────────────────────────────
    for crf in crf_list:
        matching = [r for r in dataset_rows if r.get("crf") == crf]
        if not matching:
            continue
        avg_br = _mean([r.get("bitrate_kbps", 0) for r in matching])
        avg_psnr = _mean([r.get("perceptual_proxy_psnr", 0) for r in matching])
        avg_det = _mean([r.get("avg_confidence", 0) for r in matching])
        report["baselines"][f"crf_{crf}"] = {
            "crf": crf,
            "avg_bitrate_kbps": round(avg_br, 2),
            "avg_psnr_db": round(avg_psnr, 2),
            "avg_detector_confidence": round(avg_det, 4),
            "n": len(matching),
        }

    # ── Policy prediction ─────────────────────────────────────────────
    policy_rows: List[Dict[str, Any]] = []
    for row in dataset_rows:
        chosen_crf = predict_crf(model, row.get("motion_score", 10.0))
        # find the matching CRF row for same video (if available)
        match = [r for r in dataset_rows if r.get("crf") == chosen_crf and r.get("input_video") == row.get("input_video")]
        if match:
            policy_rows.append(match[0])

    if policy_rows:
        report["policy"] = {
            "model_type": model.get("type", "unknown"),
            "avg_bitrate_kbps": round(_mean([r.get("bitrate_kbps", 0) for r in policy_rows]), 2),
            "avg_psnr_db": round(_mean([r.get("perceptual_proxy_psnr", 0) for r in policy_rows]), 2),
            "avg_detector_confidence": round(_mean([r.get("avg_confidence", 0) for r in policy_rows]), 4),
            "n": len(policy_rows),
        }
    else:
        report["policy"] = {"model_type": model.get("type"), "note": "no matching rows for predicted CRFs"}

    return report


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


# ── Report writer ─────────────────────────────────────────────────────────

def write_eval_report(report: Dict[str, Any], reports_dir: Path) -> Path:
    dest = reports_dir / "eval_report.md"
    lines = [
        "# Evaluation Report – Fixed CRF vs Policy",
        "",
        "## Fixed-CRF Baselines",
        "",
        "| CRF | Avg Bitrate (kbps) | Avg PSNR (dB) | Avg Det. Conf. | N |",
        "|-----|--------------------|---------------|----------------|---|",
    ]
    for _key, b in sorted(report.get("baselines", {}).items()):
        lines.append(
            f"| {b['crf']} | {b['avg_bitrate_kbps']:.1f} | {b['avg_psnr_db']:.2f} | {b['avg_detector_confidence']:.4f} | {b['n']} |"
        )

    lines.append("")
    lines.append("## Policy")
    lines.append("")
    p = report.get("policy", {})
    if "avg_bitrate_kbps" in p:
        lines.append(f"- **Model type:** {p.get('model_type', '?')}")
        lines.append(f"- **Avg Bitrate:** {p['avg_bitrate_kbps']:.1f} kbps")
        lines.append(f"- **Avg PSNR:** {p['avg_psnr_db']:.2f} dB")
        lines.append(f"- **Avg Det. Confidence:** {p['avg_detector_confidence']:.4f}")
        lines.append(f"- **N:** {p['n']}")
    else:
        lines.append(f"- Note: {p.get('note', 'no data')}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")
    log.info("Eval report → %s", dest)
    return dest


# ── CLI ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate policy vs fixed-CRF baselines.")
    add_common_args(parser)
    parser.add_argument("--dataset", type=str, help="Path to dataset.jsonl.")
    parser.add_argument("--model", type=str, help="Path to policy.json.")
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = cli_overrides(args)

    cfg = load_config(args.config, overrides)
    run_dir = Path(args.run_dir) if args.run_dir else resolve_run_dir(cfg)

    setup_logging(log_file=run_dir / "logs" / "run.log", verbose=cfg.get("verbose", False))

    dataset_path = Path(args.dataset) if args.dataset else run_dir / "dataset.jsonl"
    model_path = Path(args.model) if args.model else run_dir / "models" / "policy.json"

    if not dataset_path.exists():
        log.error("Dataset not found at %s – run pipeline first.", dataset_path)
        raise SystemExit(1)
    if not model_path.exists():
        log.error("Policy model not found at %s – run train_policy first.", model_path)
        raise SystemExit(1)

    rows = read_jsonl(dataset_path)
    model = read_json(model_path)

    report = evaluate(rows, model, crf_list=cfg.get("crf_list", [18, 23, 28, 33]))
    write_json(run_dir / "artifacts" / "eval_results.json", report)
    write_eval_report(report, run_dir / "reports")


if __name__ == "__main__":
    main()
