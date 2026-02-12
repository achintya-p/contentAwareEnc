#!/usr/bin/env python
"""train_policy.py – Train a content-adaptive CRF selection policy.

Supports two backends:
    heuristic – simple motion-score threshold mapping (no deps).
    sklearn   – basic DecisionTree / Ridge regression (needs scikit-learn).

Usage:
    python -m content_aware_encoder.train_policy --config configs/default.yaml --run_dir results/demo --dry_run
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

from content_aware_encoder.common.config import load_config, resolve_run_dir
from content_aware_encoder.common.io import read_jsonl, write_json
from content_aware_encoder.common.logging import get_logger, setup_logging
from content_aware_encoder.common.utils import add_common_args, cli_overrides

log = get_logger("train_policy")


# ── Heuristic policy ─────────────────────────────────────────────────────

def _train_heuristic(rows: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    """Choose CRF based on motion_score thresholds.

    Higher motion → use lower CRF (preserve quality under motion).
    """
    thresholds = [
        {"motion_max": 5.0, "recommended_crf": 33},
        {"motion_max": 12.0, "recommended_crf": 28},
        {"motion_max": 20.0, "recommended_crf": 23},
        {"motion_max": float("inf"), "recommended_crf": 18},
    ]
    model: Dict[str, Any] = {
        "type": "heuristic",
        "version": "0.1.0",
        "thresholds": thresholds,
        "training_rows": len(rows),
    }
    return model


def _predict_heuristic(model: Dict[str, Any], motion_score: float) -> int:
    for t in model["thresholds"]:
        if motion_score <= t["motion_max"]:
            return int(t["recommended_crf"])
    return 23


# ── Sklearn policy ────────────────────────────────────────────────────────

def _train_sklearn(rows: List[Dict[str, Any]], seed: int) -> Dict[str, Any] | None:
    try:
        import numpy as np  # type: ignore
        from sklearn.tree import DecisionTreeClassifier  # type: ignore
        import pickle
    except ImportError:
        log.warning("scikit-learn not installed – cannot train sklearn policy.")
        return None

    if not rows:
        return None

    feature_keys = ["motion_score", "edge_density", "brightness_mean", "brightness_std"]
    X = np.array([[r.get(k, 0.0) for k in feature_keys] for r in rows])

    # Label: for each row pick the CRF; we pick the one with best
    # quality-per-bit (PSNR / bitrate).  In a stub dataset every row
    # is already one CRF, so we use the CRF directly.
    y = np.array([r.get("crf", 23) for r in rows])

    clf = DecisionTreeClassifier(max_depth=4, random_state=seed)
    clf.fit(X, y)

    model_bytes = pickle.dumps(clf)
    model: Dict[str, Any] = {
        "type": "sklearn",
        "version": "0.1.0",
        "algorithm": "DecisionTreeClassifier",
        "feature_keys": feature_keys,
        "training_rows": len(rows),
        "model_pkl_b64": __import__("base64").b64encode(model_bytes).decode(),
    }
    return model


# ── Public API ────────────────────────────────────────────────────────────

def train_policy(
    dataset_path: Path,
    model_type: str = "heuristic",
    seed: int = 42,
) -> Dict[str, Any]:
    """Train (or derive) a policy model from *dataset_path*."""
    rows = read_jsonl(dataset_path) if dataset_path.exists() else []
    log.info("Training policy '%s' on %d rows from %s", model_type, len(rows), dataset_path)

    if model_type == "sklearn":
        model = _train_sklearn(rows, seed)
        if model is not None:
            return model
        log.info("Falling back to heuristic policy.")

    return _train_heuristic(rows, seed)


def predict_crf(model: Dict[str, Any], motion_score: float) -> int:
    """Predict the best CRF given a trained model and motion_score."""
    if model["type"] == "heuristic":
        return _predict_heuristic(model, motion_score)
    if model["type"] == "sklearn":
        try:
            import pickle, base64, numpy as np  # type: ignore
            clf = pickle.loads(base64.b64decode(model["model_pkl_b64"]))
            # For prediction we only have motion_score readily; pad others with 0
            X = np.array([[motion_score, 0.0, 0.0, 0.0]])
            return int(clf.predict(X)[0])
        except Exception as exc:
            log.warning("sklearn predict failed: %s – falling back to CRF 23.", exc)
    return 23


def save_policy(model: Dict[str, Any], models_dir: Path) -> Path:
    dest = models_dir / "policy.json"
    write_json(dest, model)
    log.info("Policy model → %s", dest)
    return dest


def write_training_report(model: Dict[str, Any], reports_dir: Path) -> Path:
    dest = reports_dir / "training_report.md"
    lines = [
        "# Policy Training Report",
        "",
        f"- **Model type:** {model.get('type', 'unknown')}",
        f"- **Version:** {model.get('version', '?')}",
        f"- **Training rows:** {model.get('training_rows', 0)}",
        "",
    ]
    if model["type"] == "heuristic":
        lines.append("## Threshold Table")
        lines.append("")
        lines.append("| Motion ≤ | Recommended CRF |")
        lines.append("|----------|-----------------|")
        for t in model.get("thresholds", []):
            mm = t["motion_max"] if t["motion_max"] != float("inf") else "∞"
            lines.append(f"| {mm} | {t['recommended_crf']} |")
    elif model["type"] == "sklearn":
        lines.append(f"- **Algorithm:** {model.get('algorithm', '?')}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")
    log.info("Training report → %s", dest)
    return dest


# ── CLI ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train encoding policy from dataset.")
    add_common_args(parser)
    parser.add_argument("--dataset", type=str, help="Path to dataset.jsonl.")
    parser.add_argument("--model_type", type=str, choices=["heuristic", "sklearn"], help="Policy type.")
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = cli_overrides(args)
    if args.model_type:
        overrides["policy_model_type"] = args.model_type

    cfg = load_config(args.config, overrides)
    run_dir = Path(args.run_dir) if args.run_dir else resolve_run_dir(cfg)

    setup_logging(log_file=run_dir / "logs" / "run.log", verbose=cfg.get("verbose", False))

    dataset_path = Path(args.dataset) if args.dataset else run_dir / "dataset.jsonl"
    model = train_policy(
        dataset_path=dataset_path,
        model_type=cfg.get("policy_model_type", "heuristic"),
        seed=cfg.get("random_seed", 42),
    )
    save_policy(model, run_dir / "models")
    write_training_report(model, run_dir / "reports")


if __name__ == "__main__":
    main()
