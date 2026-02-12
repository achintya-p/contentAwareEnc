# Content-Aware Video Encoding Policy

> Learn a content-adaptive encoding decision layer: given an input clip's
> features, choose the CRF (quality dial) that minimises bitrate while
> preserving perceptual quality **and** downstream computer-vision
> performance.

---

## Quick-start (dry run – no GPU, no ffmpeg, no large models)

```bash
# From the repo root (the directory containing content_aware_encoder/)
python -m content_aware_encoder.scripts.run_pipeline \
    --config content_aware_encoder/configs/default.yaml \
    --run_name demo \
    --dry_run --verbose
```

This produces a complete run directory at `results/demo/` with:

| Path | Contents |
|------|----------|
| `encodes/` | Placeholder encoded files + `encode_manifest.json` |
| `artifacts/features.json` | Content features (stubbed) |
| `artifacts/metrics.json` | Bitrate & PSNR per variant |
| `artifacts/detector.json` | Downstream CV metrics (stubbed) |
| `dataset.jsonl` | Merged dataset rows |
| `models/policy.json` | Trained (heuristic) policy |
| `reports/summary.md` | Human-readable run summary |
| `reports/eval_report.md` | Policy vs fixed-CRF comparison |
| `reports/training_report.md` | Policy training details |
| `logs/run.log` | Structured log |
| `manifest.json` | Reproducibility manifest (config, git hash, env) |

---

## Running with real encoding

1. Install ffmpeg:
   ```bash
   brew install ffmpeg   # macOS
   sudo apt install ffmpeg  # Ubuntu
   ```
2. Place an input video (e.g. `sample.mp4`) in the repo root.
3. Run:
   ```bash
   python -m content_aware_encoder.scripts.run_pipeline \
       --config content_aware_encoder/configs/default.yaml \
       --run_name real_run
   ```
   The pipeline will use ffmpeg for encoding and (if OpenCV is installed)
   extract real content features.

---

## Running individual steps

Each module works standalone:

```bash
# Encode only
python -m content_aware_encoder.encode \
    --config content_aware_encoder/configs/default.yaml \
    --run_dir results/demo --dry_run

# Features only
python -m content_aware_encoder.features \
    --config content_aware_encoder/configs/default.yaml \
    --run_dir results/demo --dry_run

# Metrics (requires encode step first)
python -m content_aware_encoder.metrics \
    --config content_aware_encoder/configs/default.yaml \
    --run_dir results/demo --dry_run

# Detector (requires encode step first)
python -m content_aware_encoder.run_detector \
    --config content_aware_encoder/configs/default.yaml \
    --run_dir results/demo --dry_run

# Train policy (requires dataset.jsonl from pipeline)
python -m content_aware_encoder.train_policy \
    --config content_aware_encoder/configs/default.yaml \
    --run_dir results/demo

# Evaluate policy (requires dataset.jsonl + policy.json)
python -m content_aware_encoder.evaluate_policy \
    --config content_aware_encoder/configs/default.yaml \
    --run_dir results/demo
```

---

## Configuration

Edit `configs/default.yaml` to change:

| Key | Default | Description |
|-----|---------|-------------|
| `input_video` | `sample.mp4` | Path to source video |
| `codec` | `libx264` | FFmpeg video codec |
| `crf_list` | `[18, 23, 28, 33]` | CRF values to test |
| `frame_sample_rate` | `1` | Sample every Nth frame for features |
| `detector_backend` | `stub` | `stub` or `yolo` |
| `policy_model_type` | `heuristic` | `heuristic` or `sklearn` |
| `random_seed` | `42` | For reproducibility |
| `dry_run` | `false` | Skip heavy computation |

---

## Run directory layout

```
results/<run_name>/
├── encodes/
│   ├── sample_libx264_crf18.mp4
│   ├── sample_libx264_crf23.mp4
│   ├── sample_libx264_crf28.mp4
│   ├── sample_libx264_crf33.mp4
│   └── encode_manifest.json
├── artifacts/
│   ├── features.json
│   ├── metrics.json
│   ├── detector.json
│   └── eval_results.json
├── models/
│   └── policy.json
├── reports/
│   ├── summary.md
│   ├── eval_report.md
│   └── training_report.md
├── logs/
│   └── run.log
├── dataset.jsonl
└── manifest.json
```

---

## Optional dependencies

| Package | Enables |
|---------|---------|
| `pyyaml` | Robust YAML config loading (built-in fallback exists) |
| `opencv-python` | Real feature extraction from video frames |
| `scikit-learn` | ML-based policy training (DecisionTree) |
| `ultralytics` | YOLO detector backend |

Install all optional deps:
```bash
pip install pyyaml opencv-python scikit-learn ultralytics
```

---

## Next steps

- [ ] Add SSIM and VMAF quality metrics
- [ ] Add real YOLO detector backend with mAP tracking
- [ ] Per-segment (scene-level) encoding decisions
- [ ] Multi-codec support (H.265, AV1, VP9)
- [ ] Pareto-front visualisation (bitrate vs quality vs detection)
- [ ] Hyperparameter sweep for policy training
- [ ] CI pipeline with dry_run smoke test
