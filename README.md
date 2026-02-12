# Content-Aware Video Encoding Policy

This repo builds a **content-adaptive encoding decision layer** for streaming-style workloads.

Instead of using one fixed encoding ladder for every video, we:
1) extract **content features** (motion, texture/edges, brightness statistics),
2) generate multiple **encoded variants** (e.g., H.264 at different CRFs),
3) evaluate each variant by:
   - **bitrate** (cost / size),
   - **perceptual quality proxy** (PSNR now, SSIM later),
   - **downstream computer vision performance** (YOLO detections/confidence),
4) learn a **policy** that predicts the best encoding setting for new clips:
   `clip_features → encoding_choice`.

The goal is to minimize bitrate while preserving perceptual quality and keeping downstream CV performance stable.

---

## Project Objective

Given an input clip, choose an encoding setting (initially `codec=H.264` and `CRF ∈ {18, 23, 28, 33}`) that optimizes a tradeoff:

- lower bitrate is better
- higher perceptual quality is better
- higher CV stability/performance is better

This mirrors real-world streaming tradeoffs: quality vs cost, and content-aware decisions.

---

## Expected Outputs

For each input clip, the pipeline should produce:

- Encoded files in `results/<run_name>/encodes/`
- A machine-readable summary:
  - `results/<run_name>/metrics.json`
- A human-readable table:
  - `results/<run_name>/summary.md` (or `.csv`)

At the end of the week:
- a trained model checkpoint in `results/models/`
- an evaluation report comparing:
  - fixed CRF baseline vs learned policy

---

## File Overview

### `encode.py`
Encodes a video into multiple variants using ffmpeg.

Responsibilities:
- run `ffmpeg` to create multiple outputs for a given codec + CRF sweep
- store encode metadata:
  - output path
  - duration
  - bitrate (from ffprobe or file size / duration)
- write a JSON manifest for the run

Typical settings:
- codec: `libx264` (H.264)
- CRFs: `[18, 23, 28, 33]`
- preset: `veryfast` (for iteration)

---

### `features.py`
Extracts content features from a video clip (computed once per original clip).

Minimum features:
- **motion score**: mean absolute difference between consecutive frames
- **edge density / texture**: mean Sobel magnitude (or Laplacian variance)
- **brightness stats**: grayscale mean and std

Output:
- a dict (or JSON) of features for the clip

---

### `metrics.py`
Computes evaluation metrics for each encoded variant.

Minimum metrics:
- **bitrate**: kbps from ffprobe or size/duration
- **perceptual proxy**: PSNR vs reference (reference = highest-quality encode, e.g. CRF 18)

Planned upgrade:
- add SSIM / MS-SSIM
- add segment-level metrics

Output:
- a dict (or JSON) of metrics per encoded file

---

### `run_detector.py`
Runs a pretrained CV model on each encoded variant to measure downstream impact.

Minimum v0:
- YOLOv8n inference on sampled frames
- report:
  - detections per frame
  - average confidence
  - throughput (effective FPS)

Optional (stronger):
- “agreement vs reference” metric:
  - compare detections on CRF 18 vs others (simple matching heuristic)

Output:
- `results/<run_name>/detector_metrics.json`

---

### `train_policy.py`
Trains a model that maps `features → encoding_choice`.

Label generation:
- For each clip, evaluate all candidate encodes.
- Choose the “best” encode under constraints, e.g.:
  - meet perceptual threshold AND CV threshold
  - among those, choose minimal bitrate
If no encode meets thresholds, choose the best weighted objective.

Model:
- start simple (logistic regression / small MLP / XGBoost if you want)

Output:
- saved model artifact in `results/models/`
- training report (basic accuracy + confusion matrix is enough)

---

### `evaluate_policy.py`
Evaluates the learned policy against baselines on held-out clips.

Baselines:
- fixed CRF (e.g., CRF=23)
- fixed CRF (e.g., CRF=28)

Report:
- avg bitrate
- avg perceptual score
- avg detector score
- policy vs baseline comparisons

Output:
- `results/eval_report.md` (or JSON/CSV)


## Dependencies

- Python 3.10+
- ffmpeg installed and available on PATH

Recommended Python packages:
- opencv-python
- numpy
- pandas
- ultralytics (YOLO)
- scikit-learn (for policy training)

---

## Suggested CLI (not required, but recommended)

These are the commands the repo should eventually support:

Encode sweep:
- `python encode.py --input <video.mp4> --crfs 18 23 28 33 --outdir results/demo`

Extract features:
- `python features.py --input <video.mp4> --out results/demo/features.json`

Compute metrics:
- `python metrics.py --reference results/demo/encodes/crf18.mp4 --encodes results/demo/encodes/*.mp4 --out results/demo/metrics.json`

Run detector:
- `python run_detector.py --videos results/demo/encodes/*.mp4 --out results/demo/detector.json`

Train policy:
- `python train_policy.py --dataset results/dataset.csv --out results/models/policy.pkl`

Evaluate:
- `python evaluate_policy.py --model results/models/policy.pkl --test results/test.csv --out results/eval_report.md`

---
