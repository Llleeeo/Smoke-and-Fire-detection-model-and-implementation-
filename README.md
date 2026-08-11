# Smoke and Fire Detection with YOLOv5

YOLOv5 research project for detecting `cigarette`, `flame`, and `smoke`. The repository keeps the original prototype for reference and provides a separate, leakage-safe workflow for trustworthy retraining and evaluation.

## Repository structure

| Path | Purpose | Recommended use |
| --- | --- | --- |
| [`training/leakage_safe_yolov5/`](training/leakage_safe_yolov5/) | Dataset audit, clean split, setup, training, evaluation, inference, and published experiment outputs | **Current workflow** |
| [`models/legacy/`](models/legacy/) | Original teammate checkpoint | Baseline/reference only |
| [`legacy/`](legacy/) | Original camera, Colab, and copied YOLOv5 training code | Historical reference |
| [`docs/`](docs/) | Project reports and supporting PDFs | Documentation |

## Recommended workflow

```bash
cd training/leakage_safe_yolov5
./setup_environment.sh
source .venv/bin/activate
python scripts/verify_dataset.py dataset
./train_clean.sh
./evaluate_clean.sh
```

Generated dataset images/labels, downloaded YOLOv5 source, virtual environments, and ad-hoc runs remain local. The reviewed 45-epoch experiment, its test results, and its validated weights are published under [`training/leakage_safe_yolov5/runs/`](training/leakage_safe_yolov5/runs/).

## Latest clean-split experiment

The latest model was trained for 45 epochs at 640 px on the leakage-safe split and evaluated once on the independent 188-image test set.

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: |
| Legacy baseline | 0.461 | 0.105 | 0.084 | 0.033 |
| **Retrained model** | **0.841** | **0.680** | **0.755** | **0.453** |

- Best checkpoint: [`training/leakage_safe_yolov5/runs/full_clean_45e/weights/best.pt`](training/leakage_safe_yolov5/runs/full_clean_45e/weights/best.pt)
- Full experiment report: [`training/leakage_safe_yolov5/EXPERIMENT_RESULTS.md`](training/leakage_safe_yolov5/EXPERIMENT_RESULTS.md)

## Dataset audit status

- Clean split: 1,511 train / 188 validation / 188 test images.
- Source-name overlap across splits: 0.
- Exact SHA-256 overlap across splits: 0.
- Perceptual-hash overlap (Hamming distance ≤ 4): 0.
- Full results: [`training/leakage_safe_yolov5/audit/report.html`](training/leakage_safe_yolov5/audit/report.html).

The checkpoint in `models/legacy/best.pt` was trained on the old split and also uses `face-cigarette-smoking` where the clean dataset uses `flame`. It remains useful as initialization or a historical baseline, but the retrained checkpoint above is the current validated model.

## Security note

Credentials must be supplied through environment variables and must never be committed. The legacy Roboflow helper expects `ROBOFLOW_API_KEY`.
