# Smoke and Fire Detection with YOLOv5

YOLOv5 research project for detecting `cigarette`, `flame`, and `smoke`. The repository keeps the original prototype for reference and provides a separate, leakage-safe workflow for trustworthy retraining and evaluation.

## Repository structure

| Path | Purpose | Recommended use |
| --- | --- | --- |
| [`training/leakage_safe_yolov5/`](training/leakage_safe_yolov5/) | Dataset audit, clean split, setup, training, evaluation, and inference | **Current workflow** |
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

Generated dataset images/labels, downloaded YOLOv5 source, virtual environments, and training runs remain local and are excluded from GitHub.

## Dataset audit status

- Clean split: 1,511 train / 188 validation / 188 test images.
- Source-name overlap across splits: 0.
- Exact SHA-256 overlap across splits: 0.
- Perceptual-hash overlap (Hamming distance ≤ 4): 0.
- Full results: [`training/leakage_safe_yolov5/audit/report.html`](training/leakage_safe_yolov5/audit/report.html).

The checkpoint in `models/legacy/best.pt` was trained on the old split. It can be used as an initialization or baseline, but new metrics should only be reported after retraining and evaluation on the clean split.

## Security note

Credentials must be supplied through environment variables and must never be committed. The legacy Roboflow helper expects `ROBOFLOW_API_KEY`.
