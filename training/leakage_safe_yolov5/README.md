# Leakage-safe YOLOv5 research workflow

This directory contains the reproducible workflow used for the URIS Completion Report: duplicate-aware splitting, annotation-ontology adjudication, controlled factorial dataset construction, three-seed training, evaluation, and inference.

## Evidence retained in `main`

- `audit/`: data-quality audit, final ontology decisions, agreement statistics, and split manifests.
- `experiments/factorial/`: A/B/C/D specifications, frozen contamination manifests, runners, and recovery instructions.
- `scripts/`: dataset reconstruction, verification, ontology review, and experiment-preparation utilities.
- `dataset/data.yaml`: the three-class schema and expected local split paths.
- shell and PowerShell entry points for macOS/Linux and Windows/NVIDIA.

Superseded checkpoints, early experiment plots, copied training snapshots, and later exploratory material are preserved in the `archive/pre-final-report-20260820` branch.

## Data restoration

The raw export and generated datasets are stored in the private repository `Llleeeo/Smoke-and-Fire-detection-data-private`. That repository contains SHA-256 manifests and does not expose credentials in this public repository.

Linux, WSL, or macOS:

```bash
./restore_factorial_data.sh /path/to/Smoke-and-Fire-detection-data-private
```

Windows PowerShell:

```powershell
.\restore_factorial_data.ps1 -PrivateRepository ..\..\..\Smoke-and-Fire-detection-data-private
```

The restoration workflow verifies the raw and clean inputs, restores the 1,511/188/188 split, applies the frozen 498-row ontology audit, creates paired A/B datasets, and recreates seed-specific C/D controlled-contamination datasets.

## Formal design

| Cell | Split condition | Training ontology | Seeds |
| --- | --- | --- | --- |
| A | leakage-safe | original mixed | 0, 1, 2 |
| B | leakage-safe | audited | 0, 1, 2 |
| C | controlled near-duplicate contamination | original mixed | 0, 1, 2 |
| D | controlled near-duplicate contamination | audited | 0, 1, 2 |

The formal YOLOv5s runs use 45 epochs, 640 px images, batch size 8, SGD, generic pretrained initialization, and a common audited validation/internal-test ontology. The verified result archive is [`../../research/final_report_evidence/`](../../research/final_report_evidence/).

## Dataset verification

```bash
python3 scripts/verify_dataset.py dataset
```

## Inference and evaluation

Provide a locally restored formal checkpoint explicitly:

```bash
WEIGHTS=/path/to/best.pt ./infer_clean.sh /path/to/image-or-video
WEIGHTS=/path/to/best.pt ./evaluate_clean.sh
```

The internal result tables are descriptive. They do not establish performance in substantially different deployment environments.
