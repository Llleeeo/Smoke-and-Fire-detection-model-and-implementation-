# Smoke and Fire Detection with YOLOv5

YOLOv5 research project for detecting `cigarette`, `flame`, and `smoke`. The repository keeps the original prototype for reference and provides a separate, leakage-safe workflow for trustworthy retraining and evaluation.

## Repository structure

| Path | Purpose | Recommended use |
| --- | --- | --- |
| [`training/leakage_safe_yolov5/`](training/leakage_safe_yolov5/) | Dataset audit, clean split, setup, training, evaluation, inference, and published experiment outputs | **Current workflow** |
| [`models/legacy/`](models/legacy/) | Historical checkpoint used as the published experiment baseline and initialisation | Baseline/reference only |
| [`legacy/`](legacy/) | Original camera, Colab, and copied YOLOv5 training code | Historical reference |
| [`docs/`](docs/) | Project reports and supporting PDFs | Documentation |
| [`docs/generated-output/`](docs/generated-output/) | Final ePoster, earlier poster exports, PDFs, and rendered previews | Submission/archive |

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

The exact raw teammate handoff and the generated leakage-safe dataset are backed up separately in the private repository `Llleeeo/Smoke-and-Fire-detection-data-private`. That repository contains restore instructions and SHA-256 manifests. Access requires permission from the owner; no credentials are stored in either repository.

## Latest split-clean experiment (development evidence)

The latest model was trained for 45 epochs at 640 px on the leakage-safe split and evaluated once on the 188-image test set. A later annotation audit found that historical class 1 contains both flame and face/cigarette-smoking regions. The table therefore documents the completed development experiment, but it is not final three-class paper evidence.

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

The checkpoint in `models/legacy/best.pt` was trained on the old split and names class 1 `face-cigarette-smoking`; the clean YAML renamed this class to `flame`, but renaming did not adjudicate mixed annotation semantics. It is the checkpoint actually used to initialise and benchmark the published 45-epoch development run. The later teammate handoff folder `finetune_D3_stageD92iteration` contains a different historical checkpoint and is preserved verbatim in the private data repository. Neither legacy checkpoint should be treated as a final three-class paper model.

## Paper extension

The next research phase treats the current metrics as development evidence and tests how leakage and the mixed historical class ontology affect the conclusion. The 498-row ontology audit is complete, and the four-cell controlled factorial datasets are reproducibly specified for seeds 0, 1, and 2. The preregistered design, paper outline, experiment log, and deadline-based venue plan are under [`research/`](research/); the executable matrix is under [`training/leakage_safe_yolov5/experiments/factorial/`](training/leakage_safe_yolov5/experiments/factorial/). The external holdout remains a blocker for final paper claims.

## Security note

Credentials must be supplied through environment variables and must never be committed. The legacy Roboflow helper expects `ROBOFLOW_API_KEY`.
