# AI Fireguard: YOLOv5 Fire-Safety Detection Research

This repository contains the evidence selected for the URIS Completion Report. The project detects `cigarette`, `flame`, and `smoke`, audits dataset leakage and label semantics, and evaluates a controlled four-condition YOLOv5 experiment on a fixed internal test set.

Historical prototypes, superseded experiments, later exploratory work, and venue-planning files are preserved in the [`archive/pre-final-report-20260820`](https://github.com/Llleeeo/Smoke-and-Fire-detection-model-and-implementation-/tree/archive/pre-final-report-20260820) branch.

## Final-report evidence

| Path | Contents |
| --- | --- |
| [`docs/`](docs/) | Previous URIS report, escape-route prototype, final ePoster, and report source map |
| [`research/`](research/) | Experiment log, ontology protocol, final internal results, figures, and reproducibility records |
| [`training/leakage_safe_yolov5/`](training/leakage_safe_yolov5/) | Dataset auditing, ontology review, factorial dataset construction, training, evaluation, and inference code |

## Dataset and annotation audit

- Raw export: 5,068 image records grouped into 1,887 independent perceptual components.
- Leakage-safe split: 1,511 train / 188 validation / 188 internal-test images.
- Cross-split source-name, exact SHA-256, and pHash-distance-at-most-4 overlap: 0.
- Ontology audit: 498 historical class-1 annotations reviewed.
- Independent review sample: 96% raw agreement; Cohen's kappa 0.903.
- Final actions: 371 boxes kept as flame, 95 relabelled as cigarette, and 32 invalid or smoking-action boxes removed.

## Formal internal experiment

The completed experiment crosses split condition and training-label ontology:

| Condition | Split condition | Training ontology |
| --- | --- | --- |
| A | leakage-safe | original mixed labels |
| B | leakage-safe | audited labels |
| C | controlled near-duplicate contamination | original mixed labels |
| D | controlled near-duplicate contamination | audited labels |

All four conditions use YOLOv5s, seeds 0/1/2, 45 epochs, 640 px input, batch size 8, SGD, and the same audited validation and internal-test labels. All 12 formal runs and 12 internal evaluations completed.

### Internal test, mean +/- sample SD across three seeds

| Condition | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: |
| A | 0.7970 +/- 0.0576 | 0.7280 +/- 0.0219 | 0.7700 +/- 0.0223 | 0.4630 +/- 0.0193 |
| B | **0.8487 +/- 0.0245** | 0.7420 +/- 0.0187 | **0.8070 +/- 0.0295** | **0.4767 +/- 0.0200** |
| C | 0.8130 +/- 0.0308 | 0.6957 +/- 0.0049 | 0.7647 +/- 0.0212 | 0.4587 +/- 0.0110 |
| D | 0.7953 +/- 0.0560 | **0.7597 +/- 0.0185** | 0.8013 +/- 0.0274 | 0.4753 +/- 0.0203 |

These are descriptive internal results. They do not establish causal effects or performance in substantially different deployment environments. See [`research/final_report_evidence/METRICS_SUMMARY.md`](research/final_report_evidence/METRICS_SUMMARY.md) for the verified result tables.

## Reproduction

Public code excludes private images, labels, credentials, and the 12 formal checkpoint binaries. Dataset and checkpoint manifests preserve provenance and identity. Restoration instructions are under [`training/leakage_safe_yolov5/experiments/factorial/`](training/leakage_safe_yolov5/experiments/factorial/).

Credentials must be supplied through environment variables and must never be committed.
