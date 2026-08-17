# Windows GPU Phase 1 Smoke-Test Report

## Technical summary

The Windows GPU pipeline passed the Phase 1 validation gate. The private data repository was restored from the verified `ebfdcc8` commit, all three seed-specific factorial matrices reported A/B/C/D as `ready`, and four sequential YOLOv5s five-epoch runs completed on CUDA with seed 0, batch size 8, and four workers.

These results are execution diagnostics only. They use one seed and five epochs; they do not establish significance, causality, generalisation, or paper novelty.

## Key findings

| Cell | Condition | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Approx. wall time |
|---|---|---:|---:|---:|---:|---:|
| A | clean / original | 0.5355 | 0.5538 | 0.5247 | 0.2288 | 2m43s |
| B | clean / audited | 0.6457 | 0.5397 | 0.5263 | 0.2229 | 2m24s |
| C | controlled leakage / original | 0.4695 | 0.3983 | 0.4192 | 0.1685 | 2m23s |
| D | controlled leakage / audited | 0.4766 | 0.4119 | 0.3909 | 0.1911 | 2m30s |

Final-epoch descriptive contrasts:

| Metric | B−A | D−C | C−A | D−B | Interaction `(D−C)−(B−A)` |
|---|---:|---:|---:|---:|---:|
| Precision | +0.1102 | +0.0071 | −0.0660 | −0.1691 | −0.1031 |
| Recall | −0.0140 | +0.0137 | −0.1555 | −0.1278 | +0.0277 |
| mAP@0.5 | +0.0016 | −0.0283 | −0.1055 | −0.1353 | −0.0299 |
| mAP@0.5:0.95 | −0.0059 | +0.0226 | −0.0603 | −0.0317 | +0.0285 |

The clean/leakage contrast is negative in this short smoke test, while the audited-vs-original direction differs by metric. This is not a scientific conclusion: the run length and seed are intentionally insufficient for inference.

## Scope, data, and metric definitions

- Primary architecture: pinned YOLOv5s at commit `3fb11111c6a8088fbc91430a1f99d207c16f0620`.
- Cells: A clean/original, B clean/audited, C controlled-leakage/original, D controlled-leakage/audited.
- Common validation and internal test: 188 images each; training pool: 1,511 images per cell.
- C/D leakage endpoint: 41 of 188 internal test components (21.8% overall; 100% of strictly eligible components), with 41 matched training replacements.
- Reported metrics are the final validation values from each run's `results.csv`; best observed mAP@0.5:0.95 was A 0.2288 (epoch 5), B 0.2229 (epoch 5), C 0.1758 (epoch 4), and D 0.1990 (epoch 4).

## Methodology and validation gates

### Phase H smoke-gate result

| Gate | A | B | C | D |
|---|---|---|---|---|
| Run completed with exit success | PASS | PASS | PASS | PASS |
| Five validation rows present; numeric metrics finite | PASS | PASS | PASS | PASS |
| `results.csv`, `opt.yaml`, `weights/last.pt`, `weights/best.pt` present | PASS | PASS | PASS | PASS |
| Training/validation scan: 1,511 / 188 images; corrupt files | 0 | 0 | 0 | 0 |
| CUDA device used | RTX 4060 Laptop GPU | RTX 4060 Laptop GPU | RTX 4060 Laptop GPU | RTX 4060 Laptop GPU |

All four cells therefore passed the Phase H smoke gate. Approximate observed peak VRAM was 2.1 GiB of 8 GiB during the sequential runs; this is an observation, not a memory-capacity guarantee. Approximate wall times were A 2m43s, B 2m24s, C 2m23s, and D 2m30s. The only recurring runtime warning was the upstream PyTorch deprecation warning for `torch.cuda.amp.autocast`; it did not fail a run. A one-time PowerShell harness typo emitted a `Stop` command warning before the loop, but the four training processes and their artifacts completed successfully.

The private repository was checked at `ebfdcc859e994181fe0758074e1307ed8dea0f5e`. SHA-256 validation succeeded after using a separate LF-preserving work tree (`data-lf`) to avoid Windows `core.autocrlf` conversion. All three `MATRIX_STATUS_seed*.json` files reported A/B/C/D `ready`, including 1,511 training images, 188 validation images, 188 test images, zero exact test hashes in controlled-leakage training, and identical C/D image pools within each seed.

The environment used Python 3.11.9, PyTorch `2.12.0+cu126`, torchvision `0.27.0+cu126`, CUDA runtime 12.6, and an NVIDIA GeForce RTX 4060 Laptop GPU. `torch.cuda.is_available()` returned `True`. The common run configuration was image size 640, batch size 8, workers 4, device 0, seed 0, and 5 epochs. All four run directories contain `results.csv`, `opt.yaml`, `weights/last.pt`, and `weights/best.pt`; validation and training scans reported zero corrupt samples.

All contrasts in this report are **non-confirmatory diagnostics**. They are descriptive arithmetic on one seed and five epochs and must not be interpreted as statistical evidence, causal effects, or a basis for publication claims.

## Limitations and uncertainty

- One seed and five epochs are pipeline diagnostics, not confirmatory evidence.
- No confidence intervals, significance tests, external-holdout evaluation, or causal interpretation are included.
- Approximate wall times are derived from run-directory/result timestamps, not a dedicated process timer.
- The Windows Git checkout originally used CRLF conversion; the original clone was preserved and the LF-preserving `data-lf` work tree was used for immutable checksum validation.

## Recommended next step

Keep the four-run outputs as the Phase 1 baseline. If approved separately, run the specified 25-epoch screening stage with a new report; do not start the three-seed 45-epoch matrix or external evaluation from this smoke-test report alone.

## Reproducibility locations

- Code: `D:\fire-factorial\code`
- Verified data work tree used for restoration: `D:\fire-factorial\data-lf`
- Smoke runs: `D:\fire-factorial\code\training\leakage_safe_yolov5\runs\factorial_yolov5`
- Matrix status: `D:\fire-factorial\code\training\leakage_safe_yolov5\experiments\factorial\generated\MATRIX_STATUS_seed0.json` through `MATRIX_STATUS_seed2.json`
