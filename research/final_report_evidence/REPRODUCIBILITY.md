# Reproducibility

## Environment

- Windows with NVIDIA GeForce RTX 4060 Laptop GPU, approximately 8 GB VRAM.
- Python 3.11.9.
- PyTorch 2.12.0+cu126; torchvision 0.27.0+cu126; CUDA 12.6.
- YOLOv5 commit: `3fb11111c6a8088fbc91430a1f99d207c16f0620`.
- Private data commit: `ebfdcc859e994181fe0758074e1307ed8dea0f5e`.

## Formal protocol

- Architecture: YOLOv5s.
- Classes: `0 cigarette`, `1 flame`, `2 smoke`.
- Conditions: A clean/original, B clean/audited, C controlled-contamination/original, D controlled-contamination/audited.
- Seeds: 0, 1, 2.
- Epochs: 45; image size: 640; batch: 8; optimizer: SGD; device: CUDA 0.
- Seeds 0 and 1 used workers=4. All seed-2 final runs used workers=0 after the initial A/seed2 DataLoader worker failure established the stable recovery configuration.
- Internal evaluation used each run's `best.pt` on the fixed 188-image/167-instance test.

## Public evidence boundaries

The public repository contains code, CSVs, Markdown, figures, commit identities, and SHA-256 manifests. It excludes private images and labels, credentials, caches, generated dataset copies, and the 12 formal checkpoint binaries. Local checkpoint identity can be verified with `CHECKPOINT_MANIFEST.csv`.

The canonical result tables are `RUN_REGISTRY.csv` and `INTERNAL_RESULTS.csv`. Missing values are encoded as `NA`.
