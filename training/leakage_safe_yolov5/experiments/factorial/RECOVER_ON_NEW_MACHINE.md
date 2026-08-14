# Recover the factorial experiment on a new GPU machine

The intended GPU host uses Windows. Prefer the native PowerShell procedure in [`RECOVER_ON_WINDOWS_GPU.md`](RECOVER_ON_WINDOWS_GPU.md). This document covers Linux, WSL, and macOS.

The public code repository deliberately excludes generated image copies, virtual environments, downloaded model sources, and machine-local YAML files. The immutable inputs live in the private companion repository `Llleeeo/Smoke-and-Fire-detection-data-private`; both repositories are required.

## 1. Clone both repositories

```bash
git clone https://github.com/Llleeeo/Smoke-and-Fire-detection-model-and-implementation-.git
git clone https://github.com/Llleeeo/Smoke-and-Fire-detection-data-private.git
```

The GitHub account on the GPU machine must have access to the private repository. Do not copy or commit access tokens.

## 2. Rebuild the pinned environment

```bash
cd Smoke-and-Fire-detection-model-and-implementation-/training/leakage_safe_yolov5
./setup_environment.sh
```

This creates `.venv/` and downloads YOLOv5 at commit `3fb11111c6a8088fbc91430a1f99d207c16f0620`.

## 3. Restore and regenerate all factorial datasets

```bash
./restore_factorial_data.sh ../../../Smoke-and-Fire-detection-data-private
```

The script performs these fail-closed steps:

1. Verify the raw and clean private data against their SHA-256 manifests.
2. Restore the fixed clean split: 1,511 train, 188 validation, and 188 test images.
3. Apply the frozen 498-row ontology audit.
4. Convert historical polygon rows to tight detection boxes for consistent loaders.
5. Build paired C/D controlled-leakage datasets for seeds 0, 1, and 2.
6. Generate machine-local data YAML files and validate the complete A/B/C/D matrix.

The script refuses to overwrite an existing dataset. If it stops for that reason, inspect the existing directory instead of deleting it automatically.

## 4. Confirm the recovered matrix

```bash
for seed in 0 1 2; do
  .venv/bin/python scripts/prepare_factorial_configs.py --seed "$seed"
done
```

Every cell in `experiments/factorial/generated/MATRIX_STATUS_seed{0,1,2}.json` must report `ready`, with 1,511 training images and the common 188/188 audited validation/test sets.

## 5. Training handoff

Do not begin confirmatory training immediately. First run the four YOLOv5s seed-0 five-epoch smoke tests documented in `README.md`, inspect the outputs, and then freeze the final training protocol. The external holdout remains separate and must not be used for hyperparameter selection.

## Preserved evidence

- Frozen protocol and research log: repository `research/` directory.
- Final ontology decisions and agreement reports: `audit/` directory.
- Factorial specification and runners: this directory.
- Exact controlled-leakage choices: `manifests/` directory.
- Immutable raw and clean data checksums: private companion repository `manifests/` directory.
