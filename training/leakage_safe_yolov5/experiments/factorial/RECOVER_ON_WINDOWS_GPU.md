# Recover and reproduce the factorial experiment on Windows/NVIDIA

This is the primary recovery procedure for a Windows GPU computer. Use 64-bit Python, Git, PowerShell, a current NVIDIA driver, and enough free disk space for the two repositories, generated datasets, environments, and training runs.

## 1. Clone the public and private repositories

```powershell
git clone https://github.com/Llleeeo/Smoke-and-Fire-detection-model-and-implementation-.git
git clone https://github.com/Llleeeo/Smoke-and-Fire-detection-data-private.git
Set-Location .\Smoke-and-Fire-detection-model-and-implementation-
```

The signed-in GitHub account must be authorised for the private data repository. Never paste a token into a script or commit it.

## 2. Build the pinned YOLOv5 environment

```powershell
Set-ExecutionPolicy -Scope Process Bypass
Set-Location .\training\leakage_safe_yolov5
.\setup_environment.ps1 -PythonVersion 3.11
```

The script creates `.venv`, downloads YOLOv5, checks out commit `3fb11111c6a8088fbc91430a1f99d207c16f0620`, installs dependencies, and prints the PyTorch CUDA status.

Before continuing, this command must print `True` and identify the NVIDIA GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

If it prints `False`, stop and resolve the NVIDIA driver or CUDA-enabled PyTorch installation using the official PyTorch instructions before training.

## 3. Restore and verify all datasets

From `training\leakage_safe_yolov5`:

```powershell
.\restore_factorial_data.ps1 -PrivateRepository ..\..\..\Smoke-and-Fire-detection-data-private
```

This verifies 10,137 raw files and the fixed clean dataset by SHA-256, restores the 1,511/188/188 split, applies the frozen 498-row audit, builds detection-only A/B and seed-specific C/D, and compares every C/D choice against the committed frozen manifests. It refuses to overwrite existing datasets.

Confirm all cells report `ready`:

```powershell
Get-Content .\experiments\factorial\generated\MATRIX_STATUS_seed0.json
Get-Content .\experiments\factorial\generated\MATRIX_STATUS_seed1.json
Get-Content .\experiments\factorial\generated\MATRIX_STATUS_seed2.json
```

## 4. Run a short pipeline check

```powershell
.\experiments\factorial\run_yolov5.ps1 -Cell A -Seed 0 -Epochs 5 -Device 0 -BatchSize 16
.\experiments\factorial\run_yolov5.ps1 -Cell B -Seed 0 -Epochs 5 -Device 0 -BatchSize 16
.\experiments\factorial\run_yolov5.ps1 -Cell C -Seed 0 -Epochs 5 -Device 0 -BatchSize 16
.\experiments\factorial\run_yolov5.ps1 -Cell D -Seed 0 -Epochs 5 -Device 0 -BatchSize 16
```

If memory is insufficient, reduce all four runs to the same batch size. Do not change only one cell. These five-epoch checks validate the pipeline and are not final performance evidence.

## 5. Reproduce the formal matrix

Run cells A-D for seeds 0-2 with 45 epochs and batch size 8. Preserve the GPU model, driver, CUDA/PyTorch versions, command lines, wall time, and run directories. Compare results with the checkpoint identities and aggregate metrics under `research/final_report_evidence/`.

```powershell
.\experiments\factorial\run_yolov5.ps1 -Cell A -Seed 0 -Epochs 45 -Device 0 -BatchSize 8
```

Repeat the command for all four cells and three seeds.
