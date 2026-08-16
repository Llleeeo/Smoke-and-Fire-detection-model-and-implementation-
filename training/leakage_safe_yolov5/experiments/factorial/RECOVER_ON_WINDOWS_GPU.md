# Recover and screen the factorial experiment on Windows/NVIDIA

This is the primary handoff for the future Windows GPU computer. Use 64-bit Python, Git, PowerShell, a current NVIDIA driver, and enough free disk space for the two repositories, generated datasets, environments, and training runs.

## 1. Clone the public branch and private data repository

Until the research pull request is merged, replace `<research-branch>` with the branch supplied in the handoff. After merge, omit the two branch commands.

```powershell
git clone https://github.com/Llleeeo/Smoke-and-Fire-detection-model-and-implementation-.git
git clone https://github.com/Llleeeo/Smoke-and-Fire-detection-data-private.git
Set-Location .\Smoke-and-Fire-detection-model-and-implementation-
git switch <research-branch>
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

If it prints `False`, stop. Have Codex inspect the installed NVIDIA driver and install the appropriate CUDA-enabled PyTorch build from the official PyTorch instructions before training.

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

## 4. Run only the four five-epoch YOLOv5s smoke tests first

```powershell
.\experiments\factorial\run_yolov5.ps1 -Cell A -Seed 0 -Epochs 5 -Device 0 -BatchSize 16
.\experiments\factorial\run_yolov5.ps1 -Cell B -Seed 0 -Epochs 5 -Device 0 -BatchSize 16
.\experiments\factorial\run_yolov5.ps1 -Cell C -Seed 0 -Epochs 5 -Device 0 -BatchSize 16
.\experiments\factorial\run_yolov5.ps1 -Cell D -Seed 0 -Epochs 5 -Device 0 -BatchSize 16
```

If memory is insufficient, reduce all four runs to the same batch size. Do not change only one cell. These five-epoch runs validate the pipeline and are not paper evidence.

## 5. Preserve outputs before deciding on confirmatory training

Record GPU model, driver, CUDA/PyTorch versions, command lines, wall time, peak memory, and the four run directories. Compare A/B/C/D validation trends and variance before launching the three-seed 45-epoch matrix. Do not evaluate or tune on the locked external holdout at this stage.

## 6. YOLO26n replication environment (after YOLOv5 screening)

Keep the replication architecture in a separate environment:

```powershell
py -3 -m venv .venv_yolo26
.\.venv_yolo26\Scripts\python.exe -m pip install --upgrade pip
.\.venv_yolo26\Scripts\python.exe -m pip install -r .\experiments\factorial\requirements_yolo26.txt
.\.venv_yolo26\Scripts\python.exe .\experiments\factorial\run_yolo26.py A 0 5 --device 0
```

Run the replication only after CUDA availability and the primary YOLOv5 pipeline have been confirmed.
