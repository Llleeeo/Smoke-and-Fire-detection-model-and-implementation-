[CmdletBinding()]
param(
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"
$PackageDir = $PSScriptRoot
$Yolov5Dir = Join-Path $PackageDir "yolov5"
$VenvDir = Join-Path $PackageDir ".venv"
$Yolov5Commit = "3fb11111c6a8088fbc91430a1f99d207c16f0620"

if (-not (Test-Path -LiteralPath (Join-Path $Yolov5Dir "train.py"))) {
    & git clone --filter=blob:none https://github.com/ultralytics/yolov5.git $Yolov5Dir
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone YOLOv5." }
    & git -C $Yolov5Dir checkout $Yolov5Commit
    if ($LASTEXITCODE -ne 0) { throw "Failed to check out pinned YOLOv5 commit." }
}
elseif (Test-Path -LiteralPath (Join-Path $Yolov5Dir ".git")) {
    $CurrentCommit = (& git -C $Yolov5Dir rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Failed to inspect YOLOv5 commit." }
    if ($CurrentCommit -ne $Yolov5Commit) {
        Write-Warning "YOLOv5 is at $CurrentCommit; expected $Yolov5Commit."
    }
}

if (Test-Path -LiteralPath $VenvDir) {
    throw "Refusing to replace existing environment: $VenvDir"
}

if ($PythonLauncher -eq "py") {
    & py "-$PythonVersion" -m venv $VenvDir
}
else {
    & $PythonLauncher -m venv $VenvDir
}
if ($LASTEXITCODE -ne 0) { throw "Failed to create Python environment." }

$Python = Join-Path $VenvDir "Scripts\python.exe"
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }
& $Python -m pip install -r (Join-Path $Yolov5Dir "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to install YOLOv5 requirements." }
& $Python -m pip install -r (Join-Path $PackageDir "requirements-audit.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to install audit requirements." }

& $Python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_devices', torch.cuda.device_count())"
if ($LASTEXITCODE -ne 0) { throw "Failed to inspect PyTorch CUDA support." }

Write-Host "Environment ready: $Python"
Write-Host "Before training, cuda_available must be True on the GPU computer."
