[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("A", "B", "C", "D")]
    [string]$Cell,
    [Parameter(Mandatory = $true)][int]$Seed,
    [Parameter(Mandatory = $true)][int]$Epochs,
    [string]$Device = "0",
    [int]$BatchSize = 16,
    [int]$Workers = 8,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AdditionalArguments
)

$ErrorActionPreference = "Stop"
$FactorialDir = $PSScriptRoot
$PackageDir = (Resolve-Path -LiteralPath (Join-Path $FactorialDir "..\..")).Path
$Python = Join-Path $PackageDir ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment missing: $Python"
}

$CellNames = @{
    A = "clean_original"
    B = "clean_audited"
    C = "leaked_original"
    D = "leaked_audited"
}
& $Python (Join-Path $PackageDir "scripts\prepare_factorial_configs.py") --seed $Seed | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Factorial configuration validation failed." }

$SeedSuffix = if ($Cell -in @("C", "D")) { "_seed$Seed" } else { "" }
$DataYaml = Join-Path $FactorialDir "generated\${Cell}_$($CellNames[$Cell])${SeedSuffix}.yaml"
if (-not (Test-Path -LiteralPath $DataYaml -PathType Leaf)) {
    throw "Cell $Cell seed $Seed is not ready."
}
$ProjectDir = Join-Path $PackageDir "runs\factorial_yolov5"
$RunName = "${Cell}_$($CellNames[$Cell])_seed${Seed}_${Epochs}e"
if (Test-Path -LiteralPath (Join-Path $ProjectDir $RunName)) {
    throw "Refusing to reuse existing run directory: $(Join-Path $ProjectDir $RunName)"
}

$Arguments = @(
    (Join-Path $PackageDir "yolov5\train.py"),
    "--weights", (Join-Path $PackageDir "yolov5s.pt"),
    "--data", $DataYaml,
    "--hyp", (Join-Path $PackageDir "original_model_reference\hyp.yaml"),
    "--img", "640",
    "--batch-size", "$BatchSize",
    "--epochs", "$Epochs",
    "--seed", "$Seed",
    "--device", $Device,
    "--workers", "$Workers",
    "--project", $ProjectDir,
    "--name", $RunName
)
if ($AdditionalArguments) { $Arguments += $AdditionalArguments }

if ($DryRun) {
    Write-Host "$Python $($Arguments -join ' ')"
    exit 0
}
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "YOLOv5 training failed." }
