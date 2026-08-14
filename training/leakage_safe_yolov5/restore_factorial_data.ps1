[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PrivateRepository,
    [string]$Python
)

$ErrorActionPreference = "Stop"
$PackageDir = $PSScriptRoot
$PrivateRoot = (Resolve-Path -LiteralPath $PrivateRepository).Path
$RawDataset = Join-Path $PrivateRoot "handoff\yolo 5"
$CleanSource = Join-Path $PrivateRoot "cleaned\leakage_safe_yolov5\dataset"
if (-not $Python) {
    $Python = Join-Path $PackageDir ".venv\Scripts\python.exe"
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment missing: $Python. Run .\setup_environment.ps1 first."
}
if (-not (Test-Path -LiteralPath $RawDataset -PathType Container) -or
    -not (Test-Path -LiteralPath $CleanSource -PathType Container)) {
    throw "Private repository is missing the raw or cleaned dataset."
}

function Test-HashManifest {
    param([Parameter(Mandatory = $true)][string]$RelativeManifest)
    $Manifest = Join-Path $PrivateRoot $RelativeManifest
    $Count = 0
    foreach ($Line in Get-Content -LiteralPath $Manifest) {
        $Count += 1
        if ($Line -notmatch '^([0-9a-fA-F]{64})  (.+)$') {
            throw "Malformed checksum line $Count in $RelativeManifest"
        }
        $Expected = $Matches[1].ToLowerInvariant()
        $RelativeFile = $Matches[2].Replace('/', [IO.Path]::DirectorySeparatorChar)
        $File = Join-Path $PrivateRoot $RelativeFile
        if (-not (Test-Path -LiteralPath $File -PathType Leaf)) {
            throw "Missing checksum target: $RelativeFile"
        }
        $Actual = (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Actual -ne $Expected) {
            throw "SHA-256 mismatch: $RelativeFile"
        }
    }
    Write-Host "$RelativeManifest verified ($Count files)."
}

function Invoke-Python {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $($Arguments -join ' ')"
    }
}

function Assert-SameFile {
    param([Parameter(Mandatory = $true)][string]$Actual, [Parameter(Mandatory = $true)][string]$Expected)
    $ActualHash = (Get-FileHash -LiteralPath $Actual -Algorithm SHA256).Hash
    $ExpectedHash = (Get-FileHash -LiteralPath $Expected -Algorithm SHA256).Hash
    if ($ActualHash -ne $ExpectedHash) {
        throw "Recovered artifact differs from frozen manifest: $Actual"
    }
}

Write-Host "Verifying immutable private data..."
Test-HashManifest "manifests\raw-yolo5.sha256"
Test-HashManifest "manifests\cleaned-dataset.sha256"

foreach ($Split in @("train", "valid", "test")) {
    $Destination = Join-Path $PackageDir "dataset\$Split"
    if (Test-Path -LiteralPath $Destination) {
        throw "Refusing to overwrite existing dataset split: $Destination"
    }
}
$GeneratedDatasets = @(
    "dataset_audited",
    "dataset_detect_original",
    "dataset_detect_audited",
    "dataset_leaked_original_seed0", "dataset_leaked_audited_seed0",
    "dataset_leaked_original_seed1", "dataset_leaked_audited_seed1",
    "dataset_leaked_original_seed2", "dataset_leaked_audited_seed2"
)
foreach ($Name in $GeneratedDatasets) {
    $Destination = Join-Path $PackageDir $Name
    if (Test-Path -LiteralPath $Destination) {
        throw "Refusing to overwrite existing generated dataset: $Destination"
    }
}

Write-Host "Restoring the verified 1,511/188/188 clean split..."
foreach ($Split in @("train", "valid", "test")) {
    Copy-Item -LiteralPath (Join-Path $CleanSource $Split) -Destination (Join-Path $PackageDir "dataset\$Split") -Recurse
}

Write-Host "Applying the frozen 498-row ontology audit..."
Invoke-Python -Arguments @(
    (Join-Path $PackageDir "scripts\build_audited_dataset.py"),
    (Join-Path $PackageDir "dataset"),
    (Join-Path $PackageDir "audit\ontology_review_final.csv"),
    (Join-Path $PackageDir "dataset_audited")
)

Write-Host "Normalising original and audited labels to detection boxes..."
Invoke-Python -Arguments @(
    (Join-Path $PackageDir "scripts\build_detect_only_dataset.py"),
    (Join-Path $PackageDir "dataset"),
    (Join-Path $PackageDir "dataset_detect_original")
)
Invoke-Python -Arguments @(
    (Join-Path $PackageDir "scripts\build_detect_only_dataset.py"),
    (Join-Path $PackageDir "dataset_audited"),
    (Join-Path $PackageDir "dataset_detect_audited")
)

Write-Host "Building controlled-leakage C/D datasets for seeds 0, 1, and 2..."
Invoke-Python -Arguments @(
    (Join-Path $PackageDir "scripts\build_controlled_leakage_datasets.py"),
    $RawDataset,
    $PackageDir,
    "--seeds", "0", "1", "2"
)

$ExpectedDir = Join-Path $PackageDir "experiments\factorial\manifests"
foreach ($Seed in @(0, 1, 2)) {
    foreach ($Ontology in @("original", "audited")) {
        $GeneratedDir = Join-Path $PackageDir "dataset_leaked_${Ontology}_seed${Seed}"
        Assert-SameFile -Actual (Join-Path $GeneratedDir "CONTROLLED_LEAKAGE_MANIFEST.csv") -Expected (Join-Path $ExpectedDir "CONTROLLED_LEAKAGE_MANIFEST_seed${Seed}.csv")
        Assert-SameFile -Actual (Join-Path $GeneratedDir "CONTROLLED_LEAKAGE_SUMMARY.json") -Expected (Join-Path $ExpectedDir "CONTROLLED_LEAKAGE_SUMMARY_seed${Seed}.json")
    }
    Invoke-Python -Arguments @(
        (Join-Path $PackageDir "scripts\prepare_factorial_configs.py"),
        "--seed", "$Seed"
    )
}

Write-Host "Factorial data restoration complete."
Write-Host "Inspect experiments\factorial\generated\MATRIX_STATUS_seed0.json through seed2.json."
