# Windows GPU reproduction handoff

This document is the complete execution specification for the target Windows computer with an NVIDIA GPU. Follow it directly and record the requested results without relying on missing instructions from another environment.

## Objective

Restore the frozen fire/smoke object-detection research workflow from GitHub, verify that the Windows CUDA environment and all factorial datasets are reproducible, and run the four YOLOv5s seed-0 five-epoch A/B/C/D smoke tests under identical settings.

This is Phase 1 pipeline validation. Do not start the 25-epoch screening, three-seed 45-epoch confirmatory matrix, YOLO26 replication, internal-test evaluation, or external-holdout evaluation in this task.

## Authorised actions

You are authorised to:

- inspect Windows, NVIDIA, disk, Python, Git, PowerShell, and GitHub authentication state;
- clone or safely update the two repositories below;
- create project-local Python virtual environments and install required Python packages;
- restore Git-ignored datasets from the private repository;
- run checksum, dataset, loader, CUDA, and smoke-training checks;
- write training runs and diagnostics inside the public repository's ignored local paths;
- create a small, sanitised smoke-test report under `research/gpu_runs/`;
- create a new Git branch, commit only that small report, push it, and open a draft PR if GitHub write access is available.

You are not authorised to:

- delete or overwrite an existing dataset, run, checkpoint, repository, or user file;
- reset a dirty worktree, force-push, push directly to `main`, or merge a PR;
- request, print, save, or commit GitHub tokens or other credentials;
- commit raw/generated images, labels, virtual environments, downloaded model sources, cache files, full console logs, or model weights;
- modify the frozen ontology decisions, leakage manifests, internal test, or research protocol;
- install or change NVIDIA drivers or other system-wide software without explicit approval from the person at the Windows computer;
- use the semantically mixed legacy checkpoint for these smoke tests.

If an existing local path conflicts with the instructions, preserve it and inspect it. Never solve a conflict by deleting it.

## Fixed research facts

- Primary architecture: pinned YOLOv5s.
- Generic initialisation: `yolov5s.pt`, not `models/legacy/best.pt`.
- Cells:
  - A: clean training images, original mixed training ontology;
  - B: clean training images, audited training ontology;
  - C: controlled leakage, original mixed training ontology;
  - D: controlled leakage, audited training ontology.
- Seeds frozen for later confirmation: 0, 1, and 2.
- Every cell has 1,511 training images.
- All cells share the same audited 188-image validation set and 188-image internal test set.
- Each seed-specific C/D pair has an identical image pool.
- Each C/D seed injects 41 unused offline variants and removes 41 matched training images.
- The leakage endpoint is 41/188 = 21.8% of all internal test components and 100% of strictly eligible components. Never describe it simply as “100% test leakage.”
- The ontology audit contains 498 resolved rows. Reviewer 1 is `yhm`; reviewer 2 is `hzp`; the 100-row overlap achieved 96% agreement and Cohen's kappa 0.903.
- Five-epoch smoke-test metrics are execution evidence only and must not be presented as paper findings.

## Authoritative repositories and refs

Public code repository:

`https://github.com/Llleeeo/Smoke-and-Fire-detection-model-and-implementation-`

Private immutable data repository:

`https://github.com/Llleeeo/Smoke-and-Fire-detection-data-private`

Research PR:

`https://github.com/Llleeeo/Smoke-and-Fire-detection-model-and-implementation-/pull/6`

Research branch:

`agent/publish-factorial-research`

Required ancestor commit:

`df297e6941fd8506bb5a7aae5c8c58d9ee50258f`

The private repository's verified `main` must contain ancestor commit `ebfdcc8`.

If PR #6 has been merged and `origin/main` contains the required ancestor commit, use the current clean `main`. Otherwise use `origin/agent/publish-factorial-research`. Do not use an older default branch that lacks the required commit.

## Communication and blocking rules

Work autonomously through all non-destructive steps. Keep the person at the Windows computer informed with concise progress updates. Ask them only when one of these genuine blockers occurs:

- their GitHub account lacks access to the private repository;
- GitHub authentication is absent and requires an interactive login;
- Python 3.11, Git, GitHub CLI, or the NVIDIA driver is missing and a system-wide installation needs approval;
- CUDA remains unavailable after project-local PyTorch correction;
- free disk space or GPU memory is insufficient;
- an existing non-empty/dirty path would have to be overwritten;
- a frozen checksum or manifest comparison fails.

Do not ask the person to decide routine package commands, paths, batch-size reductions, log parsing, or Git operations that are already specified here.

## Phase A — preflight and evidence capture

Use native 64-bit PowerShell, not WSL, unless the person explicitly requests WSL. Before changing anything, inspect and record:

- Windows edition and build;
- PowerShell version;
- NVIDIA GPU model and dedicated VRAM;
- `nvidia-smi` output, driver version, reported CUDA compatibility, temperature, and current GPU processes;
- free space on the intended drive;
- `git --version`, `gh --version`, `gh auth status`, and `py -0p`;
- whether the GitHub account can read both repositories.

Use `nvidia-smi`, `Get-CimInstance Win32_VideoController`, `Get-ComputerInfo`, `Get-PSDrive`, and `py -0p` as appropriate. Do not expose authentication tokens in output.

Require at least 25 GB free for Phase 1. Prefer 50 GB or more if the machine will later run the formal matrix. If space is insufficient, stop and report the measured value.

Prefer Python 3.11. Do not use Python 3.13 for this pinned YOLOv5 workflow unless you have independently proved all pinned dependencies compatible.

## Phase B — clone or safely reuse both repositories

Place the repositories as sibling directories under a clear work root, for example:

```powershell
$WorkRoot = Join-Path $env:USERPROFILE "fire-factorial"
New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
Set-Location $WorkRoot
gh repo clone "Llleeeo/Smoke-and-Fire-detection-model-and-implementation-" "code"
gh repo clone "Llleeeo/Smoke-and-Fire-detection-data-private" "data"
```

If either directory already exists, do not clone over it. Inspect its remote, branch, `git status -sb`, and latest commit. Reuse it only if it is the correct repository and has no uncommitted changes that would be disturbed.

Fetch the public repository, then select the correct ref:

1. If `origin/main` contains `df297e6941fd8506bb5a7aae5c8c58d9ee50258f`, use a clean, updated `main`.
2. Otherwise switch to a local branch tracking `origin/agent/publish-factorial-research`.
3. Verify with `git merge-base --is-ancestor df297e6941fd8506bb5a7aae5c8c58d9ee50258f HEAD`; a non-zero result is a hard stop.

Update the private repository's clean `main` without force or reset, then verify `ebfdcc8` is an ancestor of `HEAD`. Confirm these paths exist:

- `data\handoff\yolo 5`
- `data\cleaned\leakage_safe_yolov5\dataset`
- `data\manifests\raw-yolo5.sha256`
- `data\manifests\cleaned-dataset.sha256`

## Phase C — read the frozen instructions before executing

Read these files completely:

- `code\WINDOWS_GPU_REPRODUCTION_HANDOFF.md`
- `code\research\EXPERIMENT_PROTOCOL.md`
- `code\research\EXPERIMENT_LOG.md`
- `code\training\leakage_safe_yolov5\experiments\factorial\RECOVER_ON_WINDOWS_GPU.md`
- `code\training\leakage_safe_yolov5\experiments\factorial\matrix.yaml`
- `code\training\leakage_safe_yolov5\audit\ONTOLOGY_AUDIT_FINAL_REPORT.md`
- all six files under `code\training\leakage_safe_yolov5\experiments\factorial\manifests\`.

Treat the protocol, audit decisions, and manifests as frozen inputs. If this handoff and an older document conflict, follow this handoff for Windows execution boundaries and report the conflict.

## Phase D — create the pinned project environment

From the package directory:

```powershell
Set-Location "$WorkRoot\code\training\leakage_safe_yolov5"
Set-ExecutionPolicy -Scope Process Bypass
.\setup_environment.ps1 -PythonVersion 3.11
```

If `.venv` or `yolov5` already exists, the script may correctly refuse to replace it. Inspect rather than delete it. Reuse it only after verifying:

- the virtual environment uses 64-bit Python 3.11;
- YOLOv5 is at commit `3fb11111c6a8088fbc91430a1f99d207c16f0620`;
- required packages import successfully.

Then run:

```powershell
.\.venv\Scripts\python.exe -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
.\.venv\Scripts\python.exe -m pip freeze
```

`torch.cuda.is_available()` must be `True`. If it is `False` while `nvidia-smi` works, inspect the installed Torch build. Correct Torch only inside `.venv`, using the current official instructions from `pytorch.org`. Do not rely on third-party package instructions and do not install the full CUDA Toolkit unless the official package path actually requires it. Re-run the CUDA check afterward.

If CUDA is still unavailable, stop before restoring/training and provide an exact diagnosis.

## Phase E — restore and verify all factorial data

From `code\training\leakage_safe_yolov5`, run:

```powershell
.\restore_factorial_data.ps1 -PrivateRepository "$WorkRoot\data"
```

The script must:

- verify 10,137 raw checksum entries;
- verify 3,779 cleaned-dataset checksum entries;
- restore 1,511 train / 188 validation / 188 test image-label pairs;
- apply all 498 ontology decisions;
- build detection-only original and audited datasets;
- build C/D for seeds 0, 1, and 2;
- compare every regenerated C/D manifest and summary against the frozen Git copies;
- generate machine-local A/B/C/D YAML and matrix status files.

If restoration fails partway, do not delete partial outputs. Diagnose the cause. If a retry genuinely requires a clean target, move the partial generated directories into a timestamped backup directory outside the expected names, record the move, and retry. Never move or alter the private immutable repository inputs.

Independently inspect all three files:

- `experiments\factorial\generated\MATRIX_STATUS_seed0.json`
- `experiments\factorial\generated\MATRIX_STATUS_seed1.json`
- `experiments\factorial\generated\MATRIX_STATUS_seed2.json`

Required gate:

- A/B/C/D all report `ready` for every seed;
- every training count is exactly 1,511 images and 1,511 labels;
- common validation is 188/188;
- common test is 188/188;
- no exact test hash appears in controlled-leakage training;
- C and D have identical image pools within each seed.

Any failure is a hard stop. Do not “fix” a frozen count manually.

## Phase F — choose one safe, common smoke-test configuration

Inspect dedicated VRAM and current GPU load. Select one batch size that all four cells will use. Start conservatively:

- 6–8 GB VRAM: batch 8;
- 10–16 GB VRAM: batch 16;
- more than 16 GB: batch 16 initially unless a short calibration proves a larger value stable.

Use `Device 0`. On Windows, start with `Workers 4`; reduce to 2 or 0 only if the dataloader fails. Keep the final worker count identical across A/B/C/D.

You may run one A-cell one-epoch calibration solely to detect OOM or dataloader failure. It is not evidence. If any A/B/C/D five-epoch run later fails from OOM, move the failed run directory to a timestamped `FAILED` backup, reduce the batch size, and restart all four five-epoch cells with the same reduced batch size. Do not compare cells trained with different batch sizes.

Before training, capture:

- GPU and CUDA environment;
- Git commits of both repositories;
- Python and complete `pip freeze`;
- selected batch size and workers;
- free disk and start time.

## Phase G — run the four YOLOv5s smoke tests sequentially

Use the native PowerShell runner. Substitute the one frozen batch and worker choice consistently:

```powershell
.\experiments\factorial\run_yolov5.ps1 -Cell A -Seed 0 -Epochs 5 -Device 0 -BatchSize 16 -Workers 4
.\experiments\factorial\run_yolov5.ps1 -Cell B -Seed 0 -Epochs 5 -Device 0 -BatchSize 16 -Workers 4
.\experiments\factorial\run_yolov5.ps1 -Cell C -Seed 0 -Epochs 5 -Device 0 -BatchSize 16 -Workers 4
.\experiments\factorial\run_yolov5.ps1 -Cell D -Seed 0 -Epochs 5 -Device 0 -BatchSize 16 -Workers 4
```

Run sequentially, not concurrently. Capture each console stream to a separate local log while preserving the original process exit code. Monitor `nvidia-smi` during the first run and record approximate peak VRAM, temperature, utilisation, and wall time. Keep the computer awake and connected to power.

Expected run directories:

- `runs\factorial_yolov5\A_clean_original_seed0_5e`
- `runs\factorial_yolov5\B_clean_audited_seed0_5e`
- `runs\factorial_yolov5\C_leaked_original_seed0_5e`
- `runs\factorial_yolov5\D_leaked_audited_seed0_5e`

The runner refuses to reuse a run directory. Preserve that protection. Never add `exist_ok`, overwrite a previous result, or silently resume a failed smoke run.

## Phase H — validate results before interpreting anything

The smoke gate passes only if all of the following are true:

- all four processes exit successfully;
- Torch reports CUDA, not CPU;
- each run used generic `yolov5s.pt` initialisation;
- seed, epochs, image size, batch size, workers, device, model, hyperparameters, and augmentation settings match across cells except for the intended data YAML;
- all training loaders report 1,511 images and zero corrupt samples;
- all validation loaders report 188 images and zero corrupt samples;
- no loss or metric is NaN or infinite;
- each run contains `results.csv`, `opt.yaml`, `weights/last.pt`, and `weights/best.pt`;
- the four result directories are distinct and complete.

Parse `results.csv` yourself. For each cell, report the final-epoch and best-observed validation values for:

- precision;
- recall;
- mAP@0.5;
- mAP@0.5:0.95;
- box/object/class losses where available;
- wall time and approximate peak VRAM.

Also compute the descriptive smoke-test contrasts:

- B − A;
- D − C;
- C − A;
- D − B;
- interaction `(D − C) − (B − A)`.

Label all of these as non-confirmatory pipeline diagnostics. Do not claim significance, generalisation, causality, or paper novelty from five epochs and one seed.

## Phase I — create a small, safe GitHub report

Create a timestamped directory such as:

`research\gpu_runs\windows_smoke_YYYYMMDD_HHMM\`

Store only small, sanitised evidence:

- `REPORT.md` with preflight, exact commands, gate results, metrics, contrasts, warnings, and next recommendation;
- `environment.json` with Windows/GPU/driver/CUDA/Torch/Python/Git commits, batch, workers, and timings, but no usernames, tokens, serial numbers, or absolute home paths;
- four renamed copies of `results.csv`;
- optionally small sanitised configuration summaries that contain no absolute paths.

Do not copy or commit model weights, datasets, caches, raw console logs, `.venv`, `yolov5`, or generated absolute-path YAML files.

Before staging, inspect `git status -sb`, file sizes, and the complete report diff. Create a new branch named like `agent/windows-gpu-smoke-YYYYMMDD`. Explicitly stage only the new `research/gpu_runs/...` directory. Commit with a concise message, push the branch, and open a draft PR:

- target `agent/publish-factorial-research` if PR #6 is still unmerged;
- otherwise target `main`.

If GitHub write access is unavailable, do not request a token in chat. Preserve the local report and clearly state the authentication/permission blocker.

## Mandatory stop point

After publishing or preserving the Phase 1 report, stop. Do not automatically proceed to longer training even if all smoke tests pass.

Your final response to the person at the Windows computer must be self-contained and use this structure:

1. `STATUS: PASSED`, `STATUS: BLOCKED`, or `STATUS: FAILED`.
2. Windows, GPU, VRAM, driver, CUDA, Torch, Python, and repository commits.
3. Data/checksum/matrix verification result.
4. The exact common A/B/C/D configuration.
5. A compact four-row metrics and runtime table.
6. The five descriptive contrasts and explicit non-confirmatory warning.
7. Warnings/errors and how they were handled.
8. Local output paths and the draft PR/report URL, if published.
9. One next recommendation: either fix the named blocker or approve a separately specified 25-epoch screening stage.

Keep responsibility for the Windows execution workflow through the mandatory stop point and record any unresolved blocker clearly.
