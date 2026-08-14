# Primary 2x2 factorial experiment

The four cells cross training ontology with controlled near-duplicate leakage. All cells use the same adjudicated validation and internal test labels.

The factorial loaders use detection-only dataset variants. Historical polygon rows are converted to their tight axis-aligned bounding boxes so pinned YOLOv5 and modern Ultralytics models consume exactly the same image/label pool; class IDs and split membership are unchanged.

For the intended Windows/NVIDIA workstation, follow [`RECOVER_ON_WINDOWS_GPU.md`](RECOVER_ON_WINDOWS_GPU.md). Linux, WSL, and macOS instructions remain in [`RECOVER_ON_NEW_MACHINE.md`](RECOVER_ON_NEW_MACHINE.md). The committed files under `manifests/` freeze the exact injected variant and matched replacement selected for every eligible component in seeds 0, 1, and 2; restoration fails if regenerated choices differ.

| Cell | Leakage | Training ontology | Status |
| --- | --- | --- | --- |
| A | clean | original mixed | ready |
| B | clean | audited | ready |
| C | controlled | original mixed | ready for seeds 0/1/2 |
| D | controlled | audited | ready for seeds 0/1/2 |

Regenerate machine-local absolute data YAML files after copying the project to a GPU host:

```bash
.venv/bin/python scripts/prepare_factorial_configs.py --seed 0
```

Inspect `experiments/factorial/generated/MATRIX_STATUS_seed0.json`. Repeat for seeds 1 and 2; all four cells must report `ready` for each seed.

## YOLOv5s primary architecture

One-seed five-epoch smoke test:

```bash
DEVICE=0 BATCH_SIZE=16 experiments/factorial/run_yolov5.sh A 0 5
DEVICE=0 BATCH_SIZE=16 experiments/factorial/run_yolov5.sh B 0 5
DEVICE=0 BATCH_SIZE=16 experiments/factorial/run_yolov5.sh C 0 5
DEVICE=0 BATCH_SIZE=16 experiments/factorial/run_yolov5.sh D 0 5
```

Inspect all four smoke-test runs before starting confirmatory training. Confirmatory runs use seeds 0, 1, and 2 with 45 epochs only after the screening design is frozen.

## YOLO26n replication architecture

Create a separate environment on the GPU host and install the pinned package:

```bash
python3 -m venv .venv_yolo26
.venv_yolo26/bin/pip install --upgrade pip
.venv_yolo26/bin/pip install -r experiments/factorial/requirements_yolo26.txt
.venv_yolo26/bin/python experiments/factorial/run_yolo26.py A 0 5 --device 0
```

YOLOv5s and YOLO26n are analysed separately. The purpose of YOLO26n is to test whether the direction of the factorial effects survives a modern architecture, not to compare raw AP values across model families.
