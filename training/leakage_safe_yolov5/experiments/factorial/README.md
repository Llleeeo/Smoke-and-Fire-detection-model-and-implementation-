# Completed 2x2 YOLOv5s factorial experiment

The four cells cross training ontology with controlled near-duplicate leakage. All cells use the same adjudicated validation and internal test labels.

The factorial loaders use detection-only dataset variants. Historical polygon rows are converted to their tight axis-aligned bounding boxes so pinned YOLOv5 and modern Ultralytics models consume exactly the same image/label pool; class IDs and split membership are unchanged.

For the intended Windows/NVIDIA workstation, follow [`RECOVER_ON_WINDOWS_GPU.md`](RECOVER_ON_WINDOWS_GPU.md). Linux, WSL, and macOS instructions remain in [`RECOVER_ON_NEW_MACHINE.md`](RECOVER_ON_NEW_MACHINE.md). The committed files under `manifests/` freeze the exact injected variant and matched replacement selected for every eligible component in seeds 0, 1, and 2; restoration fails if regenerated choices differ.

| Cell | Leakage | Training ontology | Formal status |
| --- | --- | --- | --- |
| A | clean | original mixed | seeds 0/1/2 completed |
| B | clean | audited | seeds 0/1/2 completed |
| C | controlled | original mixed | seeds 0/1/2 completed |
| D | controlled | audited | seeds 0/1/2 completed |

All 12 formal training runs and all 12 fixed internal-test evaluations were completed. Verified aggregate results and checkpoint identities are retained under `research/final_report_evidence/`.

Regenerate machine-local absolute data YAML files after copying the project to a GPU host:

```bash
.venv/bin/python scripts/prepare_factorial_configs.py --seed 0
```

Inspect `experiments/factorial/generated/MATRIX_STATUS_seed0.json`. Repeat for seeds 1 and 2; all four cells must report `ready` for each seed.

## Reproducing the YOLOv5s runs

After restoring the private data and generating the machine-local configuration files, run one cell and seed as follows:

```bash
DEVICE=0 BATCH_SIZE=8 experiments/factorial/run_yolov5.sh A 0 45
```

Repeat for cells A-D and seeds 0-2. The formal configuration is 45 epochs, 640 px images, batch size 8, SGD, and the pinned hyperparameters in `config/yolov5_hyp.yaml`.
