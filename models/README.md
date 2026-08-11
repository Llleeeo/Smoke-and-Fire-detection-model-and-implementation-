# Model files

`legacy/best.pt` is the historical checkpoint actually used as the baseline and initialisation for the published 45-epoch clean-split experiment. It was trained before the dataset leakage repair and is retained only for reproducibility.

The later teammate handoff `finetune_D3_stageD92iteration/weights/best.pt` is a different file. Its complete folder is preserved verbatim in the private repository `Llleeeo/Smoke-and-Fire-detection-data-private`; it was not used to produce the published clean-split metrics.

The current validated model is published at `training/leakage_safe_yolov5/runs/full_clean_45e/weights/best.pt`. It was evaluated on the independent clean test split; see `training/leakage_safe_yolov5/EXPERIMENT_RESULTS.md` for metrics and reproducibility details. Other ad-hoc training runs remain ignored by Git.
