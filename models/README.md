# Model files

`legacy/best.pt` is the original teammate checkpoint. It was trained before the dataset leakage repair and is retained only as a baseline or initialization checkpoint.

The current validated model is published at `training/leakage_safe_yolov5/runs/full_clean_45e/weights/best.pt`. It was evaluated on the independent clean test split; see `training/leakage_safe_yolov5/EXPERIMENT_RESULTS.md` for metrics and reproducibility details. Other ad-hoc training runs remain ignored by Git.
