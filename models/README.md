# Model files

`legacy/best.pt` is the original teammate checkpoint. It was trained before the dataset leakage repair and is retained only as a baseline or initialization checkpoint.

New training outputs are written below `training/leakage_safe_yolov5/runs/` and are intentionally ignored by Git. Promote a newly validated model here only after evaluating it on the clean test split and documenting its metrics.
