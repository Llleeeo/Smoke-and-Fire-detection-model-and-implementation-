# Legacy reference code

These files are preserved to document the original project state; they are not the recommended training path.

- `Implementation.py`: original real-time camera inference prototype, updated only to use the reorganized paths.
- `smoke_detection.py`: sanitized Roboflow download helper derived from the old Colab export. It reads its credential from `ROBOFLOW_API_KEY`.
- `train.py`: copied YOLOv5 training snapshot. It depends on a matching YOLOv5 source tree and is not standalone.

Use `../training/leakage_safe_yolov5/` for current setup, training, evaluation, and inference.
