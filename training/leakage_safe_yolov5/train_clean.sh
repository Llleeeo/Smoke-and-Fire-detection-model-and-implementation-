#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x "$package_dir/.venv/bin/python" ]]; then
  python_bin="$package_dir/.venv/bin/python"
else
  python_bin="python3"
fi
default_weights="$package_dir/../../models/legacy/best.pt"

if [[ -f "$default_weights" ]]; then
  weights="${INITIAL_WEIGHTS:-$default_weights}"
else
  weights="${INITIAL_WEIGHTS:-yolov5s.pt}"
fi

"$python_bin" "$package_dir/yolov5/train.py" \
  --weights "$weights" \
  --data "$package_dir/dataset/data.yaml" \
  --hyp "$package_dir/original_model_reference/hyp.yaml" \
  --img 640 \
  --batch 16 \
  --epochs 45 \
  --seed 0 \
  --project "$package_dir/runs" \
  --name retrain_clean \
  "$@"
