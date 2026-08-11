#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd "$(dirname "$0")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
weights="${WEIGHTS:-$package_dir/../best.pt}"

"$python_bin" "$package_dir/yolov5/val.py" \
  --weights "$weights" \
  --data "$package_dir/dataset/data.yaml" \
  --img 640 \
  --task test \
  --project "$package_dir/runs" \
  --name clean_test \
  "$@"
