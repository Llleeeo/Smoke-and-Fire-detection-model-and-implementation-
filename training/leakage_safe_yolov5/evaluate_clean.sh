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
if [[ -z "${WEIGHTS:-}" ]]; then
  echo "Set WEIGHTS to a restored formal checkpoint before evaluation." >&2
  exit 2
fi
weights="$WEIGHTS"

"$python_bin" "$package_dir/yolov5/val.py" \
  --weights "$weights" \
  --data "$package_dir/dataset/data.yaml" \
  --img 640 \
  --task test \
  --project "$package_dir/runs" \
  --name clean_test \
  "$@"
