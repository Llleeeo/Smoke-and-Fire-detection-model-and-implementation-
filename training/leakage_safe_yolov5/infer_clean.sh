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
  echo "Set WEIGHTS to a restored formal checkpoint before inference." >&2
  exit 2
fi
weights="$WEIGHTS"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 IMAGE_OR_VIDEO [extra detect.py arguments]" >&2
  exit 2
fi

source_path="$1"
shift
"$python_bin" "$package_dir/yolov5/detect.py" \
  --weights "$weights" \
  --source "$source_path" \
  --img 640 \
  --project "$package_dir/runs" \
  --name inference \
  "$@"
