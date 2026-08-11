#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd "$(dirname "$0")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
weights="${WEIGHTS:-$package_dir/../best.pt}"

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
