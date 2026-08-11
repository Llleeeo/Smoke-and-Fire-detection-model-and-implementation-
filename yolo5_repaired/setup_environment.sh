#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd "$(dirname "$0")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
yolov5_commit="3fb11111c6a8088fbc91430a1f99d207c16f0620"

if [[ ! -f "$package_dir/yolov5/train.py" ]]; then
  git clone --filter=blob:none https://github.com/ultralytics/yolov5.git "$package_dir/yolov5"
  git -C "$package_dir/yolov5" checkout "$yolov5_commit"
elif [[ -d "$package_dir/yolov5/.git" ]]; then
  current_commit="$(git -C "$package_dir/yolov5" rev-parse HEAD)"
  if [[ "$current_commit" != "$yolov5_commit" ]]; then
    echo "Warning: yolov5 is at $current_commit; expected $yolov5_commit" >&2
  fi
fi

"$python_bin" -m venv "$package_dir/.venv"
"$package_dir/.venv/bin/python" -m pip install --upgrade pip
"$package_dir/.venv/bin/python" -m pip install -r "$package_dir/yolov5/requirements.txt"
"$package_dir/.venv/bin/python" -m pip install -r "$package_dir/requirements-audit.txt"

echo "Environment ready. Run: source '$package_dir/.venv/bin/activate'"
