#!/usr/bin/env bash
set -euo pipefail

factorial_dir="$(cd "$(dirname "$0")" && pwd)"
package_dir="$(cd "$factorial_dir/../.." && pwd)"
cell="${1:?usage: run_yolov5.sh CELL SEED EPOCHS [extra train.py args]}"
seed="${2:?usage: run_yolov5.sh CELL SEED EPOCHS [extra train.py args]}"
epochs="${3:?usage: run_yolov5.sh CELL SEED EPOCHS [extra train.py args]}"
shift 3

case "$cell" in
  A) cell_name="clean_original" ;;
  B) cell_name="clean_audited" ;;
  C) cell_name="leaked_original" ;;
  D) cell_name="leaked_audited" ;;
  *) echo "CELL must be A, B, C, or D" >&2; exit 2 ;;
esac

python_bin="${PYTHON_BIN:-$package_dir/.venv/bin/python}"
"$python_bin" "$package_dir/scripts/prepare_factorial_configs.py" --seed "$seed" >/dev/null
seed_suffix=""
if [[ "$cell" == "C" || "$cell" == "D" ]]; then
  seed_suffix="_seed${seed}"
fi
data_yaml="$factorial_dir/generated/${cell}_${cell_name}${seed_suffix}.yaml"
if [[ ! -f "$data_yaml" ]]; then
  echo "Cell $cell seed $seed is not ready. See $factorial_dir/generated/MATRIX_STATUS_seed${seed}.json" >&2
  exit 3
fi

project_dir="$package_dir/runs/factorial_yolov5"
run_name="${cell}_${cell_name}_seed${seed}_${epochs}e"
if [[ -e "$project_dir/$run_name" ]]; then
  echo "Refusing to reuse existing run directory: $project_dir/$run_name" >&2
  exit 4
fi

command=(
  "$python_bin" "$package_dir/yolov5/train.py"
  --weights "$package_dir/yolov5s.pt"
  --data "$data_yaml"
  --hyp "$package_dir/config/yolov5_hyp.yaml"
  --img 640
  --batch-size "${BATCH_SIZE:-16}"
  --epochs "$epochs"
  --seed "$seed"
  --device "${DEVICE:-0}"
  --workers "${WORKERS:-8}"
  --project "$project_dir"
  --name "$run_name"
)
command+=("$@")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
else
  "${command[@]}"
fi
