#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd "$(dirname "$0")" && pwd)"
private_root="${1:?usage: ./restore_factorial_data.sh /path/to/Smoke-and-Fire-detection-data-private}"
private_root="$(cd "$private_root" && pwd)"
raw_dataset="$private_root/handoff/yolo 5"
clean_source="$private_root/cleaned/leakage_safe_yolov5/dataset"
python_bin="${PYTHON_BIN:-$package_dir/.venv/bin/python}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python environment missing: $python_bin" >&2
  echo "Run ./setup_environment.sh first, then retry." >&2
  exit 2
fi
if [[ ! -d "$raw_dataset" || ! -d "$clean_source" ]]; then
  echo "Private repository is missing the raw or cleaned dataset." >&2
  exit 3
fi

verify_manifest() {
  local manifest="$1"
  if command -v shasum >/dev/null 2>&1; then
    (cd "$private_root" && shasum -a 256 -c "$manifest" | tail -n 1)
  elif command -v sha256sum >/dev/null 2>&1; then
    (cd "$private_root" && sha256sum -c "$manifest" | tail -n 1)
  else
    echo "Neither shasum nor sha256sum is available." >&2
    exit 4
  fi
}

echo "Verifying immutable private data..."
verify_manifest manifests/raw-yolo5.sha256
verify_manifest manifests/cleaned-dataset.sha256

for split in train valid test; do
  if [[ -e "$package_dir/dataset/$split" ]]; then
    echo "Refusing to overwrite existing dataset split: $package_dir/dataset/$split" >&2
    exit 5
  fi
done
for output in \
  dataset_audited \
  dataset_detect_original \
  dataset_detect_audited \
  dataset_leaked_original_seed0 dataset_leaked_audited_seed0 \
  dataset_leaked_original_seed1 dataset_leaked_audited_seed1 \
  dataset_leaked_original_seed2 dataset_leaked_audited_seed2; do
  if [[ -e "$package_dir/$output" ]]; then
    echo "Refusing to overwrite existing generated dataset: $package_dir/$output" >&2
    exit 6
  fi
done

echo "Restoring the verified 1,511/188/188 clean split..."
for split in train valid test; do
  cp -a "$clean_source/$split" "$package_dir/dataset/$split"
done

echo "Applying the frozen 498-row ontology audit..."
"$python_bin" "$package_dir/scripts/build_audited_dataset.py" \
  "$package_dir/dataset" \
  "$package_dir/audit/ontology_review_final.csv" \
  "$package_dir/dataset_audited"

echo "Normalising original and audited labels to detection boxes..."
"$python_bin" "$package_dir/scripts/build_detect_only_dataset.py" \
  "$package_dir/dataset" "$package_dir/dataset_detect_original"
"$python_bin" "$package_dir/scripts/build_detect_only_dataset.py" \
  "$package_dir/dataset_audited" "$package_dir/dataset_detect_audited"

echo "Building controlled-leakage C/D datasets for seeds 0, 1, and 2..."
"$python_bin" "$package_dir/scripts/build_controlled_leakage_datasets.py" \
  "$raw_dataset" "$package_dir" --seeds 0 1 2

for seed in 0 1 2; do
  expected="$package_dir/experiments/factorial/manifests"
  cmp \
    "$package_dir/dataset_leaked_original_seed${seed}/CONTROLLED_LEAKAGE_MANIFEST.csv" \
    "$expected/CONTROLLED_LEAKAGE_MANIFEST_seed${seed}.csv"
  cmp \
    "$package_dir/dataset_leaked_audited_seed${seed}/CONTROLLED_LEAKAGE_MANIFEST.csv" \
    "$expected/CONTROLLED_LEAKAGE_MANIFEST_seed${seed}.csv"
  cmp \
    "$package_dir/dataset_leaked_original_seed${seed}/CONTROLLED_LEAKAGE_SUMMARY.json" \
    "$expected/CONTROLLED_LEAKAGE_SUMMARY_seed${seed}.json"
  cmp \
    "$package_dir/dataset_leaked_audited_seed${seed}/CONTROLLED_LEAKAGE_SUMMARY.json" \
    "$expected/CONTROLLED_LEAKAGE_SUMMARY_seed${seed}.json"
  "$python_bin" "$package_dir/scripts/prepare_factorial_configs.py" --seed "$seed"
done

echo "Factorial data restoration complete."
echo "Inspect experiments/factorial/generated/MATRIX_STATUS_seed{0,1,2}.json."
