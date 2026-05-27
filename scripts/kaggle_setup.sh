#!/usr/bin/env bash
set -euo pipefail

echo "== locate Kaggle data =="
find /kaggle/input -maxdepth 4 -type f | sort | sed -n '1,200p'

TRAIN_CSV="$(find /kaggle/input -maxdepth 4 -type f -name train.csv | head -1)"
TEST_CSV="$(find /kaggle/input -maxdepth 4 -type f -name test.csv | head -1 || true)"

if [[ -z "${TRAIN_CSV}" ]]; then
  echo "ERROR: train.csv not found under /kaggle/input"
  exit 1
fi

echo "TRAIN_CSV=${TRAIN_CSV}"
echo "TEST_CSV=${TEST_CSV:-<not found>}"

WORK_DATA="/kaggle/working/data/processed"

echo
echo "== install repo =="
pip install -e .

echo
echo "== smoke test Kaggle data =="
python scripts/smoke_test_data.py --train-csv "${TRAIN_CSV}"

echo
echo "== make Kaggle split =="
python scripts/make_stratified_split.py \
  --train "${TRAIN_CSV}" \
  --out "${WORK_DATA}"

echo
echo "== done =="
