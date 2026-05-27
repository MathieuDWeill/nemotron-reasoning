#!/usr/bin/env bash
set -euo pipefail

COMP_DATA="/kaggle/input/nvidia-nemotron-model-reasoning-challenge"
WORK_DATA="/kaggle/working/data/processed"

echo "== install repo =="
pip install -e .

echo
echo "== smoke test Kaggle data =="
python scripts/smoke_test_data.py \
  --train-csv "${COMP_DATA}/train.csv"

echo
echo "== make Kaggle split =="
python scripts/make_stratified_split.py \
  --train "${COMP_DATA}/train.csv" \
  --out "${WORK_DATA}"

echo
echo "== done =="
