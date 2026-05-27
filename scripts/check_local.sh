#!/usr/bin/env bash
set -euo pipefail

echo "== git status =="
git status --short

echo
echo "== smoke test data =="
python scripts/smoke_test_data.py --train-csv data/raw/train.csv

echo
echo "== stratified split =="
python scripts/make_stratified_split.py \
  --train data/raw/train.csv \
  --out data/processed

echo
echo "== cli help =="
python scripts/kaggle_train.py --help >/dev/null
python scripts/package_adapter.py --help >/dev/null
echo "CLI OK"

echo
echo "== done =="
