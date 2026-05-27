#!/usr/bin/env bash
set -euo pipefail

KERNEL_DIR=".kaggle_kernel"
KERNEL_SLUG="nemotron-reasoning-run"
COMP="nvidia-nemotron-model-reasoning-challenge"

rm -rf "${KERNEL_DIR}"
mkdir -p "${KERNEL_DIR}"

echo "== copy repo into Kaggle kernel bundle =="
rsync -a ./ "${KERNEL_DIR}/" \
  --exclude ".git/" \
  --exclude ".kaggle_kernel/" \
  --exclude "kaggle_outputs/" \
  --exclude "data/raw/*.csv" \
  --exclude "data/processed/*.csv" \
  --exclude "__pycache__/" \
  --exclude "*.egg-info/" \
  --exclude ".DS_Store"

cat > "${KERNEL_DIR}/kernel-metadata.json" <<EOF
{
  "id": "mathieuw/${KERNEL_SLUG}",
  "title": "nemotron-reasoning-run",
  "code_file": "run.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": ["${COMP}"],
  "kernel_sources": []
}
EOF

python - <<'PY'
import json
from pathlib import Path

bash = r'''%%bash
set -euo pipefail

cd /kaggle/working

echo "== current files =="
pwd
ls -lah
ls -lah scripts || true

echo "== install local repo =="
pip install -e .

echo "== setup data =="
bash scripts/kaggle_setup.sh

echo "== train =="
python scripts/kaggle_train.py --config configs/sft_default.yaml

echo "== package =="
python scripts/package_adapter.py \
  --adapter-dir /kaggle/working/outputs/adapter \
  --out /kaggle/working/submission.zip

echo "== outputs =="
ls -lah /kaggle/working
unzip -l /kaggle/working/submission.zip | head -50
'''

nb = {
  "cells": [
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": bash.splitlines(True)
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}

Path(".kaggle_kernel/run.ipynb").write_text(json.dumps(nb, indent=2))
PY

kaggle kernels push -p "${KERNEL_DIR}"
