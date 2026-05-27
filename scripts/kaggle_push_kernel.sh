#!/usr/bin/env bash
set -euo pipefail

KERNEL_DIR=".kaggle_kernel"
KERNEL_SLUG="nemotron-reasoning-run-$(date +%Y%m%d-%H%M%S)"
COMP="nvidia-nemotron-model-reasoning-challenge"

rm -rf "${KERNEL_DIR}"
mkdir -p "${KERNEL_DIR}"

cat > "${KERNEL_DIR}/kernel-metadata.json" <<EOF
{
  "id": "MathieuDWeill/${KERNEL_SLUG}",
  "title": "Nemotron Reasoning Run",
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
rm -rf nemotron-reasoning

git clone https://github.com/MathieuDWeill/nemotron-reasoning.git
cd nemotron-reasoning

bash scripts/kaggle_setup.sh

python scripts/kaggle_train.py --config configs/sft_default.yaml

python scripts/package_adapter.py \
  --adapter-dir /kaggle/working/outputs/adapter \
  --out /kaggle/working/submission.zip

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
