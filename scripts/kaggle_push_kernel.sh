#!/usr/bin/env bash
set -euo pipefail

KERNEL_DIR=".kaggle_kernel"
KERNEL_SLUG="nemotron-reasoning-run"
COMP="nvidia-nemotron-model-reasoning-challenge"

rm -rf "${KERNEL_DIR}"
mkdir -p "${KERNEL_DIR}"

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
  "kernel_sources": ["ryanholbrook/nvidia-utility-script"],
  "model_sources": [
    "metric/nemotron-3-nano-30b-a3b-bf16/Transformers/default/1"
  ]
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

echo "== model input tree =="
find /kaggle/input -maxdepth 10 -type f | sort | sed -n '1,400p'
echo "== config candidates =="
find /kaggle/input -type f -name config.json -print

echo "== NVIDIA utility imports =="
mkdir -p /kaggle/working/shims
if [ -d /kaggle/usr/lib/nvidia-utility-script/cutlass_cppgen ]; then
  cp -r /kaggle/usr/lib/nvidia-utility-script/cutlass_cppgen /kaggle/working/shims/cutlass
elif [ -d /kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/cutlass_cppgen ]; then
  cp -r /kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/cutlass_cppgen /kaggle/working/shims/cutlass
else
  echo "ERROR: cutlass_cppgen not found"
  find /kaggle/usr/lib -maxdepth 4 -type d -name 'cutlass_cppgen' -print
  exit 1
fi

export PYTHONPATH="/kaggle/working/shims:/kaggle/usr/lib/nvidia-utility-script:/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script:${PYTHONPATH:-}"

python - <<'PY2'
import sys
print("PYTHONPATH head:")
print("\n".join(sys.path[:10]))

import cutlass
print("cutlass shim OK", getattr(cutlass, "__file__", cutlass))

import mamba_ssm
print("mamba_ssm OK", mamba_ssm.__file__)
PY2

python scripts/kaggle_train.py --config configs/sft_default.yaml

python scripts/package_adapter.py \
  --adapter-dir /kaggle/working/outputs/adapter \
  --out /kaggle/working/submission.zip

ls -lah /kaggle/working
unzip -l /kaggle/working/submission.zip | head -50
'''

nb = {
  "cells": [{
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": bash.splitlines(True)
  }],
  "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}
  },
  "nbformat": 4,
  "nbformat_minor": 5
}

Path(".kaggle_kernel/run.ipynb").write_text(json.dumps(nb, indent=2))
PY

kaggle kernels push -p "${KERNEL_DIR}"
