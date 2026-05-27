#!/usr/bin/env bash
set -euo pipefail

KERNEL_DIR=".kaggle_probe_kernel"
KERNEL_SLUG="nemotron-reasoning-probe"
COMP="nvidia-nemotron-model-reasoning-challenge"

rm -rf "${KERNEL_DIR}"
mkdir -p "${KERNEL_DIR}"

cat > "${KERNEL_DIR}/kernel-metadata.json" <<EOF
{
  "id": "mathieuw/${KERNEL_SLUG}",
  "title": "nemotron-reasoning-probe",
  "code_file": "probe.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": ["${COMP}"],
  "kernel_sources": [],
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

echo "== /kaggle/input tree =="
find /kaggle/input -maxdepth 5 -type f | sort | sed -n '1,300p'

echo
echo "== directories =="
find /kaggle/input -maxdepth 4 -type d | sort | sed -n '1,300p'
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

Path(".kaggle_probe_kernel/probe.ipynb").write_text(json.dumps(nb, indent=2))
PY

kaggle kernels push -p "${KERNEL_DIR}"
