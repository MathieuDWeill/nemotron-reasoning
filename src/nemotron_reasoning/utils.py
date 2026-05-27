from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def patch_nemotron_triton() -> None:
    """Kaggle/Blackwell workaround inspired by public notebooks.

    It avoids known Triton RMSNorm / ptxas issues by forcing safe Python paths where possible.
    """
    import sys
    import torch.nn.functional as F

    def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5,
                         group_size=None, norm_before_gate=True, upcast=True):
        dtype = x.dtype
        if upcast:
            x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        out = x_normed * weight.float()
        if bias is not None:
            out = out + bias.float()
        if z is not None:
            out = out * F.silu(z.float())
        return out.to(dtype)

    for _, mod in list(sys.modules.items()):
        if hasattr(mod, "rmsnorm_fn"):
            try:
                mod.rmsnorm_fn = _pure_rmsnorm_fn
            except Exception:
                pass
        if hasattr(mod, "is_fast_path_available"):
            try:
                mod.is_fast_path_available = False
            except Exception:
                pass

    os.environ.setdefault("TRITON_PTXAS_BLACKWELL_PATH", "/tmp/ptxas-blackwell")
