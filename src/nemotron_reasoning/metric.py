from __future__ import annotations

import math
import re

_BOX_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def extract_answer(text: str) -> str:
    """Small local approximation of the competition answer extraction."""
    text = str(text).strip()
    boxes = _BOX_RE.findall(text)
    if boxes:
        return boxes[-1].strip()
    nums = _NUM_RE.findall(text)
    if nums:
        return nums[-1].strip()
    return text.splitlines()[-1].strip() if text else ""


def is_correct(prediction: str, target: str, rel_tol: float = 1e-4) -> bool:
    pred = extract_answer(prediction)
    target = str(target).strip()
    if pred == target:
        return True
    try:
        p = float(pred)
        t = float(target)
        return math.isclose(p, t, rel_tol=rel_tol, abs_tol=rel_tol)
    except Exception:
        return False
