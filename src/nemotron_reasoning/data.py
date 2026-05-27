from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_TRAIN_COLUMNS = {"id", "prompt", "answer"}


def classify_family(prompt: str) -> str:
    """Heuristic prompt family classifier for local validation splits."""
    p = str(prompt).lower()

    if "secret bit manipulation rule" in p or "8-bit binary" in p:
        return "bit"

    if (
        "numeral system" in p
        or "roman numeral" in p
        or "wonderland numeral" in p
        or "numerals" in p
    ):
        return "numeral"

    if (
        "secret encryption rules are used on text" in p
        or "encrypted text" in p
        or "encryption rules" in p
        or "cipher" in p
    ):
        return "text_cipher"

    if (
        "unit conversion" in p
        or "measurement" in p
        or "convert" in p
        or "units" in p
    ):
        return "numeric"

    if (
        "transformation rules is applied to equations" in p
        or "transformation rules are applied to equations" in p
        or "equation" in p
    ):
        return "symbolic"

    if "input -> output" in p and any(tok in p for tok in ["+", "-", "*", "/", "="]):
        return "numeric_or_symbolic"

    return "other"


def load_train_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    missing = REQUIRED_TRAIN_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required train columns: {sorted(missing)}")
    df = df.copy()
    df["family"] = df["prompt"].map(classify_family)
    return df


def load_test_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    required = {"id", "prompt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required test columns: {sorted(missing)}")
    df = df.copy()
    df["family"] = df["prompt"].map(classify_family)
    return df


def make_hf_dataset(df, prompt_builder=None, **kwargs):
    """Build a Hugging Face Dataset for SFT.

    Keeps the datasets import lazy so local CLI/help checks do not require
    the full Kaggle training stack.
    """
    from datasets import Dataset

    records = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        answer = str(row["answer"])

        if prompt_builder is not None:
            text = prompt_builder(prompt, answer)
        else:
            text = (
                f"{prompt}\n\n"
                "We need solve the puzzle and provide the final answer in LaTeX boxed format.\n"
                f"Final answer: \\boxed{{{answer}}}"
            )

        records.append(
            {
                "id": row.get("id"),
                "prompt": prompt,
                "answer": answer,
                "family": row.get("family", "unknown"),
                "text": text,
            }
        )

    return Dataset.from_list(records)
