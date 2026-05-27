from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from .prompts import build_training_text


@dataclass(frozen=True)
class FamilyRule:
    name: str
    pattern: str


FAMILY_RULES = [
    FamilyRule("bit", r"8-bit binary|bit manipulation|XOR|AND|OR|NOT|rotation|shift"),
    FamilyRule("roman_or_numeral", r"numeral system|Roman|Wonderland numeral"),
    FamilyRule("text_cipher", r"secret encryption|encrypted text|cipher|plaintext|text"),
    FamilyRule("numeric_or_algebra", r"unit conversion|measurement|equation|algebra|number|numeric"),
    FamilyRule("symbolic", r"symbol|operators|transformation rules.*equations"),
]


def classify_prompt(prompt: str) -> str:
    prompt = str(prompt)
    for rule in FAMILY_RULES:
        if re.search(rule.pattern, prompt, flags=re.IGNORECASE):
            return rule.name
    return "other"


def load_train_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "prompt", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in train csv: {sorted(missing)}")
    df["family"] = df["prompt"].map(classify_prompt)
    return df


def make_hf_dataset(df: pd.DataFrame, tokenizer, use_chat_template: bool = True):
    from datasets import Dataset
    texts = [
        build_training_text(tokenizer, row.prompt, row.answer, use_chat_template=use_chat_template)
        for row in df.itertuples(index=False)
    ]
    return Dataset.from_dict({"text": texts})
