#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from nemotron_reasoning.data import load_train_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-csv', default='data/raw/train.csv')
    ap.add_argument('--out-dir', default='data/processed')
    ap.add_argument('--valid-frac', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    df = load_train_csv(args.train_csv)
    valid = df.groupby('family', group_keys=False).sample(frac=args.valid_frac, random_state=args.seed)
    train = df.drop(valid.index)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / 'train_split.csv', index=False)
    valid.to_csv(out / 'valid_split.csv', index=False)
    print(f'train={len(train):,} valid={len(valid):,}')
    print(valid['family'].value_counts().to_string())

if __name__ == '__main__':
    main()
