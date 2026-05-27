#!/usr/bin/env python
from pathlib import Path
import argparse

from nemotron_reasoning.data import load_train_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-csv', default='data/raw/train.csv')
    args = ap.parse_args()
    df = load_train_csv(args.train_csv)
    print(f'rows={len(df):,}')
    print(df['family'].value_counts().to_string())
    print('\nExample:')
    row = df.iloc[0]
    print(row['prompt'][:700])
    print('answer=', row['answer'])

if __name__ == '__main__':
    main()
