#!/usr/bin/env python
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def package_adapter(adapter_dir: Path, out: Path) -> None:
    if not adapter_dir.exists():
        raise FileNotFoundError(f'Adapter directory not found: {adapter_dir}')
    if not (adapter_dir / 'adapter_config.json').exists():
        raise FileNotFoundError(f'Missing adapter_config.json in {adapter_dir}')

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(adapter_dir.rglob('*')):
            if file.is_file():
                zf.write(file, arcname=file.relative_to(adapter_dir))

    with zipfile.ZipFile(out, 'r') as zf:
        names = zf.namelist()
    if 'adapter_config.json' not in names:
        raise RuntimeError('adapter_config.json is not at zip root')
    print(f'Created {out} with {len(names)} files')
    print('\n'.join(names[:50]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter-dir', required=True)
    ap.add_argument('--out', default='outputs/submissions/submission.zip')
    args = ap.parse_args()
    package_adapter(Path(args.adapter_dir), Path(args.out))

if __name__ == '__main__':
    main()
