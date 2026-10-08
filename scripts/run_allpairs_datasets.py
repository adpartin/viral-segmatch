"""Build the 28 codon-screen datasets from their child bundles."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_RUNS = PROJECT_ROOT / 'data/datasets/flu/July_2025/runs'


def _build_one(bundle: str, output_dir: Path) -> tuple[str, int]:
    """Build one pair dataset and retain the Stage 3 log.

    Args:
        bundle: Child bundle name.
        output_dir: New dataset run directory.

    Returns:
        Bundle name and process exit code.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(PROJECT_ROOT / 'src/datasets/dataset_segment_pairs.py'),
               '--config_bundle', bundle, '--output_dir', str(output_dir)]
    with (output_dir / 'stage3.log').open('w') as log:
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log,
                                stderr=subprocess.STDOUT)
    return bundle, result.returncode


def main() -> None:
    """Run all 28 Stage 3 builds under one timestamped dataset prefix.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-prefix', required=True,
                        help='Run-directory prefix ending in _, including YYYYMMDD_HHMMSS')
    parser.add_argument('--max-workers', type=int, default=4)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error('--max-workers must be positive')

    bundles = sorted(p.stem for p in (PROJECT_ROOT / 'conf/bundles').glob('flu_28p_codon_*.yaml')
                     if 'balanced' not in p.stem)
    if len(bundles) != 28:
        raise ValueError(f'Expected 28 codon child bundles; found {len(bundles)}')
    jobs = [(bundle, DATASET_RUNS / (args.dataset_prefix + bundle.removeprefix('flu_28p_codon_')))
            for bundle in bundles]
    existing = [str(directory) for _, directory in jobs if directory.exists()]
    if existing:
        raise FileExistsError(f'Dataset outputs already exist: {existing}')

    print(f'Building {len(jobs)} datasets with {args.max_workers} workers.', flush=True)
    results = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_build_one, bundle, directory): bundle
                   for bundle, directory in jobs}
        for future in as_completed(futures):
            bundle, code = future.result()
            results[bundle] = code
            print(f'{"Done." if code == 0 else "ERROR:"} {bundle}: exit {code}', flush=True)

    manifest = {'dataset_prefix': args.dataset_prefix, 'results': results,
                'dataset_directories': {bundle: str(directory) for bundle, directory in jobs}}
    path = DATASET_RUNS / f'{args.dataset_prefix.rstrip("_")}_manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    failed = {bundle: code for bundle, code in results.items() if code != 0}
    if failed:
        raise RuntimeError(f'{len(failed)} dataset builds failed; inspect their stage3.log files')
    print(f'Done. Manifest: {path}', flush=True)


if __name__ == '__main__':
    main()
