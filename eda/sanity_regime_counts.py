"""Sanity-check the closed-form regime counter on real HA/NA + PB2/PB1.

Two checks:
  1. Sum of regime counts == N*(N-1) where N = num isolates.
  2. Print the regime distribution as fractions, with binned vs exact year.
"""
import sys
from pathlib import Path
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.datasets._negative_regime_sampling import (
    REGIME_NAMES,
    build_isolate_cells,
    count_isolates_per_cell,
    count_available_per_regime,
)

DATASETS = [
    ('HA/NA',   PROJ / 'data/datasets/flu/July_2025/runs/dataset_flu_ha_na_20260508_171512'),
    ('PB2/PB1', PROJ / 'data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_20260509_114318'),
]


def report(label, ds_dir, year_match):
    iso_meta = pd.read_csv(ds_dir / 'isolate_metadata.csv', engine='python')
    iso_meta['assembly_id'] = iso_meta['assembly_id'].astype(str)

    cells = build_isolate_cells(iso_meta, year_match=year_match)
    cell_counts = count_isolates_per_cell(cells)

    n = len(cells)
    expected = n * (n - 1)

    avail = count_available_per_regime(cell_counts)
    total = sum(avail.values())

    print(f'\n--- {label}, year_match={year_match} ---')
    print(f'  isolates: {n:,}   distinct cells: {len(cell_counts):,}')
    print(f'  total ordered pairs: {total:,}   expected: {expected:,}   '
          f'{"OK" if total == expected else "MISMATCH"}')

    print(f'  per-regime distribution (% of total ordered pairs):')
    for r in REGIME_NAMES:
        pct = (100.0 * avail[r] / max(total, 1))
        print(f'    {r:<25}  {avail[r]:>14,}  ({pct:>6.2f}%)')


def main():
    for label, ds_dir in DATASETS:
        if not ds_dir.exists():
            continue
        report(label, ds_dir, 'binned')
        report(label, ds_dir, 'exact')


if __name__ == '__main__':
    main()
