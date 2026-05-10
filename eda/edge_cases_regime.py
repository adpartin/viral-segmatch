"""Edge case tests for the regime-aware sampler:

  1. Determinism: re-run with same seed -> identical pairs.
  2. Single-cell bundle (Human-H3N2-2024 only): only host_subtype_year is feasible.
  3. on_shortfall='error': raises with the per-regime gap.
  4. Validation errors (missing regimes, bad sum, negatives).
  5. Run on PB2/PB1 subset to confirm general behavior across schemas.
"""
import sys
from pathlib import Path
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.datasets.dataset_segment_pairs_v2 import create_negative_pairs_v2
from src.datasets._negative_regime_sampling import (
    REGIME_NAMES, build_isolate_cells,
)
from e2e_regime import load_subset  # sibling file in eda/


def _full_target():
    return {
        'none_match':           0.20,
        'host_only':            0.10,
        'subtype_only':         0.10,
        'year_only':            0.10,
        'host_subtype_only':    0.10,
        'host_year_only':       0.05,
        'subtype_year_only':    0.05,
        'host_subtype_year':    0.30,
        'unknown_metadata_neg': 0.00,
    }


def test_determinism():
    print('\n--- Determinism ---')
    pos_df, iso_meta = load_subset(n_isolates=300)
    isolate_to_cell = build_isolate_cells(iso_meta, year_match='binned')
    cooccur = set(pos_df['pair_key'])

    common = dict(
        pos_df=pos_df,
        num_negatives=1500,
        cooccur_pairs=cooccur,
        schema_pair=('Hemagglutinin precursor', 'Neuraminidase protein'),
        seed=42,
        axis_quotas=_full_target(),
        isolate_to_cell=isolate_to_cell,
        sampling_axes=['host', 'hn_subtype', 'year'],
        on_shortfall='redistribute',
    )
    neg1, _ = create_negative_pairs_v2(**common)
    neg2, _ = create_negative_pairs_v2(**common)
    same = neg1.equals(neg2)
    print(f'  Identical neg_df across two runs: {same}')
    assert same


def test_single_cell_bundle():
    print('\n--- Single-cell bundle (force one-cell pool) ---')
    pos_df, iso_meta = load_subset(n_isolates=2000)
    # Filter iso_meta to one cell that exists in real HA/NA. Pick top-frequent.
    cells_meta = iso_meta.copy()
    cells_meta['cell'] = list(zip(cells_meta['host'], cells_meta['hn_subtype'],
                                   cells_meta['year']))
    top_cell = cells_meta['cell'].value_counts().index[0]
    iso_meta_filt = cells_meta[cells_meta['cell'] == top_cell].drop(columns=['cell'])
    print(f'  one-cell isolates: {len(iso_meta_filt)}')
    pos_df_filt = pos_df[pos_df['assembly_id_a'].astype(str).isin(iso_meta_filt['assembly_id'].astype(str))]
    print(f'  filtered positives: {len(pos_df_filt)}')

    isolate_to_cell = build_isolate_cells(iso_meta_filt, year_match='binned')

    neg, stats = create_negative_pairs_v2(
        pos_df=pos_df_filt,
        num_negatives=len(pos_df_filt),
        cooccur_pairs=set(pos_df_filt['pair_key']),
        schema_pair=('Hemagglutinin precursor', 'Neuraminidase protein'),
        seed=42,
        axis_quotas=_full_target(),
        isolate_to_cell=isolate_to_cell,
        sampling_axes=['host', 'hn_subtype', 'year'],
        on_shortfall='redistribute',
    )
    counts = neg['neg_regime'].value_counts(dropna=False)
    print('  achieved regime distribution:')
    print(counts.to_string())
    non_hsy = counts.drop(index='host_subtype_year', errors='ignore').sum()
    print(f'  non-host_subtype_year achievements: {non_hsy} (should be 0)')
    assert non_hsy == 0, f'expected only host_subtype_year, got {counts.to_dict()}'


def test_on_shortfall_error():
    print('\n--- on_shortfall=error raises ---')
    pos_df, iso_meta = load_subset(n_isolates=200)
    isolate_to_cell = build_isolate_cells(iso_meta, year_match='binned')
    target = _full_target()
    target['host_subtype_year'] = 0.99  # impossible to hit on this scale
    target = {**target, 'unknown_metadata_neg': 0.01}
    target['none_match'] = 0.0
    target['host_only'] = 0.0
    target['subtype_only'] = 0.0
    target['year_only'] = 0.0
    target['host_subtype_only'] = 0.0
    target['host_year_only'] = 0.0
    target['subtype_year_only'] = 0.0
    try:
        neg, _ = create_negative_pairs_v2(
            pos_df=pos_df, num_negatives=10000, cooccur_pairs=set(pos_df['pair_key']),
            schema_pair=('Hemagglutinin precursor', 'Neuraminidase protein'),
            seed=42, axis_quotas=target, isolate_to_cell=isolate_to_cell,
            sampling_axes=['host', 'hn_subtype', 'year'], on_shortfall='error',
        )
        raise AssertionError('expected RuntimeError')
    except RuntimeError as e:
        print(f'  got expected RuntimeError: {str(e)[:120]}...')


def test_validation_missing_regime():
    print('\n--- Missing regime raises ---')
    pos_df, iso_meta = load_subset(n_isolates=100)
    isolate_to_cell = build_isolate_cells(iso_meta, year_match='binned')
    bad_target = _full_target().copy()
    del bad_target['host_only']
    try:
        create_negative_pairs_v2(
            pos_df=pos_df, num_negatives=100, cooccur_pairs=set(),
            schema_pair=('Hemagglutinin precursor', 'Neuraminidase protein'),
            seed=42, axis_quotas=bad_target, isolate_to_cell=isolate_to_cell,
            sampling_axes=['host', 'hn_subtype', 'year'],
        )
        raise AssertionError('expected ValueError')
    except ValueError as e:
        print(f'  got expected ValueError: {str(e)[:100]}...')


def test_bad_sum():
    print('\n--- Targets summing != 1.0 raises ---')
    pos_df, iso_meta = load_subset(n_isolates=100)
    isolate_to_cell = build_isolate_cells(iso_meta, year_match='binned')
    bad_target = {r: 0.05 for r in REGIME_NAMES}  # sums to 0.45
    try:
        create_negative_pairs_v2(
            pos_df=pos_df, num_negatives=100, cooccur_pairs=set(),
            schema_pair=('Hemagglutinin precursor', 'Neuraminidase protein'),
            seed=42, axis_quotas=bad_target, isolate_to_cell=isolate_to_cell,
            sampling_axes=['host', 'hn_subtype', 'year'],
        )
        raise AssertionError('expected ValueError')
    except ValueError as e:
        print(f'  got expected ValueError: {str(e)[:100]}...')


def test_pb2_pb1():
    print('\n--- PB2/PB1 schema sanity ---')
    ds = PROJ / 'data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_20260509_114318'
    pos_df, iso_meta = load_subset(n_isolates=500, ds_dir=ds)
    print(f'  positives: {len(pos_df)}, isolates: {pos_df["assembly_id_a"].nunique()}')
    isolate_to_cell = build_isolate_cells(iso_meta, year_match='binned')
    schema = (pos_df['func_a'].iloc[0], pos_df['func_b'].iloc[0])
    print(f'  schema: {schema}')

    neg, stats = create_negative_pairs_v2(
        pos_df=pos_df, num_negatives=2500, cooccur_pairs=set(pos_df['pair_key']),
        schema_pair=schema, seed=42,
        axis_quotas=_full_target(), isolate_to_cell=isolate_to_cell,
        sampling_axes=['host', 'hn_subtype', 'year'], on_shortfall='redistribute',
    )
    print(f'  achieved {len(neg)} pairs')
    print(f'  match_count: {neg["metadata_match_count"].value_counts(dropna=False).sort_index().to_dict()}')
    assert (neg['assembly_id_a'] == neg['assembly_id_b']).sum() == 0
    assert neg['pair_key'].duplicated().sum() == 0
    assert neg['pair_key'].isin(set(pos_df['pair_key'])).sum() == 0


if __name__ == '__main__':
    test_determinism()
    test_single_cell_bundle()
    test_on_shortfall_error()
    test_validation_missing_regime()
    test_bad_sum()
    test_pb2_pb1()
    print('\nDone. All edge cases passed.')
