"""End-to-end: build a small dataset with the regime-aware sampler on real
HA/NA data. Verify hard invariants + regime distribution.

Strategy: take 500 isolates from the production protein_final.csv +
metadata; run create_positive_pairs_v2 + create_negative_pairs_v2 in
regime-aware mode; check the manifest, neg_regime/metadata_match_count
columns, no same-isolate negatives, no cross-split duplicates.
"""
import sys
from pathlib import Path
import json
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.utils.metadata_enrichment import enrich_prot_data_with_metadata
from src.datasets._pair_helpers import attach_dna_to_prot_df, build_cooccurrence_set
from src.datasets.dataset_segment_pairs_v2 import (
    create_positive_pairs_v2,
    create_negative_pairs_v2,
)
from src.datasets._negative_regime_sampling import (
    REGIME_NAMES, build_isolate_cells, count_isolates_per_cell,
    count_available_per_regime,
)


def load_subset(n_isolates=500, ds_dir=None):
    """Take pos_df from a production HA/NA dataset's train_pairs.csv as
    positives (label==1) and downsample to n_isolates. Reconstruct the df
    that compute_axis_flags / split_dataset_v2 expect from the available
    columns. Avoids needing to rerun preprocessing for the e2e test."""
    ds_dir = ds_dir or (PROJ / 'data/datasets/flu/July_2025/runs/dataset_flu_ha_na_20260508_171512')
    train = pd.read_csv(ds_dir / 'train_pairs.csv', engine='python')
    pos = train[train['label'] == 1].copy()
    # Take first n unique isolates (assembly_id_a == assembly_id_b on positives).
    isos = sorted(pos['assembly_id_a'].astype(str).unique().tolist())[:n_isolates]
    pos = pos[pos['assembly_id_a'].astype(str).isin(isos)].reset_index(drop=True)

    # Reconstruct a "df" (per-protein rows) from the pos pair CSV: each pos
    # row contributes two rows (one per slot). We need columns that
    # create_negative_pairs_v2 won't actually use directly (it already has
    # everything in pos_df), but isolate_to_cell needs assembly_id + axis cols.
    iso_meta = pd.read_csv(ds_dir / 'isolate_metadata.csv', engine='python')
    iso_meta['assembly_id'] = iso_meta['assembly_id'].astype(str)
    iso_meta = iso_meta[iso_meta['assembly_id'].isin(isos)]
    return pos, iso_meta


def main():
    print('Loading + filtering...')
    pos_df, iso_meta = load_subset(n_isolates=500)
    print(f'  positives: {len(pos_df):,}, isolates: {pos_df["assembly_id_a"].nunique():,}')

    # cooccur_pairs from the pos_df itself (each pos pair_key co-occurs in
    # exactly one isolate after v2 dedup). This is enough to block
    # contradictory negs.
    cooccur_pairs = set(pos_df['pair_key'])

    isolate_to_cell = build_isolate_cells(iso_meta, year_match='binned')
    print(f'  cells (binned year): {len(set(isolate_to_cell.values()))}')

    # Closed-form counts on this subset.
    cell_counts = count_isolates_per_cell(isolate_to_cell)
    avail = count_available_per_regime(cell_counts)
    total = sum(avail.values())
    print(f'  closed-form total ordered pairs: {total:,}')

    target = {
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

    num_neg = max(len(pos_df) * 5, 3000)  # ~5x ratio to exercise fill phase
    neg_df, stats = create_negative_pairs_v2(
        pos_df=pos_df,
        num_negatives=num_neg,
        cooccur_pairs=cooccur_pairs,
        schema_pair=('Hemagglutinin precursor', 'Neuraminidase protein'),
        seed=42,
        axis_quotas=target,
        isolate_to_cell=isolate_to_cell,
        sampling_axes=['host', 'hn_subtype', 'year'],
        on_shortfall='redistribute',
    )
    print(f'\nnum_negatives requested = {num_neg:,}')

    print(f'\nResult: {len(neg_df):,} neg pairs')
    print(f'Coverage phase: {stats["coverage_phase_pairs"]:,}, '
          f'Fill phase: {stats["fill_phase_pairs"]:,}, '
          f'Total attempts: {stats["total_attempts"]:,}')

    print('\nRegime manifest:')
    print(f"{'regime':<25} {'target':>6} {'avail':>10} {'cov':>5} {'fill':>5} {'achi':>5} reason")
    for row in stats['regime_manifest']:
        print(f"  {row['regime']:<25} {row['target']:>6,} {row['available']:>10,} "
              f"{row['coverage_placed']:>5,} {row['fill_placed']:>5,} "
              f"{row['achieved']:>5,} {row['shortfall_reason'] or ''}")

    print(f'\nMatch count distribution (negatives):')
    print(neg_df['metadata_match_count'].value_counts(dropna=False).sort_index().to_string())

    print(f'\nHard invariants:')
    same_iso = ((neg_df['assembly_id_a'] == neg_df['assembly_id_b']).sum())
    print(f'  same-isolate negatives: {same_iso} (must be 0)')
    pk_dup = (neg_df['pair_key'].duplicated().sum())
    print(f'  duplicate pair_keys within neg: {pk_dup} (must be 0)')
    cooccur_violations = (neg_df['pair_key'].isin(cooccur_pairs).sum())
    print(f'  cooccur violations: {cooccur_violations} (must be 0)')

    null_regime = neg_df['neg_regime'].isna().sum()
    print(f'  neg rows with null neg_regime: {null_regime} (should be 0 unless unknown_metadata_neg pairs exist)')

    print('\nDone.')


if __name__ == '__main__':
    main()
