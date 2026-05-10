"""Integration test: call split_dataset_v2 end-to-end with the new
negative_sampling config. Verify manifest is in the duplicate_stats output,
pair CSVs carry neg_regime and metadata_match_count, all hard invariants.

Build a small synthetic df from the production HA/NA dataset's positives
+ isolate metadata.
"""
import sys
from pathlib import Path
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.datasets.dataset_segment_pairs_v2 import (
    split_dataset_v2, save_split_output_v2,
)
from e2e_regime import load_subset


def main():
    n_iso = 800
    print(f'Loading {n_iso}-isolate HA/NA subset...')
    pos_df, iso_meta = load_subset(n_isolates=n_iso)

    # Reconstruct a "df" from pos_df: each pos row has both proteins of one
    # isolate, so two protein rows per pos row.
    rows = []
    for r in pos_df.itertuples(index=False):
        rows.append({
            'assembly_id': r.assembly_id_a, 'function': r.func_a,
            'brc_fea_id': r.brc_a, 'genbank_ctg_id': r.ctg_a,
            'prot_seq': r.seq_a, 'dna_seq': r.dna_seq_a,
            'canonical_segment': r.seg_a,
            'seq_hash': r.seq_hash_a, 'dna_hash': r.dna_hash_a,
        })
        rows.append({
            'assembly_id': r.assembly_id_a, 'function': r.func_b,
            'brc_fea_id': r.brc_b, 'genbank_ctg_id': r.ctg_b,
            'prot_seq': r.seq_b, 'dna_seq': r.dna_seq_b,
            'canonical_segment': r.seg_b,
            'seq_hash': r.seq_hash_b, 'dna_hash': r.dna_hash_b,
        })
    df = pd.DataFrame(rows)
    df['assembly_id'] = df['assembly_id'].astype(str)
    iso_meta['assembly_id'] = iso_meta['assembly_id'].astype(str)
    df = df.merge(iso_meta, on='assembly_id', how='left')
    print(f'  df rows: {len(df)}, isolates: {df["assembly_id"].nunique()}')

    target = {
        'none_match':           0.20, 'host_only':            0.10,
        'subtype_only':         0.10, 'year_only':            0.10,
        'host_subtype_only':    0.10, 'host_year_only':       0.05,
        'subtype_year_only':    0.05, 'host_subtype_year':    0.30,
        'unknown_metadata_neg': 0.00,
    }

    schema = ('Hemagglutinin precursor', 'Neuraminidase protein')
    train_pairs, val_pairs, test_pairs, dup_stats, exposures = split_dataset_v2(
        df=df,
        schema_pair=schema,
        neg_to_pos_ratio=3.0,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
        axis_quotas=target,
        sampling_axes=['host', 'hn_subtype', 'year'],
        year_match='binned',
        on_shortfall='redistribute',
    )

    print(f'\nSplit sizes: train={len(train_pairs):,}, val={len(val_pairs):,}, test={len(test_pairs):,}')

    print(f'\nManifest sanity: regime_manifest in dup_stats: {"regime_manifest" in dup_stats}')
    print(f"  splits: {list(dup_stats['regime_manifest']['splits'].keys())}")
    for split, manifest in dup_stats['regime_manifest']['splits'].items():
        achieved = sum(r['achieved'] for r in manifest)
        target_sum = sum(r['target'] for r in manifest)
        cov = sum(r['coverage_placed'] for r in manifest)
        print(f'  {split}: target_sum={target_sum:,} achieved={achieved:,} (coverage_placed={cov:,})')

    print(f'\nPair CSV columns include neg_regime: {"neg_regime" in train_pairs.columns}')
    print(f'Pair CSV columns include metadata_match_count: {"metadata_match_count" in train_pairs.columns}')

    train_neg = train_pairs[train_pairs['label'] == 0]
    train_pos = train_pairs[train_pairs['label'] == 1]
    print(f'\nTrain pos with non-NA neg_regime: '
          f'{train_pos["neg_regime"].notna().sum()} (should be 0)')
    print(f'Train neg with NA neg_regime: '
          f'{train_neg["neg_regime"].isna().sum()} (= unknown_metadata_neg count or 0)')

    print(f'\nMatch count distribution (train negatives):')
    print(train_neg['metadata_match_count'].value_counts(dropna=False).sort_index().to_string())

    print(f'\nHard invariants:')
    same_iso = (train_neg['assembly_id_a'] == train_neg['assembly_id_b']).sum()
    print(f'  train: same-isolate negs={same_iso}')
    pk_train = set(train_pairs['pair_key'])
    pk_val = set(val_pairs['pair_key'])
    pk_test = set(test_pairs['pair_key'])
    print(f'  cross-split pair_key overlap: train-val={len(pk_train & pk_val)}, '
          f'train-test={len(pk_train & pk_test)}, val-test={len(pk_val & pk_test)}')

    # Save to scratch dir to confirm manifest write
    out_dir = Path('/tmp/test_regime_dataset_out')
    out_dir.mkdir(exist_ok=True)
    save_split_output_v2(
        output_dir=out_dir,
        train_pairs=train_pairs, val_pairs=val_pairs, test_pairs=test_pairs,
        duplicate_stats=dup_stats,
        exposure_tables=exposures, df=df,
        config_bundle='regime_test',
        schema_pair=schema,
        filters_applied={},
        axes_for_flags=['host', 'hn_subtype', 'year', 'geo_location', 'passage'],
        generate_visualizations=False,
    )
    print(f'\nFiles written under {out_dir}:')
    for f in sorted(out_dir.iterdir()):
        if 'regime' in f.name or f.name.endswith('.json'):
            print(f'  {f.name}: {f.stat().st_size} bytes')


if __name__ == '__main__':
    main()
