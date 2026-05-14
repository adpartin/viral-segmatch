"""
One-shot migration: add assembly_id column to master_esm2_embeddings.parquet.

Background: the ESM-2 lookup index was previously keyed by brc_fea_id alone.
The cache-symmetry design (docs/plans/2026-05-13_aa_kmer_and_cache_symmetry_plan.md)
shifts the public lookup to a composite (assembly_id, brc_fea_id) tuple — same
contract that the new aa k-mer cache will use. The HDF5 embeddings themselves
are unchanged; only the parquet index gains a column.

This script is idempotent: if assembly_id is already present, it does nothing.

Usage:
    python scripts/migrate_esm2_parquet_add_assembly_id.py \
        --parquet data/embeddings/flu/July_2025/master_esm2_embeddings.parquet \
        --protein_csv data/processed/flu/July_2025/protein_final.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', type=Path, required=True,
                        help='Path to master_esm2_embeddings.parquet')
    parser.add_argument('--protein_csv', type=Path, required=True,
                        help='Path to protein_final.csv (source of assembly_id)')
    args = parser.parse_args()

    if not args.parquet.exists():
        print(f'ERROR: parquet not found: {args.parquet}', file=sys.stderr)
        sys.exit(1)
    if not args.protein_csv.exists():
        print(f'ERROR: protein_csv not found: {args.protein_csv}', file=sys.stderr)
        sys.exit(1)

    print(f'Read parquet: {args.parquet}')
    df = pd.read_parquet(args.parquet)
    print(f'  rows: {len(df):,}, columns: {list(df.columns)}')

    if 'assembly_id' in df.columns:
        print('assembly_id already present, nothing to do.')
        return

    print(f'Read protein_final.csv: {args.protein_csv}')
    prot = pd.read_csv(args.protein_csv, usecols=['brc_fea_id', 'assembly_id'])
    print(f'  rows: {len(prot):,}')

    n_before = len(df)
    df = df.merge(prot, on='brc_fea_id', how='left')
    n_after = len(df)
    if n_before != n_after:
        print(f'ERROR: row count changed during merge: {n_before:,} -> {n_after:,}',
              file=sys.stderr)
        sys.exit(2)

    n_missing = df['assembly_id'].isna().sum()
    if n_missing > 0:
        print(f'ERROR: {n_missing:,} rows have no assembly_id after merge', file=sys.stderr)
        sys.exit(3)

    print(f'  merged, all {len(df):,} rows have assembly_id')
    df = df[['cache_key', 'row', 'brc_fea_id', 'assembly_id']]

    print(f'Write parquet: {args.parquet}')
    df.to_parquet(args.parquet, index=False)
    print('Done.')


if __name__ == '__main__':
    main()
