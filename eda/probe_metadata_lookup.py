"""Empirical probe: how badly does seq_hash -> metadata.first() misrepresent
pair-level metadata, and is dna_hash any better than seq_hash?

Approach: every pair already carries seq_hash_a/b, dna_hash_a/b, assembly_id_a/b.
isolate_metadata.csv carries the gold-standard per-isolate metadata. Reshape
the pair tables into a tall (slot, hash, assembly) frame to build the three
lookups (seq_hash, dna_hash, assembly_id), each via .first(). Compare the
seq/dna lookups against the gold (assembly_id) lookup.

Run on HA/NA mixed (external segments) and PB2/PB1 (internal segments). The
NOTE/TODO in compute_metadata_coverage calls out internal proteins as the
problem case; this probe quantifies it.

Usage:
    python eda/probe_metadata_lookup.py
"""
from pathlib import Path
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]

DATASETS = [
    ('HA/NA',   PROJ / 'data/datasets/flu/July_2025/runs/dataset_flu_ha_na_20260508_171512'),
    ('PB2/PB1', PROJ / 'data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_20260509_114318'),
]
AXES = ['host', 'year', 'hn_subtype']


def load_pairs_all_splits(ds_dir: Path) -> pd.DataFrame:
    """Concatenate train/val/test pair CSVs to give us the full row pool."""
    parts = []
    for split in ['train', 'val', 'test']:
        p = ds_dir / f'{split}_pairs.csv'
        df = pd.read_csv(p, engine='python')
        df['_split'] = split
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def build_tall_rows(pairs: pd.DataFrame) -> pd.DataFrame:
    """Reshape pair table into one-row-per-slot view: 2*N rows, with one
    seq_hash, dna_hash, and assembly_id per row. Used for groupby lookups."""
    a = pairs[['assembly_id_a', 'seq_hash_a', 'dna_hash_a']].rename(
        columns={'assembly_id_a': 'assembly_id', 'seq_hash_a': 'seq_hash', 'dna_hash_a': 'dna_hash'}
    )
    b = pairs[['assembly_id_b', 'seq_hash_b', 'dna_hash_b']].rename(
        columns={'assembly_id_b': 'assembly_id', 'seq_hash_b': 'seq_hash', 'dna_hash_b': 'dna_hash'}
    )
    return pd.concat([a, b], ignore_index=True).drop_duplicates()


def measure_one_dataset(label: str, ds_dir: Path):
    print(f'\n{"="*72}\n {label}  ({ds_dir.name})\n{"="*72}')

    iso_meta = pd.read_csv(ds_dir / 'isolate_metadata.csv', engine='python')
    iso_meta['assembly_id'] = iso_meta['assembly_id'].astype(str)
    iso_meta = iso_meta.set_index('assembly_id')
    print(f'isolate_metadata.csv: {len(iso_meta):,} isolates')

    pairs = load_pairs_all_splits(ds_dir)
    pairs['assembly_id_a'] = pairs['assembly_id_a'].astype(str)
    pairs['assembly_id_b'] = pairs['assembly_id_b'].astype(str)
    print(f'pairs (train+val+test): {len(pairs):,}')

    tall = build_tall_rows(pairs)
    tall['assembly_id'] = tall['assembly_id'].astype(str)
    # Attach metadata onto every (assembly_id, seq_hash, dna_hash) row.
    tall = tall.merge(iso_meta[AXES], left_on='assembly_id', right_index=True, how='left')
    print(f'tall (deduped slot rows): {len(tall):,} '
          f'(unique seq_hashes={tall["seq_hash"].nunique():,}, '
          f'unique dna_hashes={tall["dna_hash"].nunique():,}, '
          f'unique assemblies={tall["assembly_id"].nunique():,})')

    # ----- Per-hash conflict counts (how often a hash sees >1 distinct value) -----
    print('\n--- Per-hash axis-value conflicts (across the union of all isolates that carry it) ---')
    for axis in AXES:
        sub = tall.dropna(subset=[axis])
        seq_n = sub.groupby('seq_hash')[axis].nunique()
        dna_n = sub.groupby('dna_hash')[axis].nunique()
        seq_pct = 100 * (seq_n > 1).mean()
        dna_pct = 100 * (dna_n > 1).mean()
        print(f'  {axis:>11}: '
              f'seq_hash {(seq_n > 1).sum():>6,}/{len(seq_n):>6,} ({seq_pct:5.2f}%, max={seq_n.max()})  |  '
              f'dna_hash {(dna_n > 1).sum():>6,}/{len(dna_n):>6,} ({dna_pct:5.2f}%, max={dna_n.max()})')

    # ----- Pair-level disagreement vs gold standard -----
    # Gold-standard <axis>_a, <axis>_b come from joining assembly_id_a/b on iso_meta.
    # Compare to:
    #   (a) the existing <axis>_a / <axis>_b columns in the pair CSV (built by
    #       compute_axis_flags via seq_hash.first())
    #   (b) a fresh dna_hash.first() lookup (alternate)
    per_seq = tall.dropna(subset=AXES).groupby('seq_hash')[AXES].first()
    per_dna = tall.dropna(subset=AXES).groupby('dna_hash')[AXES].first()

    print('\n--- Pair-level lookup disagreement vs gold (assembly_id) ---')
    print('Each row: % of pairs (where both gold and the lookup are non-null) where the')
    print('lookup disagrees with what the actual pair-isolate carries.')
    for axis in AXES:
        gold_a = pairs['assembly_id_a'].map(iso_meta[axis])
        gold_b = pairs['assembly_id_b'].map(iso_meta[axis])

        seq_a = pairs['seq_hash_a'].map(per_seq[axis])
        seq_b = pairs['seq_hash_b'].map(per_seq[axis])
        dna_a = pairs['dna_hash_a'].map(per_dna[axis])
        dna_b = pairs['dna_hash_b'].map(per_dna[axis])

        # Use the existing same_<axis> column and rebuild gold/seq/dna versions.
        def disagree(g, l):
            both = g.notna() & l.notna()
            n = int(both.sum())
            d = int(((g != l) & both).sum())
            return d, n, (100 * d / n) if n else 0.0

        sa_d, sa_n, sa_p = disagree(gold_a, seq_a)
        sb_d, sb_n, sb_p = disagree(gold_b, seq_b)
        da_d, da_n, da_p = disagree(gold_a, dna_a)
        db_d, db_n, db_p = disagree(gold_b, dna_b)
        print(f'  {axis}:')
        print(f'    side A (slot=func_left):  seq_hash {sa_d:>6,}/{sa_n:,} ({sa_p:5.2f}%)  '
              f'|  dna_hash {da_d:>6,}/{da_n:,} ({da_p:5.2f}%)')
        print(f'    side B (slot=func_right): seq_hash {sb_d:>6,}/{sb_n:,} ({sb_p:5.2f}%)  '
              f'|  dna_hash {db_d:>6,}/{db_n:,} ({db_p:5.2f}%)')

        # Pair-level same_<axis> agreement
        gold_same = ((gold_a == gold_b) & gold_a.notna() & gold_b.notna())
        gold_known = gold_a.notna() & gold_b.notna()
        seq_same = ((seq_a == seq_b) & seq_a.notna() & seq_b.notna())
        seq_known = seq_a.notna() & seq_b.notna()
        comparable = gold_known & seq_known
        if comparable.any():
            mismatched = ((gold_same != seq_same) & comparable).sum()
            tot = comparable.sum()
            # split by label for diagnostic relevance
            pos = comparable & (pairs['label'] == 1)
            neg = comparable & (pairs['label'] == 0)
            mm_pos = ((gold_same != seq_same) & pos).sum()
            mm_neg = ((gold_same != seq_same) & neg).sum()
            print(f'    pair-level same_{axis}: gold-vs-seq disagreement = '
                  f'{mismatched:,}/{tot:,} ({100*mismatched/tot:5.2f}%)   '
                  f'[neg={mm_neg:,}/{neg.sum():,} ({100*mm_neg/max(1,neg.sum()):.2f}%)  '
                  f'pos={mm_pos:,}/{pos.sum():,} ({100*mm_pos/max(1,pos.sum()):.2f}%)]')


def main():
    for label, ds_dir in DATASETS:
        if not ds_dir.exists():
            print(f'SKIP: {ds_dir} not found')
            continue
        measure_one_dataset(label, ds_dir)


if __name__ == '__main__':
    main()
