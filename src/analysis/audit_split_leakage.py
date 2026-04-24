"""Post-hoc leakage audit for Stage 3 pair splits.

Purpose
-------
Operates on the already-written pair CSVs (train_pairs.csv, val_pairs.csv,
test_pairs.csv) in a single fold directory and reports a 6-bucket leakage
table at two levels:

  - protein    : feature key = canonical_pair_key(seq_hash_a, seq_hash_b)
  - nucleotide : feature key = canonical_pair_key(dna_hash_a, dna_hash_b)

The 6 buckets are (train_vs_val, train_vs_test, val_vs_test) x
(label-match rows, label-conflict rows), evaluated from the perspective of
the later split B:

  - shared_keys      : unique feature keys that appear in both A and B
  - B_match_rows     : rows in B whose feature key appears in A with the
                       SAME label (memorization hit -- inflates metrics on B)
  - B_conflict_rows  : rows in B whose feature key appears in A but only
                       with the OPPOSITE label (contradictory signal)

Relationship to src/datasets/dataset_segment_pairs.py
-----------------------------------------------------
Stage 3 *already* performs the equivalent of the protein-level check inline
before writing the pair CSVs (see the "PAIR-KEY BASED SPLITTING" block in
dataset_segment_pairs.py, currently around lines 1029-1074). Stage 3
computes the same three pair_key intersections and *removes* overlapping
rows from val and test (keeping them in train). So for any fold written by
the current Stage 3, the protein-level B_match_rows and B_conflict_rows
columns are structurally guaranteed to be 0. This script does NOT duplicate
that work during dataset creation -- it validates after the fact.

What this script adds on top of Stage 3's inline check
------------------------------------------------------
1. Regression guard. If a future refactor breaks Stage 3's pair_key dedup,
   running this script on the written pair CSVs will surface it without
   needing to re-read the Stage 3 log.
2. Sweepable across many runs / folds without re-running Stage 3.
3. Nucleotide-level check. Stage 3 only dedups on protein pair_key. This
   script also intersects (dna_hash_a, dna_hash_b) pair-keys across splits.
   For the current 28-pair Task 11 bundles (one function per segment,
   cross-segment pairs only), nucleotide pair-keys are structurally <=
   protein pair-keys -- so protein=0 implies nucleotide=0 for those pairs.
   The nucleotide check becomes genuinely additive only for pair bundles
   where two selected proteins can share a DNA contig (e.g. M1+M2, NS1+NEP,
   PA+PA-X), which the 28-pair setup deliberately avoids.
4. Portable: runs on any directory with pair CSVs that carry the required
   columns, not only those produced by this repo's Stage 3.

Scope
-----
This is a RAW-SEQUENCE-IDENTITY audit. It answers "do identical
(seq_hash or dna_hash) pair keys cross splits?". It does NOT detect
feature-vector near-duplicates -- e.g., two DNA sequences that produce
near-identical k-mer vectors while remaining distinct on dna_hash. A
feature-vector similarity audit is a separate, heavier lift.

Requires the pair CSVs to carry columns: seq_hash_a, seq_hash_b,
dna_hash_a, dna_hash_b, label. The dna_hash columns are added by Stage 3
as of the DNA-enrichment change in src/datasets/dataset_segment_pairs.py.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Set

import pandas as pd


def _canonical(a: str, b: str) -> str:
    a, b = str(a), str(b)
    return f"{a}__{b}" if a <= b else f"{b}__{a}"


def _add_feature_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['prot_key'] = [_canonical(a, b) for a, b in zip(df['seq_hash_a'], df['seq_hash_b'])]
    df['dna_key'] = [_canonical(a, b) for a, b in zip(df['dna_hash_a'], df['dna_hash_b'])]
    return df


def _audit_one_level(split_a: pd.DataFrame, split_b: pd.DataFrame, key_col: str) -> dict:
    a_labels_by_key: Dict[str, Set[int]] = {}
    for k, lab in zip(split_a[key_col], split_a['label']):
        a_labels_by_key.setdefault(k, set()).add(int(lab))

    shared_keys: Set[str] = set()
    match_rows = 0
    conflict_rows = 0
    for k, lab in zip(split_b[key_col], split_b['label']):
        a_labels = a_labels_by_key.get(k)
        if a_labels is None:
            continue
        shared_keys.add(k)
        if int(lab) in a_labels:
            match_rows += 1
        else:
            conflict_rows += 1

    return {
        'rows_B': int(len(split_b)),
        'shared_keys': int(len(shared_keys)),
        'B_match_rows': int(match_rows),
        'B_conflict_rows': int(conflict_rows),
    }


def audit(fold_dir: Path) -> dict:
    train = pd.read_csv(fold_dir / 'train_pairs.csv')
    val = pd.read_csv(fold_dir / 'val_pairs.csv')
    test = pd.read_csv(fold_dir / 'test_pairs.csv')

    required = ('seq_hash_a', 'seq_hash_b', 'dna_hash_a', 'dna_hash_b', 'label')
    for name, df in (('train', train), ('val', val), ('test', test)):
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{name}_pairs.csv is missing columns {missing}. "
                "Regenerate Stage 3 with the DNA-enrichment change landed."
            )

    train = _add_feature_keys(train)
    val = _add_feature_keys(val)
    test = _add_feature_keys(test)

    result = {
        'fold_dir': str(fold_dir),
        'split_sizes': {'train': int(len(train)), 'val': int(len(val)), 'test': int(len(test))},
    }
    for level, key_col in (('protein', 'prot_key'), ('nucleotide', 'dna_key')):
        result[level] = {
            'train_vs_val': _audit_one_level(train, val, key_col),
            'train_vs_test': _audit_one_level(train, test, key_col),
            'val_vs_test': _audit_one_level(val, test, key_col),
        }
    return result


def _print_report(r: dict) -> None:
    print(f"\nLeakage audit: {r['fold_dir']}")
    print(f"Split sizes: train={r['split_sizes']['train']:,}  "
          f"val={r['split_sizes']['val']:,}  test={r['split_sizes']['test']:,}\n")

    for level in ('protein', 'nucleotide'):
        key_desc = 'seq_hash' if level == 'protein' else 'dna_hash'
        print(f"[{level}] feature key = canonical_pair_key({key_desc}_a, {key_desc}_b)")
        header = (f"  {'bucket':<16} {'rows_B':>10} {'shared_keys':>12} "
                  f"{'match_rows':>12} {'conflict_rows':>14}")
        print(header)
        print('  ' + '-' * (len(header) - 2))
        for bucket in ('train_vs_val', 'train_vs_test', 'val_vs_test'):
            d = r[level][bucket]
            print(f"  {bucket:<16} {d['rows_B']:>10,d} {d['shared_keys']:>12,d} "
                  f"{d['B_match_rows']:>12,d} {d['B_conflict_rows']:>14,d}")
        print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--fold_dir', required=True, type=Path,
                   help="Directory containing train_pairs.csv, val_pairs.csv, test_pairs.csv")
    p.add_argument('--json_out', type=Path, default=None,
                   help="Optional path to write the report as JSON")
    args = p.parse_args()

    report = audit(args.fold_dir)
    _print_report(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"Wrote JSON report to {args.json_out}")


if __name__ == '__main__':
    main()
