"""Snapshot the per-pair_key fold assignment of a 2D-CD CV run.

Reads the recorded `fold_*/test_pairs.csv` of a `dataset_pairs_cc.py` run and writes a
`pair_key -> fold` digest. Each positive pair is the test pair of exactly one fold, so the test
sets partition the positives and the mapping is total.

Purpose: the `cc_id` labelling rule is changing (union-find root order -> `(-pair_count,
min node id)`). That is partition-preserving but not split-preserving: CCs tied on pair count are
ordered by `cc_id` in the LPT bin-pack, so tied CCs can move between folds. Run this before and
after to measure how many pairs actually moved instead of assuming.

Usage:
    python scripts/capture_2dcd_fold_baseline.py <run_dir> <out_json>
    python scripts/capture_2dcd_fold_baseline.py <run_dir> <out_json> --compare <baseline_json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def fold_assignment(run_dir: Path) -> dict:
    """Map each positive pair_key to the fold whose TEST split holds it."""
    mapping: dict = {}
    for fold_dir in sorted(run_dir.glob('fold_*')):
        test_csv = fold_dir / 'test_pairs.csv'
        if not test_csv.exists():
            continue
        # 'NA' (Neuraminidase) is a literal value in function_short columns -- keep it a string.
        df = pd.read_csv(test_csv, keep_default_na=False, na_values=[''])
        pos = df[df['label'] == 1] if 'label' in df.columns else df
        for k in pos['pair_key']:
            mapping[k] = fold_dir.name
    return mapping


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('run_dir', type=Path, help='a dataset_pairs_cc run dir holding fold_*/')
    p.add_argument('out_json', type=Path, help='where to write the digest')
    p.add_argument('--full', action='store_true',
                   help='also embed the full pair_key -> fold mapping (large; for diagnosis).')
    p.add_argument('--compare', type=Path, default=None,
                   help='an earlier digest to diff against; reports which folds changed.')
    args = p.parse_args()

    mapping = fold_assignment(args.run_dir)
    if not mapping:
        raise SystemExit(f'ERROR: no fold_*/test_pairs.csv under {args.run_dir}')

    # Digest, not the full mapping: one sha256 per fold over its sorted pair_keys. Matches the
    # sibling goldens in tests/golden/megacc_cut/ and keeps the artifact ~1 KB rather than ~6 MB.
    # Detects any change in fold membership; use --full to also emit the mapping for diagnosing
    # WHICH pairs moved (write that outside the repo -- it is large).
    per_fold: dict = {}
    for k, fold in mapping.items():
        per_fold.setdefault(fold, []).append(k)
    digest = {fold: hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest()
              for fold, keys in sorted(per_fold.items())}

    out = {
        'run_dir': str(args.run_dir),
        'n_pairs': len(mapping),
        'n_pairs_by_fold': {f: len(k) for f, k in sorted(per_fold.items())},
        'pair_keys_sha256_by_fold': digest,
    }
    if args.full:
        out['fold_by_pair_key'] = mapping

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f'wrote {args.out_json} ({len(mapping):,} pairs, {len(digest)} folds)')

    if args.compare is not None:
        base = json.loads(args.compare.read_text())
        same = base.get('pair_keys_sha256_by_fold') == digest
        print(f'\ncompared against {args.compare}')
        print(f'  fold assignment identical : {same}')
        for fold in sorted(digest):
            b = base.get('pair_keys_sha256_by_fold', {}).get(fold)
            print(f'    {fold}: {"SAME" if b == digest[fold] else "DIFFERENT"}')
        # Pair-level detail only when both sides carry the full mapping.
        bmap = base.get('fold_by_pair_key')
        if bmap:
            shared = set(bmap) & set(mapping)
            moved = sum(1 for k in shared if bmap[k] != mapping[k])
            print(f'  pairs in both             : {len(shared):,}')
            print(f'  CHANGED fold              : {moved:,} ({moved / len(shared):.2%})')
    print('\nDone.')


if __name__ == '__main__':
    main()
