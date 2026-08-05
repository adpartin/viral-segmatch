#!/usr/bin/env python
"""Bit-exact regression guard for the two production split paths.

Builds each production bundle and digests its per-fold pair assignment, so a refactor of the
routing or fragmentation code can be shown not to move a single pair.

    capture [--only NAME ...]   # build each path and write its golden
    check   [--only NAME ...]   # rebuild and diff against the goldens (exit != 0 on mismatch)
    extract --dir RUN_DIR       # print the digest of an existing run dir

Guards the paths named in `docs/plans/2026-08-03_fold_maker_consolidation_plan.md` § 1: 2D-CD
(`dataset_pairs_cc.py`) and 1D-CD on the HA axis (`dataset_segment_pairs.py`). Both are K-fold and
emit `fold_k/{train,val,test}_pairs.csv`, which is what the digest covers.

Runs the FULL corpus -- no subsampling. The 2D-CD builder rejects
`dataset.max_isolates_to_process` outright, and a partial corpus would change the cluster
components the router packs. A `check` therefore costs a few minutes per path, which is why this
is a script you run before committing split-affecting changes rather than a pytest case.

Distinct from `scripts/split_regression_harness.py`, which guards HOLDOUT-mode routing
(`random`, `seq_disjoint`, `metadata_holdout`) at a subsampled N.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = PROJECT_ROOT / 'tests' / 'golden' / 'production_splits'
BUILD_ROOT = PROJECT_ROOT / 'results' / 'flu' / 'July_2025' / 'runs' / 'production_split_harness'
SPLITS = ('train', 'val', 'test')

# One entry per production path. `builder` selects the CLI: the two take different
# output flags, and only the v2 one accepts --override.
GUARD_SET = [
    {'name': '2d_cd_t099', 'bundle': 'flu_ha_na_cc_nt_cds_cm0_wf', 'builder': 'cc'},
    {'name': '1d_cd_ha_t099', 'bundle': 'flu_ha_na_1dcd_nt_cds', 'builder': 'v2'},
]

BUILDER_CLI = {
    'cc': ('src/datasets/dataset_pairs_cc.py', '--out_dir'),
    'v2': ('src/datasets/dataset_segment_pairs.py', '--output_dir'),
}


def _sha256(values) -> str:
    """Order-independent digest of a set of strings."""
    joined = '\n'.join(sorted(str(v) for v in values))
    return hashlib.sha256(joined.encode()).hexdigest()


def digest_run_dir(run_dir: Path) -> dict:
    """Per-fold pair digests for one build.

    Records, for each fold and split, the row count and a hash of the `pair_key` set, plus a
    positives-only hash. Splitting positives out localizes a mismatch: positives moving means the
    router changed, positives holding while the full set moves means only negative sampling did.

    Args:
        run_dir: a build directory holding `fold_*/{train,val,test}_pairs.csv`.

    Returns:
        {fold: {split: {n, pair_keys_sha256, n_pos, pos_pair_keys_sha256}}}.
    """
    import pandas as pd

    fold_dirs = sorted(run_dir.glob('fold_*'))
    if not fold_dirs:
        raise SystemExit(f"no fold_* directories under {run_dir}")

    digest: dict = {}
    for fold_dir in fold_dirs:
        per_split: dict = {}
        for split in SPLITS:
            csv = fold_dir / f'{split}_pairs.csv'
            if not csv.exists():
                continue
            # Only the two columns the digest needs: reading all ~30 on 600k rows is slow and
            # trips pandas' mixed-dtype inference on the sparsely-filled hash columns.
            # 'NA' (Neuraminidase) is a literal value elsewhere in these files -- keep strings.
            df = pd.read_csv(csv, usecols=['pair_key', 'label'],
                             keep_default_na=False, na_values=[''])
            pos = df[df['label'] == 1]
            per_split[split] = {
                'n': int(len(df)),
                'pair_keys_sha256': _sha256(df['pair_key']),
                'n_pos': int(len(pos)),
                'pos_pair_keys_sha256': _sha256(pos['pair_key']),
            }
        digest[fold_dir.name] = per_split
    return digest


def build(guard: dict, out_dir: Path) -> Path:
    """Run Stage 3 for one guard into `out_dir`, using that path's builder CLI."""
    script, out_flag = BUILDER_CLI[guard['builder']]
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(PROJECT_ROOT / script),
           '--config_bundle', guard['bundle'], out_flag, str(out_dir)]
    print(f"  $ {' '.join(cmd)}", flush=True)
    if subprocess.run(cmd, cwd=PROJECT_ROOT).returncode != 0:
        raise SystemExit(f"build FAILED for {guard['bundle']} (see Stage-3 output above)")
    return out_dir


def _selected(only):
    if not only:
        return GUARD_SET
    chosen = [g for g in GUARD_SET if g['name'] in set(only)]
    unknown = set(only) - {g['name'] for g in chosen}
    if unknown:
        raise SystemExit(f"unknown guard names: {sorted(unknown)} "
                         f"(known: {[g['name'] for g in GUARD_SET]})")
    return chosen


def cmd_extract(args):
    print(json.dumps(digest_run_dir(Path(args.dir)), indent=2))


def cmd_capture(args):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for guard in _selected(args.only):
        print(f"[capture] {guard['name']}  ({guard['bundle']})", flush=True)
        run_dir = build(guard, BUILD_ROOT / f"capture_{guard['name']}")
        golden = {'bundle': guard['bundle'], 'builder': guard['builder'],
                  'folds': digest_run_dir(run_dir)}
        path = GOLDEN_DIR / f"{guard['name']}.json"
        path.write_text(json.dumps(golden, indent=2) + '\n')
        print(f"  wrote {path.relative_to(PROJECT_ROOT)}")


def cmd_check(args):
    failures = []
    for guard in _selected(args.only):
        path = GOLDEN_DIR / f"{guard['name']}.json"
        if not path.exists():
            raise SystemExit(f"no golden for {guard['name']}; run `capture` first ({path})")
        print(f"[check] {guard['name']}  ({guard['bundle']})", flush=True)
        run_dir = build(guard, BUILD_ROOT / f"check_{guard['name']}")
        expected = json.loads(path.read_text())['folds']
        actual = digest_run_dir(run_dir)
        if actual == expected:
            print(f"  OK  {len(actual)} folds match")
            continue
        failures.append(guard['name'])
        for fold in sorted(set(expected) | set(actual)):
            for split in SPLITS:
                exp = expected.get(fold, {}).get(split)
                act = actual.get(fold, {}).get(split)
                if exp != act:
                    print(f"  MISMATCH {fold}/{split}: expected {exp} got {act}")

    if failures:
        raise SystemExit(
            f"ERROR: {len(failures)} guard(s) changed: {failures}\n"
            f"  If the split code changed, this is the regression it is here to catch.\n"
            f"  If the Stage-1 corpus was rebuilt, the pairs changed for a legitimate reason --\n"
            f"  re-capture with: python {Path(__file__).name} capture --only {' '.join(failures)}")
    print('\nDone. All production splits bit-exact.')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)

    p_extract = sub.add_parser('extract', help='digest an existing run dir')
    p_extract.add_argument('--dir', required=True)
    p_extract.set_defaults(func=cmd_extract)

    p_capture = sub.add_parser('capture', help='build each path and write its golden')
    p_capture.add_argument('--only', nargs='+', default=None)
    p_capture.set_defaults(func=cmd_capture)

    p_check = sub.add_parser('check', help='rebuild and diff against the goldens')
    p_check.add_argument('--only', nargs='+', default=None)
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
