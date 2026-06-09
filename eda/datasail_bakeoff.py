"""DataSAIL vs segmatch bicc bake-off — Phase 0 sanity wrapper.

Reads positives from a Stage 3 dataset directory, optionally subsamples to
N isolates, runs DataSAIL with one or more split techniques (I2 / S2 / R / ...),
and writes a per-pair fold assignment + audit JSON.

**Important**: this script imports `datasail`, which is only installed in
the dedicated `datasail` conda env. Run with:

    conda run -n datasail python src/analysis/datasail_bakeoff.py \\
        --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_*_<TS> \\
        --n_isolates 100 \\
        --techniques I2 \\
        --output_dir results/flu/July_2025/runs/datasail_bakeoff/phase0_<TS>

See `docs/plans/2026-05-19_datasail_bakeoff_plan.md` for the full bake-off
context, decision criteria, and phase plan.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    from datasail.sail import datasail
except ImportError as e:
    sys.stderr.write(
        f"ERROR: cannot import datasail ({e}). This script must run in the "
        f"'datasail' conda env. Try:\n"
        f"    conda run -n datasail python {sys.argv[0]} ...\n"
    )
    sys.exit(1)


def load_positives(dataset_dir: Path) -> pd.DataFrame:
    """Concatenate train/val/test pair CSVs, return positives (label==1).

    The columns we need for routing are: assembly_id_a, seq_hash_a, seq_a,
    assembly_id_b, seq_hash_b, seq_b. Other columns are kept for downstream
    joins / audit.
    """
    frames = []
    for split in ('train_pairs.csv', 'val_pairs.csv', 'test_pairs.csv'):
        p = dataset_dir / split
        if not p.exists():
            raise FileNotFoundError(f"Expected {p}; not found.")
        df = pd.read_csv(p, low_memory=False)
        df['orig_split'] = split.replace('_pairs.csv', '')
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    pos = full[full['label'] == 1].reset_index(drop=True)
    return pos


def subsample_by_isolate(pos: pd.DataFrame, n_isolates: int,
                          seed: int = 0) -> pd.DataFrame:
    """Keep all positive pairs whose assembly_id_a OR assembly_id_b is in the
    sampled isolate set. Random selection seeded for reproducibility.
    """
    # In schema_ordered HA/NA, assembly_id_a == assembly_id_b for a given pair
    # (both proteins from the same isolate). Sanity-check that.
    mismatched = (pos['assembly_id_a'] != pos['assembly_id_b']).sum()
    if mismatched:
        print(f"  NOTE: {mismatched:,} pairs have assembly_id_a != assembly_id_b "
              f"(schema_ordered violation); treating each as a separate isolate.")
    all_isolates = pd.Index(pos['assembly_id_a']).union(pos['assembly_id_b']).unique()
    rng = pd.Series(all_isolates).sample(
        n=min(n_isolates, len(all_isolates)),
        random_state=seed,
    )
    keep = set(rng.values)
    mask = pos['assembly_id_a'].isin(keep) | pos['assembly_id_b'].isin(keep)
    return pos[mask].reset_index(drop=True)


def build_datasail_inputs(pos: pd.DataFrame, tmpdir: Path,
                           ) -> Tuple[Dict[str, str], Dict[str, str], Path]:
    """Build {seq_hash: seq} dicts for slot_a and slot_b, plus the inter.tsv.

    Use seq_hash as the entity identifier. DataSAIL deduplicates within an
    e_data / f_data dict by key, so identical sequences across pairs map to
    one entity automatically.
    """
    e_data = dict(zip(pos['seq_hash_a'], pos['seq_a']))  # slot_a proteins
    f_data = dict(zip(pos['seq_hash_b'], pos['seq_b']))  # slot_b proteins
    inter_pairs = list(zip(pos['seq_hash_a'], pos['seq_hash_b']))
    inter_tsv = tmpdir / 'inter.tsv'
    inter_tsv.write_text('\n'.join(f"{a}\t{b}" for a, b in inter_pairs))
    return e_data, f_data, inter_tsv


def run_datasail(e_data: Dict[str, str], f_data: Dict[str, str],
                 inter_tsv: Path, output: Path,
                 techniques: List[str],
                 e_clusters: int, f_clusters: int,
                 splits: List[float], names: List[str],
                 epsilon: float, delta: float,
                 solver: str, max_sec: int,
                 ) -> dict:
    """Thin wrapper around `datasail()`. Returns dict with the three result
    dicts and basic timing.
    """
    t0 = time.time()
    e_splits, f_splits, inter_splits = datasail(
        techniques=techniques,
        splits=splits,
        names=names,
        inter=str(inter_tsv),
        output=str(output),
        e_type='P', e_data=e_data, e_sim='mmseqs', e_clusters=e_clusters,
        f_type='P', f_data=f_data, f_sim='mmseqs', f_clusters=f_clusters,
        solver=solver, max_sec=max_sec, verbose='I',
        epsilon=epsilon, delta=delta,
    )
    return {
        'e_splits': e_splits,
        'f_splits': f_splits,
        'inter_splits': inter_splits,
        'wall_seconds': time.time() - t0,
    }


def summarize(inter_splits_for_technique: dict, n_input: int) -> dict:
    """Compute drop rate and fold sizes from one technique's result dict."""
    fold_counts = {'train': 0, 'val': 0, 'test': 0, 'not selected': 0}
    for _, fold in inter_splits_for_technique.items():
        fold_counts[fold] = fold_counts.get(fold, 0) + 1
    n_assigned = sum(v for k, v in fold_counts.items() if k != 'not selected')
    n_dropped = fold_counts.get('not selected', 0)
    n_missing = n_input - (n_assigned + n_dropped)  # pairs not in dict at all
    summary = {
        'n_input_pairs': n_input,
        'n_in_result_dict': n_assigned + n_dropped,
        'n_missing_from_dict': n_missing,
        'n_assigned_to_split': n_assigned,
        'n_dropped_not_selected': n_dropped,
        'drop_rate': round(n_dropped / max(n_input, 1), 4),
        'fold_sizes': {k: v for k, v in fold_counts.items() if k in ('train', 'val', 'test')},
        'fold_fractions': {
            k: round(v / max(n_assigned, 1), 4)
            for k, v in fold_counts.items() if k in ('train', 'val', 'test')
        },
    }
    return summary


def write_fold_csv(pos: pd.DataFrame, inter_splits_for_technique: dict,
                    out_csv: Path) -> None:
    """Emit one row per input positive, with `datasail_fold` column."""
    # The inter_splits dict is keyed by (seq_hash_a, seq_hash_b) tuples.
    fold_lookup = {(a, b): fold for (a, b), fold in inter_splits_for_technique.items()}
    pos_out = pos.copy()
    pos_out['datasail_fold'] = [
        fold_lookup.get((a, b), 'missing')
        for a, b in zip(pos_out['seq_hash_a'], pos_out['seq_hash_b'])
    ]
    pos_out.to_csv(out_csv, index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--dataset_dir', required=True, type=Path,
                   help='Stage 3 dataset directory (contains train/val/test_pairs.csv).')
    p.add_argument('--n_isolates', type=int, default=None,
                   help='Subsample to N isolates. Omit to use all positives.')
    p.add_argument('--techniques', nargs='+', default=['I2'],
                   choices=['R', 'I1', 'I2', 'S1', 'S2'],
                   help='DataSAIL split techniques to compute (one or more).')
    p.add_argument('--e_clusters', type=int, default=50, help='K for slot_a (e) clustering.')
    p.add_argument('--f_clusters', type=int, default=50, help='K for slot_b (f) clustering.')
    p.add_argument('--splits', nargs=3, type=float, default=[0.8, 0.1, 0.1],
                   metavar=('TRAIN', 'VAL', 'TEST'))
    p.add_argument('--epsilon', type=float, default=0.05,
                   help='Split-fraction relative tolerance (DataSAIL default 0.05).')
    p.add_argument('--delta', type=float, default=0.05,
                   help='Stratification relative tolerance (DataSAIL default 0.05).')
    p.add_argument('--solver', default='SCIP', choices=['SCIP', 'MOSEK', 'GUROBI'])
    p.add_argument('--max_sec', type=int, default=1000,
                   help='Per-technique time limit in seconds (DataSAIL default).')
    p.add_argument('--seed', type=int, default=0, help='Subsample seed.')
    p.add_argument('--output_dir', required=True, type=Path)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading positives from {args.dataset_dir} ...")
    pos = load_positives(args.dataset_dir)
    n_all = len(pos)
    print(f"  Loaded {n_all:,} positives.")

    if args.n_isolates is not None:
        pos = subsample_by_isolate(pos, args.n_isolates, seed=args.seed)
        print(f"  Subsampled to {len(pos):,} positives from "
              f"{args.n_isolates} isolates (seed={args.seed}).")

    e_data, f_data, inter_tsv = build_datasail_inputs(pos, args.output_dir)
    print(f"  Built inputs: {len(e_data):,} unique slot_a proteins, "
          f"{len(f_data):,} unique slot_b proteins, "
          f"{len(pos):,} interactions.")

    print(f"Running DataSAIL with techniques={args.techniques}, "
          f"K_e={args.e_clusters}, K_f={args.f_clusters}, "
          f"epsilon={args.epsilon}, delta={args.delta}, solver={args.solver} ...")
    result = run_datasail(
        e_data=e_data, f_data=f_data, inter_tsv=inter_tsv,
        output=args.output_dir / 'datasail_out',
        techniques=args.techniques,
        e_clusters=args.e_clusters, f_clusters=args.f_clusters,
        splits=args.splits, names=['train', 'val', 'test'],
        epsilon=args.epsilon, delta=args.delta,
        solver=args.solver, max_sec=args.max_sec,
    )
    print(f"  DataSAIL finished in {result['wall_seconds']:.1f}s.")

    audit = {
        'config': {
            'dataset_dir': str(args.dataset_dir),
            'n_input_positives': n_all,
            'n_used_positives': len(pos),
            'n_isolates_requested': args.n_isolates,
            'techniques': args.techniques,
            'e_clusters': args.e_clusters,
            'f_clusters': args.f_clusters,
            'splits_target': args.splits,
            'epsilon': args.epsilon, 'delta': args.delta,
            'solver': args.solver, 'max_sec': args.max_sec,
            'seed': args.seed,
        },
        'wall_seconds': result['wall_seconds'],
        'per_technique': {},
    }

    for tech in args.techniques:
        # The dicts come back as a list-per-run; we have runs=1 so take [0].
        e_split = result['e_splits'].get(tech, [{}])[0]
        f_split = result['f_splits'].get(tech, [{}])[0]
        inter_split = result['inter_splits'].get(tech, [{}])[0]

        summary = summarize(inter_split, n_input=len(pos))
        summary['n_unique_e_assigned'] = len([v for v in e_split.values() if v != 'not selected'])
        summary['n_unique_f_assigned'] = len([v for v in f_split.values() if v != 'not selected'])
        audit['per_technique'][tech] = summary

        out_csv = args.output_dir / f"positives_with_{tech}_folds.csv"
        write_fold_csv(pos, inter_split, out_csv)
        print(f"  [{tech}] wrote {out_csv.name}: "
              f"{summary['n_assigned_to_split']:,} assigned "
              f"({summary['drop_rate']*100:.1f}% dropped). "
              f"Fold sizes: {summary['fold_sizes']}")

    audit_path = args.output_dir / 'datasail_audit.json'
    audit_path.write_text(json.dumps(audit, indent=2))
    print(f"\nWrote audit: {audit_path}")
    print(f"Done.")


if __name__ == '__main__':
    main()
