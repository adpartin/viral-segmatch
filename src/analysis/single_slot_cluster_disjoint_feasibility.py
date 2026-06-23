"""Pre-flight feasibility for single-slot cluster_disjoint routing.

Sibling to `cluster_disjoint_feasibility.py`. The bilateral
cluster_disjoint constraint (both slots' clusters disjoint between
splits) collapses HA-NA to id100/id099-only on Flu A full corpus.
Single-slot relaxes this — only ONE slot's clusters must be disjoint
between splits, the other side is unconstrained. The "atom" that
cannot be split is then a single-slot cluster + all the pairs whose
that-slot protein belongs to that cluster.

Per schema_pair × alphabet × threshold × slot:
  largest_atom_pct = (# pairs whose slot-X protein is in the largest
                      slot-X cluster at this threshold) / n_pairs
  feasible_8010    = (largest_atom_pct <= 80%) AND (second <= 20%)
  max_feasible_k_strict           = floor(1 / max_atom_frac), capped
                                    at n_atoms. Zero-drift necessary
                                    condition for k-fold CV.
  max_feasible_k_at_default_drift = floor(1 / (max_atom_frac - drift)),
                                    capped at n_atoms, with
                                    drift = DEFAULT_DRIFT_PP = 0.05.
                                    Drift-aware necessary condition
                                    at deployed defaults (per D2/D3 of
                                    docs/plans/done/2026-05-27_kfold_variance_estimation_plan.md).

Per-fold drift check at build time is authoritative; the two
max_feasible_k columns bracket the common cases (drift_pp = 0 and
drift_pp = 0.05 default).

Reuses build_isolate_pairs + cluster_lookup loader from the bilateral
script so the dedup, hashing, and cluster artifact paths match exactly.

CLI:
    # aa single-slot feasibility on HA-NA at multiple thresholds:
    python -m src.analysis.single_slot_cluster_disjoint_feasibility \\
        --protein_final data/processed/flu/July_2025/protein_final.parquet \\
        --clusters_root data/processed/flu/July_2025/clusters_aa \\
        --schema_pair "Hemagglutinin precursor" "Neuraminidase protein" \\
        --thresholds 1.00 0.99 0.98 0.97 0.96 0.95 0.90 0.85

    # nt single-slot feasibility:
    python -m src.analysis.single_slot_cluster_disjoint_feasibility \\
        --cds_final     data/processed/flu/July_2025/cds_final.parquet \\
        --clusters_root data/processed/flu/July_2025/clusters_nt \\
        --schema_pair "Hemagglutinin precursor" "Neuraminidase protein" \\
        --thresholds 1.00 0.99 0.98 0.97 0.96 0.95 0.90 0.85
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.cluster_disjoint_feasibility import (
    build_isolate_pairs, FUNCTION_TO_SHORT, _threshold_label,
)

# Keep in sync with conf/dataset/default.yaml::split_strategy.feasibility.max_acceptable_drift_pp
# (default value used by D3 of the k-fold variance plan).
DEFAULT_DRIFT_PP = 0.05


def _load_cluster_lookup(clusters_root: Path, threshold: float,
                          schema_pair: Tuple[str, str]) -> pd.DataFrame:
    """Load combined prot_hash -> cluster_id for the two slot functions at threshold."""
    label = _threshold_label(threshold)
    threshold_dir = clusters_root / label
    parts = []
    for f in schema_pair:
        short = FUNCTION_TO_SHORT.get(f)
        if short is None:
            raise KeyError(f"Function not in mapping: {f}")
        p = threshold_dir / f"{short}_cluster.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing cluster parquet: {p}")
        parts.append(pd.read_parquet(p))
    return pd.concat(parts, ignore_index=True)[['prot_hash', 'cluster_id']]


def feasibility_single_slot(isolate_pairs: pd.DataFrame,
                              cluster_lookup: pd.DataFrame,
                              threshold: float, slot: str) -> dict:
    """Single-slot cluster_disjoint atom-size stats for one slot.

    `slot='a'` constrains slot a (the first function in schema_pair, e.g. HA).
    `slot='b'` constrains slot b (e.g. NA). The atom is "all pairs whose
    slot-X protein lives in slot-X cluster K"; the largest atom is the
    quantity that gates 80/10/10.
    """
    if slot not in ('a', 'b'):
        raise ValueError(f"slot must be 'a' or 'b', got {slot}")
    other = 'b' if slot == 'a' else 'a'

    # Same dedup contract as the bilateral script: join cluster_ids onto both
    # slots, dedup on (prot_hash_a, prot_hash_b) so we count what v2 actually routes.
    lookup_a = cluster_lookup.rename(columns={'prot_hash': 'prot_hash_a',
                                              'cluster_id': 'cluster_id_a'})
    lookup_b = cluster_lookup.rename(columns={'prot_hash': 'prot_hash_b',
                                              'cluster_id': 'cluster_id_b'})
    pairs = isolate_pairs.merge(lookup_a, on='prot_hash_a', how='left')
    pairs = pairs.merge(lookup_b, on='prot_hash_b', how='left')
    n_dropped = int(pairs['cluster_id_a'].isna().sum() + pairs['cluster_id_b'].isna().sum())
    pairs = pairs.dropna(subset=['cluster_id_a', 'cluster_id_b']).reset_index(drop=True)
    pairs = pairs.drop_duplicates(subset=['prot_hash_a', 'prot_hash_b']).reset_index(drop=True)
    n_pairs = len(pairs)

    if n_pairs == 0:
        return {
            'threshold': threshold, 'slot': slot, 'n_pairs': 0,
            'n_clusters_constrained': 0, 'largest_atom_pct': 0.0,
            'second_atom_pct': 0.0, 'top5_atom_pct': [],
            'singleton_atoms': 0,
            'max_feasible_k_strict': 0,
            'max_feasible_k_at_default_drift': 0,
            'feasible_8010': False,
            'n_dropped_in_join': n_dropped,
        }

    # Atom = pairs grouped by the constrained-slot cluster_id only.
    constrained_col = f'cluster_id_{slot}'
    atom_sizes = pairs.groupby(constrained_col).size().sort_values(ascending=False).values
    atom_pct = atom_sizes / n_pairs * 100.0
    n_atoms = int(len(atom_sizes))

    largest_pct = float(atom_pct[0])
    second_pct = float(atom_pct[1]) if len(atom_pct) > 1 else 0.0
    feasible = (largest_pct <= 80.0) and (second_pct <= 20.0)

    # max_feasible_k_{strict, at_default_drift}: see module docstring.
    # When max_atom_frac <= drift, the test bin absorbs the largest atom at
    # any k; the constraint is non-binding so cap at n_atoms (can't have
    # more folds than atoms).
    max_atom_frac = largest_pct / 100.0
    if max_atom_frac > 0:
        max_feasible_k_strict = min(int(np.floor(1.0 / max_atom_frac)), n_atoms)
    else:
        max_feasible_k_strict = n_atoms
    if max_atom_frac > DEFAULT_DRIFT_PP:
        max_feasible_k_at_default_drift = min(
            int(np.floor(1.0 / (max_atom_frac - DEFAULT_DRIFT_PP))), n_atoms,
        )
    else:
        max_feasible_k_at_default_drift = n_atoms

    return {
        'threshold': threshold,
        'slot': slot,
        'n_pairs': n_pairs,
        'n_clusters_constrained': n_atoms,
        'largest_atom_pct': round(largest_pct, 2),
        'second_atom_pct': round(second_pct, 2),
        'top5_atom_pct': [round(float(p), 2) for p in atom_pct[:5]],
        'singleton_atoms': int((np.array(atom_sizes) == 1).sum()),
        'max_feasible_k_strict': max_feasible_k_strict,
        'max_feasible_k_at_default_drift': max_feasible_k_at_default_drift,
        'feasible_8010': bool(feasible),
        'n_dropped_in_join': n_dropped,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--protein_final', type=Path,
                     help='aa mode: protein_final.parquet from Stage 1.')
    src.add_argument('--cds_final', type=Path,
                     help='nt mode: cds_final.parquet from Stage 1.5.')
    ap.add_argument('--clusters_root', required=True, type=Path,
                    help='Directory with idNN/<SHORT>_cluster.parquet artifacts.')
    ap.add_argument('--schema_pair', required=True, nargs=2, metavar=('FUNC_A', 'FUNC_B'),
                    help='Slot-a and slot-b function names (must match Stage 1 names).')
    ap.add_argument('--thresholds', type=float, nargs='+',
                    default=[1.00, 0.99, 0.98, 0.97, 0.96, 0.95, 0.90, 0.85])
    ap.add_argument('--out_csv', type=Path, default=None,
                    help='Output CSV. Defaults to results/.../single_slot_feasibility_<pair>_<alphabet>.csv')
    args = ap.parse_args()

    if args.protein_final is not None:
        alphabet = 'aa'
        df = pd.read_parquet(args.protein_final)
    else:
        alphabet = 'nt'
        df = pd.read_parquet(args.cds_final)

    pair_label = '_'.join(FUNCTION_TO_SHORT.get(f, f).lower() for f in args.schema_pair)
    if args.out_csv is None:
        args.out_csv = (PROJECT_ROOT / 'results' / 'flu' / 'July_2025' / 'runs'
                        / 'cluster_disjoint_feasibility'
                        / f'single_slot_feasibility_{pair_label}_{alphabet}.csv')
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    isolate_pairs = build_isolate_pairs(df, tuple(args.schema_pair), alphabet=alphabet)
    print(f'Schema pair: {args.schema_pair[0]!r} <-> {args.schema_pair[1]!r}  '
          f'({pair_label}, alphabet={alphabet})')
    print(f'Isolate pairs: {len(isolate_pairs):,}  '
          f'(unique prot_hash_a={isolate_pairs["prot_hash_a"].nunique():,}, '
          f'unique prot_hash_b={isolate_pairs["prot_hash_b"].nunique():,})')
    print()

    rows = []
    for t in args.thresholds:
        cluster_lookup = _load_cluster_lookup(args.clusters_root, t, tuple(args.schema_pair))
        for slot in ('a', 'b'):
            row = feasibility_single_slot(isolate_pairs, cluster_lookup, t, slot)
            slot_name = FUNCTION_TO_SHORT.get(args.schema_pair[0 if slot == 'a' else 1], slot)
            verdict = 'feasible' if row['feasible_8010'] else 'DEGENERATE'
            print(f"=== threshold={t:.2f}  constrain={slot}({slot_name}) ===")
            print(f"  n_pairs={row['n_pairs']:,}  "
                  f"n_clusters={row['n_clusters_constrained']:,}  "
                  f"largest={row['largest_atom_pct']:.2f}%  "
                  f"second={row['second_atom_pct']:.2f}%  "
                  f"max_k_strict={row['max_feasible_k_strict']}  "
                  f"max_k_at_default_drift={row['max_feasible_k_at_default_drift']}  "
                  f"top5={row['top5_atom_pct']}  -> {verdict}")
            rows.append(row)
        print()

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    print('Note: max_feasible_k columns are necessary conditions only — '
          'per-fold drift check at build time is authoritative. '
          f'max_feasible_k_at_default_drift uses drift_pp={DEFAULT_DRIFT_PP} '
          '(D3 default). For non-default drift configs, re-derive from '
          'max_atom_frac (= largest_atom_pct / 100).')
    print()
    print('Summary:')
    cols = ['threshold', 'slot', 'n_pairs', 'n_clusters_constrained',
            'largest_atom_pct', 'second_atom_pct', 'top5_atom_pct',
            'singleton_atoms', 'max_feasible_k_strict',
            'max_feasible_k_at_default_drift', 'feasible_8010']
    print(out_df[cols].to_string(index=False))
    print(f'\nWrote: {args.out_csv}')


if __name__ == '__main__':
    main()
