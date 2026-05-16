"""Pre-flight feasibility check for cluster_disjoint routing.

Per-function cluster sizes (`seq_redundancy_per_function.py`) are necessary
but NOT sufficient: cluster_disjoint routes pairs by the bipartite-component
structure of (slot_a_cluster, slot_b_cluster). HA clusters and NA clusters
connect through isolates that have proteins in both, so the **bipartite
component** that the routing actually operates on can be much larger than
either side's biggest cluster — sometimes catastrophically so at low identity
thresholds.

This script answers: given a schema_pair and a cluster threshold, what does the
bipartite-component-size distribution look like? If the largest component
exceeds 80% of pairs, the partition is forced and 80/10/10 is unachievable.

Supports both alphabets:
  - aa via `--protein_final`: hashes `prot_seq` with md5 to match Stage 1's
    `seq_hash`, joins to aa cluster lookups under `<clusters_root>/id<NN>/`.
  - nt via `--cds_final`: uses `cds_dna_hash` from `cds_final.parquet`
    (Stage 1.5 output) as the slot key, joins to nt cluster lookups under
    `<clusters_root>/id<NN>/` (typically `clusters_nt/`).

The two modes are mutually exclusive; `--alphabet` is inferred from which
input is provided unless overridden.

CLI:
    # aa feasibility (HA/NA)
    python -m src.analysis.cluster_disjoint_feasibility \\
        --protein_final data/processed/flu/July_2025/protein_final.parquet \\
        --clusters_root data/processed/flu/July_2025/clusters \\
        --schema_pair "Hemagglutinin precursor" "Neuraminidase protein" \\
        --thresholds 1.00 0.99 0.95 0.90 0.80 \\
        --out_csv docs/results/2026-05-14_cluster_disjoint_feasibility_ha_na.csv

    # nt feasibility (HA/NA on CDS DNA)
    python -m src.analysis.cluster_disjoint_feasibility \\
        --cds_final     data/processed/flu/July_2025/cds_final.parquet \\
        --clusters_root data/processed/flu/July_2025/clusters_nt \\
        --schema_pair "Hemagglutinin precursor" "Neuraminidase protein" \\
        --thresholds 1.00 0.99 0.95 0.90 0.85 0.80 \\
        --out_csv docs/results/2026-05-15_cluster_disjoint_feasibility_nt_ha_na.csv

Produces a per-threshold summary table:
    threshold  n_pairs  n_components  largest_pct  second_pct  p99_cumpct
    p90_cumpct  top5_pct  singleton_components  feasible_8010
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets._pair_helpers import bipartite_components


def _threshold_label(t: float) -> str:
    return f"id{int(round(t * 100)):03d}"


def build_isolate_pairs(
    df: pd.DataFrame,
    schema_pair: Tuple[str, str],
    alphabet: str = 'aa',
) -> pd.DataFrame:
    """Return a DataFrame with one row per isolate that has BOTH proteins of schema_pair.

    Columns: assembly_id, seq_hash_a, seq_hash_b (matching the v2 pair convention:
    slot a = func_left = schema_pair[0]; slot b = func_right = schema_pair[1]).

    Hash semantics depend on `alphabet`:
        - 'aa' (default): `seq_hash` is `md5(prot_seq)` (matches Stage 1).
          Input df must contain 'function' + 'prot_seq'.
        - 'nt': `seq_hash` is the CDS DNA hash (i.e. the column populated by
          `extract_cds_dna.py` as `cds_dna_hash`). Input df must contain
          'function' + 'cds_dna_hash'. The output column is still named
          `seq_hash_*` so the cluster_lookup join (which is keyed on
          `seq_hash`) works without modification.
    """
    func_left, func_right = schema_pair
    sub = df[df['function'].isin([func_left, func_right])].copy()
    if alphabet == 'aa':
        sub['seq_hash'] = sub['prot_seq'].astype(str).map(
            lambda s: hashlib.md5(s.encode()).hexdigest()
        )
    elif alphabet == 'nt':
        if 'cds_dna_hash' not in sub.columns:
            raise ValueError(
                "alphabet='nt' requires 'cds_dna_hash' column "
                "(build cds_final via src/preprocess/extract_cds_dna.py)"
            )
        sub['seq_hash'] = sub['cds_dna_hash']
    else:
        raise ValueError(f"alphabet must be 'aa' or 'nt', got {alphabet!r}")
    a = sub[sub['function'] == func_left][['assembly_id', 'seq_hash']].rename(
        columns={'seq_hash': 'seq_hash_a'})
    b = sub[sub['function'] == func_right][['assembly_id', 'seq_hash']].rename(
        columns={'seq_hash': 'seq_hash_b'})
    a = a.drop_duplicates('assembly_id')
    b = b.drop_duplicates('assembly_id')
    merged = a.merge(b, on='assembly_id', how='inner')
    return merged


def load_cluster_lookup_for_schema(
    clusters_root: Path,
    threshold: float,
    schema_pair: Tuple[str, str],
    function_to_short: dict,
) -> pd.DataFrame:
    """Load the per-function cluster parquets for the schema_pair's two functions
    at the given threshold; return a single (seq_hash, cluster_id) lookup."""
    threshold_dir = Path(clusters_root) / _threshold_label(threshold)
    func_left, func_right = schema_pair
    parts = []
    for f in (func_left, func_right):
        short = function_to_short.get(f)
        if short is None:
            raise KeyError(f"Function not in mapping: {f}")
        p = threshold_dir / f"{short}_cluster.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing cluster parquet: {p}")
        parts.append(pd.read_parquet(p))
    return pd.concat(parts, ignore_index=True)[['seq_hash', 'cluster_id']]


def feasibility_for_threshold(
    isolate_pairs: pd.DataFrame,
    cluster_lookup: pd.DataFrame,
    threshold: float,
) -> dict:
    """Compute bipartite-component stats on (cluster_id_a, cluster_id_b) AFTER
    deduping on (seq_hash_a, seq_hash_b) — to match v2's actual routing input
    (Stage 3 dedups pair_keys globally before split_dataset_v2 routes them)."""
    # Join cluster_ids onto both slots
    lookup_a = cluster_lookup.rename(columns={'seq_hash': 'seq_hash_a',
                                              'cluster_id': 'cluster_id_a'})
    lookup_b = cluster_lookup.rename(columns={'seq_hash': 'seq_hash_b',
                                              'cluster_id': 'cluster_id_b'})
    pairs = isolate_pairs.merge(lookup_a, on='seq_hash_a', how='left')
    pairs = pairs.merge(lookup_b, on='seq_hash_b', how='left')
    n_dropped = int(pairs['cluster_id_a'].isna().sum() + pairs['cluster_id_b'].isna().sum())
    pairs = pairs.dropna(subset=['cluster_id_a', 'cluster_id_b']).reset_index(drop=True)

    # Dedup to one row per unique (seq_hash_a, seq_hash_b) — this matches
    # the deduped pos_df that Stage 3 passes to the routing helper. Without
    # this dedup the analysis OVER-COUNTS the largest component because many
    # isolates can share the same protein pair (especially for conserved
    # functions like PB2/PB1).
    n_raw_pairs = len(pairs)
    pairs = pairs.drop_duplicates(subset=['seq_hash_a', 'seq_hash_b']).reset_index(drop=True)
    n_pairs = len(pairs)

    if n_pairs == 0:
        return {
            'threshold': threshold, 'n_pairs': 0, 'n_components': 0,
            'largest_pct': 0.0, 'p99_pct': 0.0, 'p90_pct': 0.0,
            'top5_pct': [], 'feasible_8010': False, 'n_dropped_in_join': n_dropped,
            'n_raw_isolate_pairs': n_raw_pairs,
        }

    component_id, cc_summary = bipartite_components(
        pairs, col_a='cluster_id_a', col_b='cluster_id_b',
    )
    sizes = component_id.value_counts().sort_values(ascending=False).values
    sizes_pct = sizes / n_pairs * 100.0

    # 80/10/10 feasibility: every split must hold its full 10% target (5,839 pairs at n_pairs ~58K).
    # The component-packing problem is achievable iff the largest single component fits inside
    # the train target (80%). If even the second-largest is > 10%, val/test can't both reach 10%.
    largest_pct = float(sizes_pct[0])
    second_pct = float(sizes_pct[1]) if len(sizes_pct) > 1 else 0.0
    feasible = (largest_pct <= 80.0) and (second_pct <= 20.0)

    p_cum = np.cumsum(sizes_pct)
    p99_pct = float(p_cum[min(len(p_cum)-1, int(np.searchsorted(p_cum, 99.0)))])
    p90_pct = float(p_cum[min(len(p_cum)-1, int(np.searchsorted(p_cum, 90.0)))])

    return {
        'threshold': threshold,
        'n_pairs': n_pairs,                # deduped pos_df rows (what v2 actually routes)
        'n_raw_isolate_pairs': n_raw_pairs,  # isolates with both proteins (pre-dedup)
        'n_dropped_in_join': n_dropped,
        'n_components': int(cc_summary['n_components']),
        'largest_pct': round(largest_pct, 2),
        'second_pct': round(second_pct, 2),
        'p99_cumpct': round(p99_pct, 2),
        'p90_cumpct': round(p90_pct, 2),
        'top5_pct': [round(float(p), 2) for p in sizes_pct[:5]],
        'singleton_components': int((np.array(sizes) == 1).sum()),
        'feasible_8010': bool(feasible),
    }


FUNCTION_TO_SHORT = {
    'RNA-dependent RNA polymerase PB2 subunit': 'PB2',
    'RNA-dependent RNA polymerase catalytic core PB1 subunit': 'PB1',
    'RNA-dependent RNA polymerase PA subunit': 'PA',
    'Hemagglutinin precursor': 'HA',
    'Nucleocapsid protein': 'NP',
    'Neuraminidase protein': 'NA',
    'Matrix protein 1': 'M1',
    'M2 ion channel': 'M2',
    'Nuclear export protein': 'NEP',
    'Non-structural protein 1, interferon antagonist and host mRNA processing inhibitor': 'NS1',
}


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-flight bipartite-component feasibility for cluster_disjoint.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--protein_final', help='aa-mode input: path to protein_final.parquet.')
    src.add_argument('--cds_final',     help='nt-mode input: path to cds_final.parquet.')
    p.add_argument('--alphabet', choices=['aa', 'nt'], default=None,
                   help='Sequence alphabet (default: aa for --protein_final, nt for --cds_final).')
    p.add_argument('--clusters_root', required=True,
                   help='Root of cluster artifacts. For nt mode this should be the '
                        'clusters_nt/ directory built with --alphabet nt.')
    p.add_argument('--schema_pair', nargs=2, required=True,
                   help='Two function names (slot_a, slot_b).')
    p.add_argument('--thresholds', nargs='+', type=float, required=True)
    p.add_argument('--out_csv', default=None, help='Optional output CSV path.')
    args = p.parse_args()

    if args.protein_final and not args.alphabet:
        args.alphabet = 'aa'
    if args.cds_final and not args.alphabet:
        args.alphabet = 'nt'
    if args.protein_final and args.alphabet == 'nt':
        raise SystemExit("--protein_final is aa-only; use --cds_final for nt mode.")
    if args.cds_final and args.alphabet == 'aa':
        raise SystemExit("--cds_final is nt-only; use --protein_final for aa mode.")

    in_path = args.protein_final or args.cds_final
    print(f"Loading {in_path}  (alphabet={args.alphabet}) ...")
    if args.alphabet == 'aa':
        df = pd.read_parquet(in_path, columns=['assembly_id', 'function', 'prot_seq'])
    else:
        df = pd.read_parquet(in_path, columns=['assembly_id', 'function', 'cds_dna_hash'])
    print(f"  {len(df):,} rows")

    schema_pair = tuple(args.schema_pair)
    print(f"\nschema_pair = {schema_pair}")
    isolate_pairs = build_isolate_pairs(df, schema_pair, alphabet=args.alphabet)
    print(f"  Isolates with both proteins: {len(isolate_pairs):,}")
    print(f"  Unique seq_hash_a: {isolate_pairs['seq_hash_a'].nunique():,}")
    print(f"  Unique seq_hash_b: {isolate_pairs['seq_hash_b'].nunique():,}")

    rows = []
    for threshold in args.thresholds:
        print(f"\n=== threshold = {threshold:.2f} ===")
        try:
            lookup = load_cluster_lookup_for_schema(
                args.clusters_root, threshold, schema_pair, FUNCTION_TO_SHORT,
            )
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue
        stats = feasibility_for_threshold(isolate_pairs, lookup, threshold)
        rows.append(stats)
        feasible_tag = "FEASIBLE" if stats['feasible_8010'] else "DEGENERATE"
        print(f"  n_pairs={stats['n_pairs']:,}  n_components={stats['n_components']:,}  "
              f"largest={stats['largest_pct']:.1f}%  second={stats['second_pct']:.1f}%  "
              f"top5={stats['top5_pct']}  -> {feasible_tag}")

    df_out = pd.DataFrame(rows)
    print("\nSummary:")
    print(df_out.to_string(index=False))

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(args.out_csv, index=False)
        print(f"\nWrote: {args.out_csv}")


if __name__ == '__main__':
    main()
