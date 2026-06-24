"""Pre-flight feasibility check for cluster_disjoint routing.

Per-function cluster sizes (`build_mmseqs_clusters.py`) are necessary
but NOT sufficient: cluster_disjoint routes pairs by the bipartite-component
structure of (slot_a_cluster, slot_b_cluster). HA clusters and NA clusters
connect through isolates that have proteins in both, so the **bipartite
component** that the routing actually operates on can be much larger than
either side's biggest cluster — sometimes catastrophically so at low identity
thresholds.

This script answers: given a schema_pair and a cluster threshold, what does the
bipartite-component-size distribution look like? If the largest component
exceeds 80% of pairs, the partition is forced and 80/10/10 is unachievable.

Supports all three alphabets (slot key = the registry hash; cluster lookups under
`<clusters_root>/t<NN>/`):
  - aa via `--protein_final`: hashes `prot_seq` with md5 to match `prot_hash`,
    joins aa clusters (clusters_aa) on `prot_hash`.
  - nt_cds via `--cds_dna_final`: uses `cds_dna_hash` (already in cds_dna_final),
    joins nt_cds clusters (clusters_nt_cds) on `cds_dna_hash`.
  - nt_ctg via `--ctg_dna_final`: uses `ctg_dna_hash` (already in ctg_dna_final),
    joins nt_ctg clusters (clusters_nt_ctg) on `ctg_dna_hash`. ctg_dna_final
    carries no `function`; it is attached via a 1-1 join from `--function_source`
    (default: the sibling cds_dna_final.parquet).

The input flags are mutually exclusive; `--alphabet` is inferred from which is
given unless overridden.

CLI:
    # aa feasibility (HA/NA). --out_csv default basename: feasibility_ha_na_aa.csv
    python -m src.analysis.cluster_disjoint_feasibility \\
        --protein_final data/processed/flu/July_2025/protein_final.parquet \\
        --clusters_root data/processed/flu/July_2025/clusters_aa \\
        --schema_pair "Hemagglutinin precursor" "Neuraminidase protein" \\
        --thresholds 1.00 0.99 0.98 0.97 0.96 0.95 0.90

    # nt_ctg feasibility (HA/NA on contig DNA).
    python -m src.analysis.cluster_disjoint_feasibility \\
        --ctg_dna_final data/processed/flu/July_2025/ctg_dna_final.parquet \\
        --clusters_root data/processed/flu/July_2025/clusters_nt_ctg \\
        --schema_pair "Hemagglutinin precursor" "Neuraminidase protein" \\
        --thresholds 1.00 0.99 0.98 0.97 0.96 0.95 0.90

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
from src.utils import schema
from src.utils.clustering_utils import attach_function_to_contigs


def _threshold_label(t: float) -> str:
    return f"t{int(round(t * 100)):03d}"


def build_isolate_pairs(
    df: pd.DataFrame,
    schema_pair: Tuple[str, str],
    alphabet: str = 'aa',
) -> pd.DataFrame:
    """Return a DataFrame with one row per isolate that has BOTH proteins of schema_pair.

    Columns: assembly_id, hash_a, hash_b (slot a = func_left = schema_pair[0];
    slot b = func_right = schema_pair[1]). The slot hash is the alphabet's
    cluster-join hash from the schema registry — `prot_hash` (aa, md5 of
    prot_seq), `cds_dna_hash` (nt_cds), `ctg_dna_hash` (nt_ctg) — renamed to the
    neutral `hash_{a,b}`. The cluster lookup is keyed on the SAME hash, so the
    bipartite components are computed on the alphabet's own hash space.

    Note (nt_ctg): this dedups isolate-pairs on `ctg_dna_hash` (the cluster
    hash), i.e. distinct contigs are distinct routing atoms. v2's nt_ctg pair_key
    dedups on `prot_hash` instead (the pair_key_alphabet inference falls to 'aa'
    for nt_ctg) — so this pre-flight is the cluster-faithful view; the bipartite
    component TOPOLOGY (the operability signal) is the same, only per-component
    pair COUNTS differ from v2's deduped pos_df.

    Input df must contain 'function', 'assembly_id', plus:
      - aa: 'prot_seq' (hashed here to prot_hash);
      - nt_cds / nt_ctg: the registry hash column, already present.
    """
    func_left, func_right = schema_pair
    hash_col = schema.hash_col(alphabet)
    sub = df[df['function'].isin([func_left, func_right])].copy()
    if alphabet == 'aa':
        sub[hash_col] = sub['prot_seq'].astype(str).map(
            lambda s: hashlib.md5(s.encode()).hexdigest()
        )
    elif hash_col not in sub.columns:
        raise ValueError(
            f"alphabet={alphabet!r} requires the {hash_col!r} column on the input df"
        )
    a = sub[sub['function'] == func_left][['assembly_id', hash_col]].rename(
        columns={hash_col: 'hash_a'})
    b = sub[sub['function'] == func_right][['assembly_id', hash_col]].rename(
        columns={hash_col: 'hash_b'})
    a = a.drop_duplicates('assembly_id')
    b = b.drop_duplicates('assembly_id')
    merged = a.merge(b, on='assembly_id', how='inner')
    return merged


def load_cluster_lookup_for_schema(
    clusters_root: Path,
    threshold: float,
    schema_pair: Tuple[str, str],
    function_to_short: dict,
    alphabet: str = 'aa',
) -> pd.DataFrame:
    """Load the per-function cluster parquets for the schema_pair's two functions
    at the given threshold; return a single (hash, cluster_id) lookup keyed on the
    alphabet's registry hash column (prot_hash / cds_dna_hash / ctg_dna_hash),
    renamed to the neutral `hash`."""
    threshold_dir = Path(clusters_root) / _threshold_label(threshold)
    hash_col = schema.hash_col(alphabet)
    func_left, func_right = schema_pair
    parts = []
    for f in (func_left, func_right):
        short = function_to_short.get(f)
        if short is None:
            raise KeyError(f"Function not in mapping: {f}")
        p = threshold_dir / f"{short}_cluster.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing cluster parquet: {p}")
        parts.append(pd.read_parquet(p, columns=[hash_col, 'cluster_id']))
    out = pd.concat(parts, ignore_index=True)
    return out.rename(columns={hash_col: 'hash'})


def feasibility_for_threshold(
    isolate_pairs: pd.DataFrame,
    cluster_lookup: pd.DataFrame,
    threshold: float,
) -> dict:
    """Compute bipartite-component stats on (cluster_id_a, cluster_id_b) AFTER
    deduping on (hash_a, hash_b) — to match v2's actual routing input (Stage 3
    dedups pair_keys globally before split_dataset_v2 routes them). The slot hash
    is the alphabet's cluster hash (prot_hash / cds_dna_hash / ctg_dna_hash)."""
    # Join cluster_ids onto both slots
    lookup_a = cluster_lookup.rename(columns={'hash': 'hash_a',
                                              'cluster_id': 'cluster_id_a'})
    lookup_b = cluster_lookup.rename(columns={'hash': 'hash_b',
                                              'cluster_id': 'cluster_id_b'})
    pairs = isolate_pairs.merge(lookup_a, on='hash_a', how='left')
    pairs = pairs.merge(lookup_b, on='hash_b', how='left')
    n_dropped = int(pairs['cluster_id_a'].isna().sum() + pairs['cluster_id_b'].isna().sum())
    pairs = pairs.dropna(subset=['cluster_id_a', 'cluster_id_b']).reset_index(drop=True)

    # Dedup to one row per unique (hash_a, hash_b) — this matches the deduped
    # pos_df that Stage 3 passes to the routing helper. Without this dedup the
    # analysis OVER-COUNTS the largest component because many isolates can share
    # the same sequence pair (especially for conserved functions like PB2/PB1).
    n_raw_pairs = len(pairs)
    pairs = pairs.drop_duplicates(subset=['hash_a', 'hash_b']).reset_index(drop=True)
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
    src.add_argument('--cds_dna_final', help='nt_cds-mode input: path to cds_dna_final.parquet.')
    src.add_argument('--ctg_dna_final', help='nt_ctg-mode input: path to ctg_dna_final.parquet '
                     '(`function` attached via a 1-1 join from --function_source).')
    p.add_argument('--function_source', default=None,
                   help='nt_ctg only: [assembly_id, genbank_ctg_id, function] source for the '
                        'contig->function join (default: sibling cds_dna_final.parquet).')
    p.add_argument('--alphabet', choices=['aa', 'nt_cds', 'nt_ctg'], default=None,
                   help='Sequence alphabet (default: inferred from the input flag).')
    p.add_argument('--clusters_root', required=True,
                   help='Root of cluster artifacts '
                        '(clusters_aa / clusters_nt_cds / clusters_nt_ctg).')
    p.add_argument('--schema_pair', nargs=2, required=True,
                   help='Two function names (slot_a, slot_b).')
    p.add_argument('--thresholds', nargs='+', type=float, required=True)
    p.add_argument('--out_csv', default=None,
                   help='Output CSV path. If omitted, defaults to '
                        'results/flu/July_2025/runs/cluster_disjoint_feasibility/'
                        'feasibility_<pair_short>_<alphabet>.csv. Pair short '
                        'is derived from the schema_pair via FUNCTION_TO_SHORT '
                        '(e.g., HA + NA -> "ha_na").')
    args = p.parse_args()

    if args.protein_final:
        args.alphabet = args.alphabet or 'aa'
        if args.alphabet != 'aa':
            raise SystemExit("--protein_final is aa-only; use --cds_dna_final / --ctg_dna_final.")
    elif args.cds_dna_final:
        args.alphabet = args.alphabet or 'nt_cds'
        if args.alphabet != 'nt_cds':
            raise SystemExit("--cds_dna_final is nt_cds-only.")
    else:  # args.ctg_dna_final
        args.alphabet = args.alphabet or 'nt_ctg'
        if args.alphabet != 'nt_ctg':
            raise SystemExit("--ctg_dna_final is nt_ctg-only.")

    in_path = args.protein_final or args.cds_dna_final or args.ctg_dna_final
    print(f"Loading {in_path}  (alphabet={args.alphabet}) ...")
    if args.alphabet == 'aa':
        df = pd.read_parquet(in_path, columns=['assembly_id', 'function', 'prot_seq'])
    elif args.alphabet == 'nt_cds':
        df = pd.read_parquet(in_path, columns=['assembly_id', 'function', 'cds_dna_hash'])
    else:  # nt_ctg: ctg_dna_final has no `function`; attach it via the 1-1 join.
        df = pd.read_parquet(in_path, columns=['assembly_id', 'genbank_ctg_id', 'ctg_dna_hash'])
        fsrc = (Path(args.function_source) if args.function_source
                else Path(in_path).with_name('cds_dna_final.parquet'))
        if not fsrc.exists():
            raise SystemExit(
                f"nt_ctg needs a --function_source for the contig->function join; "
                f"default sibling not found: {fsrc}")
        df = attach_function_to_contigs(df, fsrc).drop(columns=['genbank_ctg_id'])
    print(f"  {len(df):,} rows")

    schema_pair = tuple(args.schema_pair)
    print(f"\nschema_pair = {schema_pair}")
    isolate_pairs = build_isolate_pairs(df, schema_pair, alphabet=args.alphabet)
    print(f"  Isolates with both proteins: {len(isolate_pairs):,}")
    print(f"  Unique hash_a: {isolate_pairs['hash_a'].nunique():,}")
    print(f"  Unique hash_b: {isolate_pairs['hash_b'].nunique():,}")

    rows = []
    for threshold in args.thresholds:
        print(f"\n=== threshold = {threshold:.2f} ===")
        try:
            lookup = load_cluster_lookup_for_schema(
                args.clusters_root, threshold, schema_pair, FUNCTION_TO_SHORT,
                alphabet=args.alphabet,
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

    if args.out_csv is None:
        try:
            shorts = [FUNCTION_TO_SHORT[f] for f in schema_pair]
        except KeyError as e:
            raise SystemExit(
                f"Cannot derive default --out_csv: schema_pair entry not in "
                f"FUNCTION_TO_SHORT: {e}. Pass --out_csv explicitly."
            )
        pair_short = '_'.join(s.lower() for s in shorts)
        args.out_csv = str(
            PROJECT_ROOT / 'results' / 'flu' / 'July_2025' / 'runs'
            / 'cluster_disjoint_feasibility'
            / f'feasibility_{pair_short}_{args.alphabet}.csv'
        )
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    print(f"\nWrote: {args.out_csv}")


if __name__ == '__main__':
    main()
