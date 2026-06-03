"""Pre-dataset cluster-cluster coupling pre-check.

Computes Cramér's V between slot-a and slot-b cluster_ids (and between
each cluster_id and the whole-isolate H/N subtype) for a schema_pair
across a range of cluster thresholds. Works directly from
`protein_final.parquet` + cluster artifacts; does NOT need built
datasets.

Use case: before launching a single-slot cluster_disjoint sweep for a
new schema_pair, check whether the constrained-slot cluster boundary
implicitly drags the unconstrained slot with it (HA-NA-style
biological coupling) or whether the two slots are reasonably
independent. V(cluster_a × cluster_b) is the key quantity — if it
stays low across the sweep, single-slot routing genuinely decouples
the slots.

Cramér's V is computed via sparse arithmetic
(chi² = Σ_observed O² / E − N), so 10⁴ × 10⁴ crosstabs are no problem.

CLI:
    python -m src.analysis.cluster_pair_coupling_precheck \\
        --protein_final data/processed/flu/July_2025/protein_final.parquet \\
        --clusters_root data/processed/flu/July_2025/clusters_aa \\
        --schema_pair "RNA-dependent RNA polymerase PB2 subunit" \\
                      "RNA-dependent RNA polymerase catalytic core PB1 subunit" \\
        --thresholds 1.00 0.99 0.98 0.97 0.96 0.95
"""
from __future__ import annotations

import argparse
import re
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


_HN_RE = re.compile(r'H\d+N\d+')


def _load_cluster_lookup(clusters_root: Path, threshold: float,
                          schema_pair: Tuple[str, str]) -> pd.DataFrame:
    label = _threshold_label(threshold)
    parts = []
    for f in schema_pair:
        short = FUNCTION_TO_SHORT.get(f)
        if short is None:
            raise KeyError(f"Function not in mapping: {f}")
        p = clusters_root / label / f"{short}_cluster.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing cluster parquet: {p}")
        parts.append(pd.read_parquet(p))
    return pd.concat(parts, ignore_index=True)[['seq_hash', 'cluster_id']]


def _build_assembly_subtype(protein_df: pd.DataFrame) -> pd.DataFrame:
    """Per-assembly_id whole-genome HxNy subtype extracted from scientific_name.
    Returns DataFrame with cols [assembly_id, hn_subtype]. Rows without an
    HxNy match in the name are dropped (no fallback)."""
    sub = (protein_df[['assembly_id', 'scientific_name']]
           .drop_duplicates(subset=['assembly_id']).copy())
    sub['hn_subtype'] = sub['scientific_name'].astype(str).str.extract(
        f'({_HN_RE.pattern})')[0]
    return sub[['assembly_id', 'hn_subtype']].dropna(subset=['hn_subtype'])


def cramers_v_sparse(codes_a: np.ndarray, codes_b: np.ndarray,
                      n_a: int, n_b: int) -> float:
    """Cramér's V from integer label vectors of equal length (one row per
    observation). Uses χ² = Σ_observed O²/E − N, so works for very large
    contingency tables. Returns V in [0, 1]."""
    n = len(codes_a)
    if n == 0:
        return float('nan')
    # Count observed cells via groupby.
    cells = pd.DataFrame({'a': codes_a, 'b': codes_b}).groupby(['a', 'b']).size()
    row_sums = np.bincount(codes_a, minlength=n_a).astype(np.float64)
    col_sums = np.bincount(codes_b, minlength=n_b).astype(np.float64)
    # E_ij = R_i * C_j / n
    a_idx = cells.index.get_level_values('a').to_numpy()
    b_idx = cells.index.get_level_values('b').to_numpy()
    obs = cells.values.astype(np.float64)
    expected = row_sums[a_idx] * col_sums[b_idx] / n
    chi2 = float((obs ** 2 / expected).sum() - n)
    denom = n * max(min(n_a, n_b) - 1, 1)
    return float(np.sqrt(chi2 / denom))


def coupling_one_threshold(protein_df: pd.DataFrame,
                            isolate_pairs: pd.DataFrame,
                            subtype_map: pd.DataFrame,
                            cluster_lookup: pd.DataFrame,
                            threshold: float,
                            schema_pair: Tuple[str, str]) -> dict:
    lookup_a = cluster_lookup.rename(columns={'seq_hash': 'seq_hash_a',
                                              'cluster_id': 'cluster_id_a'})
    lookup_b = cluster_lookup.rename(columns={'seq_hash': 'seq_hash_b',
                                              'cluster_id': 'cluster_id_b'})
    pairs = (isolate_pairs.merge(lookup_a, on='seq_hash_a', how='left')
                          .merge(lookup_b, on='seq_hash_b', how='left'))
    pairs = pairs.dropna(subset=['cluster_id_a', 'cluster_id_b'])
    pairs = pairs.drop_duplicates(subset=['seq_hash_a', 'seq_hash_b']).reset_index(drop=True)
    pairs = pairs.merge(subtype_map, on='assembly_id', how='left')

    n_pairs = len(pairs)
    n_with_subtype = int(pairs['hn_subtype'].notna().sum())

    # Encode all categorical columns once.
    a_codes, a_levels = pd.factorize(pairs['cluster_id_a'])
    b_codes, b_levels = pd.factorize(pairs['cluster_id_b'])
    v_cluster = cramers_v_sparse(a_codes, b_codes, len(a_levels), len(b_levels))

    sub = pairs.dropna(subset=['hn_subtype']).reset_index(drop=True)
    if len(sub) > 0:
        sa_codes, sa_levels = pd.factorize(sub['cluster_id_a'])
        sb_codes, sb_levels = pd.factorize(sub['cluster_id_b'])
        s_codes, s_levels = pd.factorize(sub['hn_subtype'])
        v_a_subtype = cramers_v_sparse(sa_codes, s_codes, len(sa_levels), len(s_levels))
        v_b_subtype = cramers_v_sparse(sb_codes, s_codes, len(sb_levels), len(s_levels))
        n_subtypes = len(s_levels)
    else:
        v_a_subtype = v_b_subtype = float('nan')
        n_subtypes = 0

    return {
        'threshold': threshold,
        'n_pairs': n_pairs,
        'n_with_subtype': n_with_subtype,
        'n_clusters_a': len(a_levels),
        'n_clusters_b': len(b_levels),
        'n_subtypes': n_subtypes,
        'V_cluster_a_x_cluster_b': round(v_cluster, 4),
        'V_cluster_a_x_subtype':   round(v_a_subtype, 4),
        'V_cluster_b_x_subtype':   round(v_b_subtype, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--protein_final', type=Path)
    src.add_argument('--cds_final', type=Path)
    ap.add_argument('--clusters_root', required=True, type=Path)
    ap.add_argument('--schema_pair', required=True, nargs=2, metavar=('FUNC_A', 'FUNC_B'))
    ap.add_argument('--thresholds', type=float, nargs='+',
                    default=[1.00, 0.99, 0.98, 0.97, 0.96, 0.95])
    ap.add_argument('--out_csv', type=Path, default=None)
    args = ap.parse_args()

    if args.protein_final is not None:
        alphabet = 'aa'
        df = pd.read_parquet(args.protein_final)
    else:
        alphabet = 'nt'
        df = pd.read_parquet(args.cds_final)

    schema_pair = tuple(args.schema_pair)
    pair_label = '_'.join(FUNCTION_TO_SHORT.get(f, f).lower() for f in schema_pair)
    if args.out_csv is None:
        args.out_csv = (PROJECT_ROOT / 'results' / 'flu' / 'July_2025' / 'runs'
                        / 'cluster_disjoint_feasibility'
                        / f'cluster_pair_coupling_precheck_{pair_label}_{alphabet}.csv')
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    isolate_pairs = build_isolate_pairs(df, schema_pair, alphabet=alphabet)
    # Use the protein_final df for subtype (cds_final may not carry scientific_name).
    subtype_src = df if alphabet == 'aa' else pd.read_parquet(
        PROJECT_ROOT / 'data/processed/flu/July_2025/protein_final.parquet')
    subtype_map = _build_assembly_subtype(subtype_src)

    short_a = FUNCTION_TO_SHORT.get(schema_pair[0], 'A')
    short_b = FUNCTION_TO_SHORT.get(schema_pair[1], 'B')
    print(f'Schema pair: {short_a} (slot a)  <->  {short_b} (slot b)   alphabet={alphabet}')
    print(f'Isolate pairs (pre-cluster-dedup): {len(isolate_pairs):,}')
    print(f'Unique assemblies with HN subtype parsed: {subtype_map["hn_subtype"].notna().sum():,}')
    print()

    rows = []
    for t in args.thresholds:
        lookup = _load_cluster_lookup(args.clusters_root, t, schema_pair)
        row = coupling_one_threshold(df, isolate_pairs, subtype_map, lookup, t, schema_pair)
        rows.append(row)
        print(f"t{int(round(t*100)):03d}  "
              f"n_pairs={row['n_pairs']:,}  "
              f"clusters_a={row['n_clusters_a']:,} clusters_b={row['n_clusters_b']:,}  "
              f"subtypes={row['n_subtypes']}  |  "
              f"V({short_a}×{short_b})={row['V_cluster_a_x_cluster_b']:.4f}  "
              f"V({short_a}×subtype)={row['V_cluster_a_x_subtype']:.4f}  "
              f"V({short_b}×subtype)={row['V_cluster_b_x_subtype']:.4f}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    print()
    print('Summary table:')
    print(out_df[['threshold', 'V_cluster_a_x_cluster_b',
                  'V_cluster_a_x_subtype', 'V_cluster_b_x_subtype',
                  'n_pairs', 'n_clusters_a', 'n_clusters_b', 'n_subtypes']].to_string(index=False))
    print(f'\nWrote: {args.out_csv}')


if __name__ == '__main__':
    main()
