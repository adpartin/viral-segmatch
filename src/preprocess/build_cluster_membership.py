"""Per-record cluster-membership table (records-weighted master substrate).

For every major-protein record, record its mmseqs `cluster_id` at each identity
threshold (t100..t090) alongside isolate metadata. One row per (isolate, protein)
record — records-weighted, no dedup. This is the shared substrate for both axes
the biology splits on:
  - leakage  — cluster-disjoint partitions (a cluster's sequences go wholly to
               one split); read a record's `tXXX` cluster_id to partition.
  - redundancy — cluster-pair dedup/balance (each (H-cluster, N-cluster) cell
               contributes one/capped example); join two slots on `assembly_id`
               and group by the `tXXX` columns.
It also centralizes the per-(protein, threshold) `load_cluster_map` calls
currently scattered across `src/analysis/{cluster,bigraph}_*.py`.

**Per-alphabet key — do not cross the streams.** Each alphabet keys on its OWN
hash, matching the column its cluster parquets were built on:
  - aa     : `prot_hash` = compute_prot_hash(prot_seq) = md5(prot_seq). Stage 1
             writes prot_hash to `protein_final`; recomputed here for parity.
  - nt_cds : `cds_dna_hash` = md5(cds_dna), already present in `cds_dna_final`.
The aa protein hash is NEVER used to join nt_cds clusters (and vice versa): the
nt_cds cluster parquets key on `cds_dna_hash`, the aa ones on `prot_hash`.

Metadata (`hn_subtype, host, year, geo_location_clean`) is joined per isolate from
`load_flu_metadata`; `geo_location_clean` is the project's canonical downstream
location column (the one the `geo_location` holdout axis consumes).

CLI:
    python -m src.preprocess.build_cluster_membership --alphabet aa
    python -m src.preprocess.build_cluster_membership --alphabet nt_cds
    # optional overrides: --source --clusters_root --virus_yaml --out

Output columns (alphabet-specific key + length):
    assembly_id, function, <prot_hash|cds_dna_hash>, <length|cds_length>,
    hn_subtype, host, year, geo_location_clean,    # per isolate
    t100, t099, ..., t090                          # cluster_id per threshold
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.clustering_utils import compute_prot_hash  # noqa: E402
from src.utils.config_hydra import load_function_metadata  # noqa: E402
from src.utils.metadata_enrichment import load_flu_metadata  # noqa: E402

_META_COLS = ['hn_subtype', 'host', 'year', 'geo_location_clean']
_FILL_UNKNOWN = ['hn_subtype', 'host', 'geo_location_clean']  # categorical; year left numeric

# Per-alphabet wiring. `key_col` is the join key in BOTH the record table and the
# cluster parquet; `compute_key_from` is the sequence column to hash when the
# source (aa: recomputed from prot_seq, equals the stored prot_hash), or None
# when the source already carries it (nt_cds: cds_dna_final has cds_dna_hash).
_ALPHABET_CFG = {
    'aa': {
        'source': 'data/processed/flu/July_2025/protein_final.parquet',
        'clusters': 'data/processed/flu/July_2025/clusters_aa',
        'out': 'data/processed/flu/July_2025/cluster_membership/cluster_memb_aa.parquet',
        'key_col': 'prot_hash',
        'length_col': 'length',
        'compute_key_from': 'prot_seq',
    },
    'nt_cds': {
        'source': 'data/processed/flu/July_2025/cds_dna_final.parquet',
        'clusters': 'data/processed/flu/July_2025/clusters_nt_cds',
        'out': 'data/processed/flu/July_2025/cluster_membership/cluster_memb_nt_cds.parquet',
        'key_col': 'cds_dna_hash',
        'length_col': 'cds_length',
        'compute_key_from': None,
    },
}


def _list_thresholds(clusters_root: Path) -> list[str]:
    """Sorted (t100 first) tXXX subdir names under a cluster root."""
    names = [d.name for d in clusters_root.iterdir()
             if d.is_dir() and d.name.startswith('t') and d.name[1:].isdigit()]
    return sorted(names, reverse=True)


def build_cluster_columns(source: Path, clusters_root: Path, virus_yaml: Path, *,
                          key_col: str, length_col: str,
                          compute_key_from: Optional[str]) -> tuple[pd.DataFrame, list[str]]:
    """Per-major-record table with one cluster_id column per threshold.

    Joins each record's `key_col` (the alphabet's own hash) to the cluster
    parquet keyed on the same column, so aa and nt_cds never share a hash space.
    """
    fmeta = load_function_metadata(virus_yaml)
    majors_short = list(fmeta.selected_short_names)
    full_of = fmeta.short_to_function
    majors_full = [full_of[s] for s in majors_short]

    thresholds = _list_thresholds(clusters_root)
    if not thresholds:
        raise SystemExit(f"ERROR: no tXXX dirs under {clusters_root}")

    read_cols = ['assembly_id', 'function', length_col]
    read_cols += [compute_key_from] if compute_key_from else [key_col]
    if source.suffix == '.parquet':
        df = pd.read_parquet(source, columns=read_cols)
    else:
        # keep_default_na guards the project-wide 'NA'-string trap (function uses
        # full names today, but it's the safe default for any CSV read here).
        df = pd.read_csv(source, usecols=read_cols, keep_default_na=False, na_values=[''])
    df = df[df['function'].isin(majors_full)].copy()
    if compute_key_from:
        df[key_col] = df[compute_key_from].map(compute_prot_hash)
        df = df.drop(columns=[compute_key_from])
    print(f"Loaded {len(df):,} major-protein records "
          f"({df[key_col].nunique():,} unique {key_col} across {df['function'].nunique()} proteins).")

    frames: list[pd.DataFrame] = []
    for short in majors_short:
        full = full_of[short]
        recs = df[df['function'] == full].copy()
        if recs.empty:
            print(f"  WARNING: no records for {short} ({full}); skipping.")
            continue
        for t in thresholds:
            pq = clusters_root / t / f'{short}_cluster.parquet'
            if not pq.exists():
                print(f"  WARNING: missing {pq}; {t} set NaN for {short}.")
                recs[t] = pd.NA
                continue
            cl = pd.read_parquet(pq, columns=[key_col, 'cluster_id'])
            recs[t] = recs[key_col].map(dict(zip(cl[key_col], cl['cluster_id'])))
        n_unmapped = int(recs[thresholds].isna().any(axis=1).sum())
        print(f"  {short:<4} {len(recs):>8,} records, {recs[key_col].nunique():>7,} unique"
              + (f"  ({n_unmapped:,} unmapped at >=1 t)" if n_unmapped else "  (all mapped)"))
        frames.append(recs)

    table = pd.concat(frames, ignore_index=True)
    return table, thresholds


def attach_metadata(table: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Left-join per-isolate metadata on assembly_id; return (table, match_rate)."""
    meta = load_flu_metadata()[['assembly_id'] + _META_COLS].copy()
    meta['assembly_id'] = meta['assembly_id'].astype(str)
    meta = meta.drop_duplicates('assembly_id')  # one row per isolate -> clean 1:1 join

    table['assembly_id'] = table['assembly_id'].astype(str)
    match_rate = float(table['assembly_id'].isin(set(meta['assembly_id'])).mean())
    table = table.merge(meta, on='assembly_id', how='left')
    for c in _FILL_UNKNOWN:
        table[c] = table[c].fillna('unknown')
    return table, match_rate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--alphabet', default='aa', choices=list(_ALPHABET_CFG))
    p.add_argument('--source', default=None, help='Override the record-table source.')
    p.add_argument('--clusters_root', default=None, help='Override the cluster root.')
    p.add_argument('--virus_yaml', default=str(PROJ / 'conf/virus/flu.yaml'))
    p.add_argument('--out', default=None, help='Override the output parquet path.')
    args = p.parse_args()

    cfg = _ALPHABET_CFG[args.alphabet]
    source = Path(args.source) if args.source else PROJ / cfg['source']
    clusters_root = Path(args.clusters_root) if args.clusters_root else PROJ / cfg['clusters']
    out = Path(args.out) if args.out else PROJ / cfg['out']
    key_col, length_col = cfg['key_col'], cfg['length_col']

    print(f"=== building cluster_membership [{args.alphabet}] "
          f"(key={key_col}) ===")
    table, thresholds = build_cluster_columns(
        source, clusters_root, Path(args.virus_yaml),
        key_col=key_col, length_col=length_col, compute_key_from=cfg['compute_key_from'])
    table, match_rate = attach_metadata(table)

    ordered = ['assembly_id', 'function', key_col, length_col] + _META_COLS + thresholds
    table = table[ordered]

    n_t_nan = int(table[thresholds].isna().any(axis=1).sum())
    print(f"\nmetadata match-rate: {match_rate:.4%} of records found in flu metadata")
    if n_t_nan:
        print(f"WARNING: {n_t_nan:,} records have an unmapped cluster at >=1 threshold.")
    else:
        print("All records mapped at every threshold.")

    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    print(f"\nwrote {out}")
    print(f"  {len(table):,} rows x {len(table.columns)} cols: {list(table.columns)}")
    print("\nDone.")


if __name__ == '__main__':
    main()
