#!/usr/bin/env python3
"""
Apriori DNA-coverage feasibility sweep across active flu bundles.

Status (2026-05-13)
-------------------
Companion to `docs/results/2026-05-08_dna_coverage_feasibility_sweep.md`,
which documents the verdict. The feasibility result it produced
informed the DNA-level extension of v2's coverage phase (mode #2 leakage
fix at DNA level — landed 2026-05-08). Preserved as a reproducibility
hook for that result doc.

**Note on hard-coded inputs:** the `BUNDLES` list near the top of the
file references several bundle names that have since been retired
(`flu_ha_na_human_h3n2`, `flu_ha_na_human_h3n2_2024`, plus the
"(implied filter)" and "(hypothetical)" PB2/PB1 variants). These were
included for the original sweep; trim the list to currently-active
bundles (see `conf/bundles/` and especially `conf/bundles/README.md`
for status) before re-running. The metric and methodology described
below are unchanged.

Reads protein_final.csv and genome_final.csv (Stage 1 outputs), enriches
with metadata, applies each bundle's filter spec (selected_functions,
host, hn_subtype, year), and computes the cross-split feasibility metric:

    demand(H, slot)   = number of distinct dna_hash values encoding H on slot
    cooccur_blocked   = | cooccur(H) ∩ partner_universe |
    supply(H, slot)   = | partner_universe (opposite slot) | - cooccur_blocked
    ratio             = demand / supply

If max ratio < 1 across all (slot, seq_hash), DNA-level coverage is
feasible apriori (before any train/val/test split). Splits make the
constraint tighter only when partner pools shrink below the global
universe — typically not by enough to flip feasibility.

Reports per-bundle summary and worst seq_hashes; writes CSVs to
/tmp/dna_coverage_sweep/. No commits, no pipeline integration.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Project root + import metadata loader
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.utils.metadata_enrichment import load_flu_metadata  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'flu' / 'July_2025'
OUT_DIR = Path('/tmp/dna_coverage_sweep')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Function names as they appear in protein_final.csv
HA = 'Hemagglutinin precursor'
NA = 'Neuraminidase protein'
PB2 = 'RNA-dependent RNA polymerase PB2 subunit'
PB1 = 'RNA-dependent RNA polymerase catalytic core PB1 subunit'

BUNDLES = [
    {
        'name': 'flu_ha_na',
        'schema_pair': (HA, NA),
        'filter': {},
    },
    {
        'name': 'flu_ha_na_human_h3n2',
        'schema_pair': (HA, NA),
        'filter': {'host': 'Human', 'hn_subtype': 'H3N2'},
    },
    {
        'name': 'flu_ha_na_human_h3n2_2024',
        'schema_pair': (HA, NA),
        'filter': {'host': 'Human', 'hn_subtype': 'H3N2', 'year': 2024},
    },
    {
        'name': 'flu_pb2_pb1',
        'schema_pair': (PB2, PB1),
        'filter': {},
    },
    {
        'name': 'flu_pb2_pb1_human_h3n2 (implied filter)',
        'schema_pair': (PB2, PB1),
        'filter': {'host': 'Human', 'hn_subtype': 'H3N2'},
    },
    {
        'name': 'flu_pb2_pb1_human_h3n2_2024 (hypothetical)',
        'schema_pair': (PB2, PB1),
        'filter': {'host': 'Human', 'hn_subtype': 'H3N2', 'year': 2024},
    },
]


def _load_full_data() -> pd.DataFrame:
    """Load protein + genome + metadata into one row-per-protein dataframe with
    seq_hash and dna_hash already attached.
    """
    import hashlib

    print('Loading protein_final.csv ...')
    prot = pd.read_csv(
        PROCESSED_DIR / 'protein_final.csv',
        usecols=['assembly_id', 'function', 'prot_seq', 'genbank_ctg_id'],
    )
    print(f'  {len(prot):,} protein rows')

    print('Loading genome_final.csv ...')
    gen = pd.read_csv(
        PROCESSED_DIR / 'genome_final.csv',
        usecols=['assembly_id', 'genbank_ctg_id', 'dna_seq'],
    )
    print(f'  {len(gen):,} genome rows')

    print('Hashing protein and DNA sequences ...')
    prot['seq_hash'] = prot['prot_seq'].map(
        lambda s: hashlib.md5(str(s).encode()).hexdigest())
    gen['dna_hash'] = gen['dna_seq'].map(
        lambda s: hashlib.md5(str(s).encode()).hexdigest())

    print('Joining protein <- genome (assembly_id, genbank_ctg_id) ...')
    df = prot.merge(
        gen[['assembly_id', 'genbank_ctg_id', 'dna_hash']],
        on=['assembly_id', 'genbank_ctg_id'],
        how='inner',
    )
    print(f'  {len(df):,} rows after join')

    print('Loading metadata ...')
    meta = load_flu_metadata()
    meta_cols = ['assembly_id', 'host', 'hn_subtype', 'year']
    df = df.merge(meta[meta_cols], on='assembly_id', how='left')
    print(f'  metadata attached')

    return df[['assembly_id', 'function', 'seq_hash', 'dna_hash',
               'host', 'hn_subtype', 'year']]


def _apply_bundle_filter(df: pd.DataFrame, schema_pair: tuple,
                         filter_spec: dict) -> pd.DataFrame:
    """Filter to schema_pair functions and the bundle's metadata filters."""
    func_left, func_right = schema_pair
    out = df[df['function'].isin([func_left, func_right])].copy()
    for k, v in filter_spec.items():
        if k == 'year':
            out = out[out['year'] == v]
        else:
            out = out[out[k] == v]
    return out


def _per_isolate_one_row_per_function(df: pd.DataFrame, schema_pair: tuple) -> pd.DataFrame:
    """Mimic v2 strict mode: one row per (assembly_id, function). For
    isolates with multiple rows per function, keep the first deterministically.
    Then drop isolates that don't have BOTH schema_pair functions present.
    """
    func_left, func_right = schema_pair

    df = df.sort_values(['assembly_id', 'function']).drop_duplicates(
        ['assembly_id', 'function'], keep='first')

    has_left = set(df.loc[df['function'] == func_left, 'assembly_id'])
    has_right = set(df.loc[df['function'] == func_right, 'assembly_id'])
    isolates_with_both = has_left & has_right

    df = df[df['assembly_id'].isin(isolates_with_both)].copy()
    return df


def _feasibility_for_bundle(df_bundle: pd.DataFrame, schema_pair: tuple) -> dict:
    """Compute apriori feasibility for one bundle's filtered data.

    Treats every isolate as one positive pair (slot a = func_left,
    slot b = func_right). Cross-split demand vs supply is computed
    against the full filtered universe.
    """
    func_left, func_right = schema_pair
    n_isolates = df_bundle['assembly_id'].nunique()
    if n_isolates == 0:
        return {'n_isolates': 0, 'feasibility': 'NO DATA'}

    # Per slot: assembly_id -> (seq_hash, dna_hash)
    rows_a = df_bundle[df_bundle['function'] == func_left].set_index('assembly_id')[['seq_hash', 'dna_hash']]
    rows_b = df_bundle[df_bundle['function'] == func_right].set_index('assembly_id')[['seq_hash', 'dna_hash']]
    isolates = sorted(set(rows_a.index) & set(rows_b.index))
    rows_a = rows_a.loc[isolates]
    rows_b = rows_b.loc[isolates]

    # K(seq_hash, slot) = number of distinct dna_hash values for that seq_hash
    def _K_per_seq(slot_df):
        return slot_df.groupby('seq_hash')['dna_hash'].nunique()

    K_a = _K_per_seq(rows_a)
    K_b = _K_per_seq(rows_b)

    # Cooccur set: protein-level pair_keys (canonical sorted) per isolate.
    # Build cooccur_by_seq[h] = set of partner seq_hashes that h co-occurs with.
    cooccur_by_seq: dict = defaultdict(set)
    for h_a, h_b in zip(rows_a['seq_hash'].values, rows_b['seq_hash'].values):
        cooccur_by_seq[h_a].add(h_b)
        cooccur_by_seq[h_b].add(h_a)

    universe_a = set(rows_a['seq_hash'].unique())
    universe_b = set(rows_b['seq_hash'].unique())

    # demand vs supply per (slot, seq_hash)
    rows_out = []
    for slot, K_ser, partner_universe in [
        ('a', K_a, universe_b),
        ('b', K_b, universe_a),
    ]:
        for h, K in K_ser.items():
            blocked = cooccur_by_seq.get(h, set()) & partner_universe
            supply = len(partner_universe) - len(blocked)
            ratio = K / supply if supply > 0 else float('inf')
            rows_out.append({
                'slot': slot,
                'seq_hash': h,
                'K': int(K),
                'partner_universe': len(partner_universe),
                'cooccur_blocked': len(blocked),
                'supply': supply,
                'ratio': ratio,
            })

    feas_df = pd.DataFrame(rows_out)
    n_infeasible = int((feas_df['ratio'] > 1).sum())

    return {
        'n_isolates': n_isolates,
        'n_uniq_seq_a': len(universe_a),
        'n_uniq_seq_b': len(universe_b),
        'K_max_a': int(K_a.max()) if len(K_a) else 0,
        'K_max_b': int(K_b.max()) if len(K_b) else 0,
        'K_mean_a': float(K_a.mean()) if len(K_a) else 0.0,
        'K_mean_b': float(K_b.mean()) if len(K_b) else 0.0,
        'supply_min': int(feas_df['supply'].min()) if len(feas_df) else 0,
        'ratio_max': float(feas_df['ratio'].max()) if len(feas_df) else 0.0,
        'n_infeasible': n_infeasible,
        'feasibility': 'FEASIBLE' if n_infeasible == 0 else f'INFEASIBLE ({n_infeasible} seq_hashes)',
        'feas_df': feas_df,
    }


def main() -> None:
    print('=' * 72)
    print('DNA coverage feasibility sweep — apriori, across active bundles')
    print('=' * 72)

    full = _load_full_data()

    summary_rows = []
    for spec in BUNDLES:
        print(f'\n--- Bundle: {spec["name"]} ---')
        print(f'    schema_pair = {spec["schema_pair"]}')
        print(f'    filter = {spec["filter"]}')

        df_bundle = _apply_bundle_filter(full, spec['schema_pair'], spec['filter'])
        df_bundle = _per_isolate_one_row_per_function(df_bundle, spec['schema_pair'])

        result = _feasibility_for_bundle(df_bundle, spec['schema_pair'])

        # Save per-bundle feas_df
        feas_df = result.pop('feas_df', pd.DataFrame())
        if len(feas_df):
            safe_name = spec['name'].replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
            feas_df.sort_values('ratio', ascending=False).to_csv(
                OUT_DIR / f'feas_{safe_name}.csv', index=False)

        result['bundle'] = spec['name']
        result['filter'] = str(spec['filter'])
        summary_rows.append(result)

        print(f'    n_isolates={result["n_isolates"]:,}')
        print(f'    K_max (a, b) = ({result["K_max_a"]}, {result["K_max_b"]});  '
              f'K_mean = ({result["K_mean_a"]:.2f}, {result["K_mean_b"]:.2f})')
        print(f'    supply_min = {result["supply_min"]:,}')
        print(f'    ratio_max = {result["ratio_max"]:.4f}')
        print(f'    >>> {result["feasibility"]} <<<')

    # Cross-bundle summary
    print('\n' + '=' * 72)
    print('Cross-bundle summary')
    print('=' * 72)
    summary_df = pd.DataFrame(summary_rows)[[
        'bundle', 'filter', 'n_isolates',
        'K_max_a', 'K_max_b', 'K_mean_a', 'K_mean_b',
        'supply_min', 'ratio_max', 'feasibility',
    ]]
    summary_df.to_csv(OUT_DIR / 'sweep_summary.csv', index=False)
    print(summary_df.to_string(index=False))
    print(f'\nWrote {OUT_DIR / "sweep_summary.csv"} and per-bundle feas_*.csv')


if __name__ == '__main__':
    main()
