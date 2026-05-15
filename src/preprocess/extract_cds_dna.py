"""Stage 1.5: emit `cds_final.parquet` from Stage 1 outputs.

For every selected protein row in `protein_final.csv`, reconstruct the
coding DNA from `genome_final.csv` using the `location` field (per
`docs/methods/gto_format_reference.md` § 9), hash it, and write a
slim parquet with the columns Stage 3 / clustering will consume.

Validation is performed in-loop:
- HARD-FAIL on any selected-function row where translate-back disagrees
  with the stored `prot_seq` (modulo a trailing '*'). Selected functions
  are the 8 majors used by the modeling pipeline (PB2, PB1, PA, HA, NP,
  NA, M1, NS1 by default — read from `conf/virus/<virus>.yaml`).
- Rows whose CDS slice runs off the end of the contig, or whose
  protein cannot be located in the genome table, are dropped with a
  WARNING.

This script is the prerequisite for Experiment B-nt (nt-level
cluster_disjoint splits); see
`docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` § B-nt.

Usage:
    python src/preprocess/extract_cds_dna.py \
        --config_bundle flu_ha_na \
        [--functions "PB2 subunit" "PB1 subunit"]  # override selection

Outputs:
    data/processed/<virus>/<data_version>/cds_final.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from src.utils.cds_utils import (
    compute_cds_dna_hash,
    extract_cds_dna,
    translate_dna,
)
from src.utils.config_hydra import get_virus_config_hydra


_OUTPUT_COLUMNS = [
    'assembly_id',
    'genbank_ctg_id',
    'function',
    'seq_hash',           # md5(prot_seq) — joins back to protein_final
    'prot_seq',           # for downstream identity-disjointness checks
    'cds_dna',
    'cds_dna_hash',       # md5(cds_dna) — Experiment B-nt cluster key
    'length',             # AA length (from Stage 1)
    'cds_length',         # nt length (== 3 * AA length for major CDSs)
]


def _validate_row(prot_seq: str, cds_dna: str) -> tuple[bool, str]:
    """Return (ok, reason). 'ok' means translate-back matches prot_seq
    modulo a single trailing '*'.

    `prot_seq` may legally contain 'X' for ambiguous residues. Treat
    'X' as a wildcard when comparing against the translator's output
    (which may produce a concrete residue when the underlying IUPAC
    code resolves synonymously).
    """
    try:
        aa = translate_dna(cds_dna)
    except ValueError as e:
        return False, f'translate_failed: {e}'
    a = aa.rstrip('*')
    b = prot_seq.rstrip('*')
    if len(a) != len(b):
        return False, f'aa_len {len(a)} != prot_seq_len {len(b)}'
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            continue
        if x == 'X' or y == 'X':
            continue
        return False, (
            f'mismatch at aa pos {i}: {x!r} vs {y!r} '
            f'(codon={cds_dna[i*3:(i+1)*3]!r})'
        )
    return True, ''


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--config_bundle', required=True,
                   help='Hydra bundle name (only the virus block is used).')
    p.add_argument('--functions', nargs='+', default=None,
                   help='Override selected_functions (full function strings).')
    p.add_argument('--out_filename', default='cds_final.parquet',
                   help='Output filename (placed under '
                        'data/processed/<virus>/<version>/).')
    p.add_argument('--allow_validation_warnings', action='store_true',
                   help='Demote translate-back mismatch on selected '
                        'functions from HARD-FAIL to WARNING.')
    args = p.parse_args()

    config = get_virus_config_hydra(args.config_bundle,
                                    config_path=str(PROJ / 'conf'))
    virus_name = config.virus.virus_name
    data_version = config.virus.data_version
    selected_functions = list(args.functions or config.virus.selected_functions)
    print(f'virus: {virus_name}')
    print(f'data_version: {data_version}')
    print(f'selected_functions ({len(selected_functions)}):')
    for f in selected_functions:
        print(f'  - {f}')

    base = PROJ / 'data' / 'processed' / virus_name / data_version
    prot_csv = base / 'protein_final.csv'
    gen_csv = base / 'genome_final.csv'
    out_path = base / args.out_filename
    if not prot_csv.exists() or not gen_csv.exists():
        print(f'ERROR: missing Stage 1 outputs under {base}', file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    print(f'\nReading {prot_csv} ...')
    prot = pd.read_csv(
        prot_csv,
        dtype={'assembly_id': str, 'genbank_ctg_id': str},
        usecols=['assembly_id', 'genbank_ctg_id', 'function',
                 'prot_seq', 'location', 'length'],
    )
    # protein_final.csv does not store seq_hash; compute it inline using
    # the same formula Stage 3 uses (`md5(prot_seq)`).
    prot['seq_hash'] = prot['prot_seq'].map(
        lambda s: hashlib.md5(str(s).encode()).hexdigest()
    )
    print(f'  protein_final rows: {len(prot):,}')
    print(f'\nFiltering to selected functions ...')
    prot = prot[prot['function'].isin(selected_functions)].reset_index(drop=True)
    print(f'  kept rows: {len(prot):,}')
    print(f'  per-function counts:')
    for fn, n in prot['function'].value_counts().items():
        print(f'    {n:>10,}  {fn}')

    print(f'\nReading {gen_csv} ...')
    gen = pd.read_csv(
        gen_csv,
        dtype={'assembly_id': str, 'genbank_ctg_id': str},
        usecols=['assembly_id', 'genbank_ctg_id', 'dna_seq'],
    )
    print(f'  genome_final rows: {len(gen):,}')
    ctg_lookup: dict[tuple[str, str], str] = {
        (a, c): d for a, c, d in
        zip(gen['assembly_id'], gen['genbank_ctg_id'], gen['dna_seq'])
    }
    del gen
    print(f'  contig lookup size: {len(ctg_lookup):,}')

    print(f'\nExtracting CDS ...')
    cds_list: list[str] = []
    cds_hashes: list[str] = []
    keep_idx: list[int] = []
    n_missing_contig = 0
    n_extract_fail = 0
    n_validate_fail = 0
    sample_log: list[str] = []

    for i, row in enumerate(prot.itertuples(index=False)):
        key = (row.assembly_id, row.genbank_ctg_id)
        contig = ctg_lookup.get(key)
        if contig is None:
            n_missing_contig += 1
            if len(sample_log) < 5:
                sample_log.append(
                    f'missing_contig: assembly_id={row.assembly_id} '
                    f'genbank_ctg_id={row.genbank_ctg_id}'
                )
            continue
        try:
            cds_dna = extract_cds_dna(contig, row.location)
        except ValueError as e:
            n_extract_fail += 1
            if len(sample_log) < 5:
                sample_log.append(
                    f'extract_fail: assembly_id={row.assembly_id} '
                    f'function={row.function[:35]}: {e}'
                )
            continue
        ok, reason = _validate_row(row.prot_seq, cds_dna)
        if not ok:
            n_validate_fail += 1
            if len(sample_log) < 5:
                sample_log.append(
                    f'validate_fail: assembly_id={row.assembly_id} '
                    f'function={row.function[:35]}: {reason}'
                )
            if not args.allow_validation_warnings:
                raise RuntimeError(
                    f'translate-back mismatch on selected function '
                    f'{row.function}: {reason}. To override, pass '
                    f'--allow_validation_warnings.'
                )
            continue
        keep_idx.append(i)
        cds_list.append(cds_dna)
        cds_hashes.append(compute_cds_dna_hash(cds_dna))
        if (i + 1) % 100_000 == 0:
            print(f'  ... processed {i + 1:,} / {len(prot):,} '
                  f'({time.time() - t0:.0f}s elapsed)')

    print(f'\nExtraction summary:')
    print(f'  rows attempted:        {len(prot):,}')
    print(f'  rows kept:             {len(keep_idx):,}')
    print(f'  WARNING missing_contig: {n_missing_contig:,}')
    print(f'  WARNING extract_fail:   {n_extract_fail:,}')
    print(f'  WARNING validate_fail:  {n_validate_fail:,}')
    if sample_log:
        print(f'  sample warnings ({len(sample_log)}):')
        for line in sample_log:
            print(f'    {line}')

    print(f'\nBuilding output frame ...')
    out = prot.iloc[keep_idx].copy().reset_index(drop=True)
    out['cds_dna'] = cds_list
    out['cds_dna_hash'] = cds_hashes
    out['cds_length'] = out['cds_dna'].apply(len)
    out = out[_OUTPUT_COLUMNS]
    print(f'  output rows:  {len(out):,}')
    print(f'  output cols:  {list(out.columns)}')
    print(f'  unique cds_dna_hash: {out["cds_dna_hash"].nunique():,}')
    print(f'  per-function unique cds_dna_hash:')
    for fn, sub in out.groupby('function')['cds_dna_hash']:
        print(f'    {sub.nunique():>10,} / {len(sub):>10,}  {fn}')

    print(f'\nWriting {out_path} ...')
    out.to_parquet(out_path, index=False)
    sz_mb = out_path.stat().st_size / 1e6
    print(f'  size: {sz_mb:.1f} MB')

    print(f'\nDone. ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
