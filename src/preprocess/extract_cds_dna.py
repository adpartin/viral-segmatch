"""Stage 1.5: emit `cds_dna_final.parquet` from Stage 1 outputs.

1. The protein record says where in the contig this CDS sits (its coordinates).
2. The DNA letters are cut out of the contig at those coordinates.
3. The code translates that DNA and checks it matches the stored protein.

Important flags carried from Stage 1:

- starts_with_m — the stored protein begins with M. In this data version, this
  agrees with ATG as the 1st codon in every extracted CDS. It is used as the
  5'-completeness marker.

- has_internal_stop — a stop marker turns up somewhere in the middle instead of
  only at the end. The record is still full length, so the positions do not
  shift; what it means is that the read is bad, or that copy of the CDS is
  broken. The flag is reliable because the translation is compared position by
  position against the stored protein, so a stop in the middle of the DNA has
  to show up as a * in the middle of the protein. There are none in this data
  version: 0 of 868,240 rows.

- has_terminal_stop — the stored protein ends with a stop marker. It agrees with
  the translated final DNA codon in every extracted CDS. Six final codons are
  the unambiguous IUPAC stop TAR rather than a literal TAA, TAG, or TGA.

For every selected protein row in `protein_final.csv`, reconstruct the
coding DNA from `ctg_dna_final.csv` using the `location` field (per
`docs/methods/gto_format_reference.md` § 9), hash it, and write a
slim parquet with the columns Stage 3 / clustering will consume.

Validation is performed in-loop:
- HARD-FAIL on any selected-function row where translate-back disagrees
  with the stored `prot_seq` (modulo a trailing '*'). Selected functions
  are the 8 major proteins used by the modeling pipeline (PB2, PB1, PA,
  HA, NP, NA, M1, NS1 by default — read from `conf/virus/flu.yaml`).
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
    data/processed/<virus>/<data_version>/cds_dna_final.parquet

    One row per (isolate, function): each row contains the coding sequence
    (`cds_dna_seq`) of one selected protein along with its hash (`cds_dna_hash`,
    used as the nt-level cluster key). UTRs, introns, and intergenic DNA
    are NOT included — only the CDS. For spliced proteins (M2, NEP, M42,
    etc.) exons are joined in the order they appear in `location`. The
    full genome contig (not consumed downstream by clustering) lives
    separately in `ctg_dna_final.csv` under the `ctg_dna_seq` column.

TODO (over-specified `--config_bundle`):
    This script reads only three fields from the bundle config —
    `virus_name`, `data_version`, `selected_functions` — all of which come
    from `config.virus.*` (i.e., from `conf/virus/<name>.yaml` directly,
    not from any bundle-level keys). A future simplification could replace
    `--config_bundle` with `--virus <name>` and load `conf/virus/<name>.yaml`
    directly via Hydra's group-config loader. That would be more
    semantically accurate (this is a virus-level preprocessing step, not
    bundle-level), but it deviates from the convention used by Stages 1,
    2, 3, 4 (all take `--config_bundle`). Leaving as-is for now; not
    blocking.

Validation status (2026-05-15):
    Unspliced selected_functions (8 Flu A majors): 868,240 / 868,240
        rows extracted and translate-back-validated — 100% pass under
        --strict mode (the default).
    Spliced auxiliary functions (M2, NEP, M42, NS3, PA-X, PB2 splice
        variant): empirically validated end-to-end. Per-function
        retention with --allow_validation_warnings:
          M2  98.76%, NEP 99.69%, M42 68.48% (~30% bear BV-BRC `-1`
          sentinel on the second exon — incomplete annotations rejected
          at parse time), NS3 99.90%, PA-X 99.61%, PB2-splice 100.00%.
        Overall 456,265 / 491,208 = 92.9% across all 6 spliced
        functions; sub-1% translate-back failures on 4 of 6 trace to
        anomalous GTO prot_seq annotations (terminal 'X' instead of
        stop). Pass `--allow_validation_warnings` to include spliced
        functions; the default mode hard-fails on the first
        translate-back mismatch to protect against silently wrong CDS.
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
    'brc_fea_id',         # protein occurrence id (CDS<->protein 1-1) — nt_cds k-mer key
    'function',
    'canonical_segment',  # segment label — carried for the nt_cds k-mer index / grouping
    'prot_hash',          # md5(prot_seq) — read from protein_final (produced at Stage 1)
    'prot_seq',           # for downstream identity-disjointness checks
    'cds_dna_seq',
    'cds_dna_hash',       # md5(cds_dna_seq) — nt_cds cluster key
    'length',             # AA length (from Stage 1)
    'cds_length',         # nt length (== 3 * AA length for major CDSs)
    # Completeness, carried from Stage 1 rather than recomputed. The protein-level answer is the
    # more reliable one: translation resolves IUPAC ambiguity codes that a literal DNA match
    # misses (6 rows corpus-wide end in TAR, where R is A-or-G and either way a stop).
    'starts_with_m',
    'has_terminal_stop',
    'has_internal_stop',
    'is_complete_cds',    # starts_with_m & has_terminal_stop & ~has_internal_stop
]


def _coerce_boolean_flag(series: pd.Series, flag_name: str) -> pd.Series:
    """Return a strict boolean series; reject missing or unrecognized values."""
    if pd.api.types.is_bool_dtype(series.dtype):
        if series.isna().any():
            raise ValueError(f'{flag_name} contains missing values')
        return series.astype(bool)

    def parse(value):
        if pd.isna(value):
            raise ValueError(f'{flag_name} contains a missing value')
        if isinstance(value, bool):
            return value
        if value == 'True':
            return True
        if value == 'False':
            return False
        raise ValueError(
            f'{flag_name} contains invalid boolean value {value!r}; '
            "expected True or False"
        )

    return series.map(parse).astype(bool)


def _build_output_frame(
    prot: pd.DataFrame,
    keep_idx: list[int],
    cds_list: list[str],
    cds_hashes: list[str],
) -> pd.DataFrame:
    """Build the Stage 1.5 output and preserve the Stage 1 completeness flags."""
    if not (len(keep_idx) == len(cds_list) == len(cds_hashes)):
        raise ValueError('keep_idx, cds_list, and cds_hashes must have equal lengths')

    out = prot.iloc[keep_idx].copy().reset_index(drop=True)
    out['cds_dna_seq'] = cds_list
    out['cds_dna_hash'] = cds_hashes
    out['cds_length'] = out['cds_dna_seq'].map(len)
    # is_complete_cds checks 3 things: does it start with M, does it end with a
    # stop, is there no stop in the middle. It does NOT check length.
    #
    # Length can't be judged from one record by itself — length needs to be compared
    # against other records to know what "normal" looks like. That comparison is a job
    # for the experiment that uses the data, not for this shared preprocessing step.
    out['is_complete_cds'] = (out['starts_with_m']
                              & out['has_terminal_stop']
                              & ~out['has_internal_stop'])
    return out[_OUTPUT_COLUMNS]


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
    p.add_argument('--config_bundle',
                   required=True,
                   help='Hydra bundle name (only the virus block is used).')
    p.add_argument('--functions',
                   nargs='+', default=None,
                   help='Override selected_functions (full function strings).')
    p.add_argument('--out_filename',
                   default='cds_dna_final.parquet',
                   help='Output filename (placed under '
                        'data/processed/<virus>/<version>/).')
    p.add_argument('--allow_validation_warnings',
                   action='store_true',
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
    gen_csv = base / 'ctg_dna_final.csv'
    out_path = base / args.out_filename
    if not prot_csv.exists() or not gen_csv.exists():
        print(f'ERROR: missing Stage 1 outputs under {base}', file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    print(f'\nReading {prot_csv} ...')
    prot = pd.read_csv(
        prot_csv,
        dtype={'assembly_id': str, 'genbank_ctg_id': str, 'brc_fea_id': str},
        usecols=['assembly_id', 'genbank_ctg_id', 'brc_fea_id', 'function',
                 'canonical_segment', 'prot_seq', 'prot_hash', 'location', 'length',
                 'starts_with_m', 'has_terminal_stop', 'has_internal_stop'],
    )
    # CSV may return booleans or their text representation. Reject anything else.
    for _flag in ('starts_with_m', 'has_terminal_stop', 'has_internal_stop'):
        prot[_flag] = _coerce_boolean_flag(prot[_flag], _flag)
    # prot_hash is produced + persisted at Stage 1 (protein_final); read it, don't recompute.
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
        usecols=['assembly_id', 'genbank_ctg_id', 'ctg_dna_seq'],
    )
    print(f'  ctg_dna_final rows: {len(gen):,}')
    ctg_lookup: dict[tuple[str, str], str] = {
        (a, c): d for a, c, d in
        zip(gen['assembly_id'], gen['genbank_ctg_id'], gen['ctg_dna_seq'])
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

    # Per-row CDS extraction. Each protein row produces one cds_dna.
    # Steps (mirrored in the loop body below):
    #   1. Look up the protein's matching contig DNA in `ctg_lookup`
    #      (built from genome_final) via the (assembly_id, genbank_ctg_id)
    #      key. Rows whose contig is missing are dropped with a WARNING.
    #   2. Parse the protein's `location` field — a Python list of exon
    #      tuples (contig_id, start_1based, strand, length), one per exon.
    #      `parse_location` (src/utils/cds_utils.py) handles the in-memory
    #      list form and the CSV repr-string form, and rejects BV-BRC's
    #      `length=-1` sentinel (incomplete spliced annotations on
    #      ~30% of M42 rows).
    #   3. Inside `extract_cds_dna`: slice the contig at each exon's
    #      [start, start+length) coordinates, concatenate the slices in
    #      the order they appear in `location` (joining splice junctions
    #      for spliced proteins), and reverse-complement the whole
    #      concatenation if the strand is `-`. Mixed strands within one
    #      location raise ValueError.
    #   4. Validate by translate-back (`_validate_row`): the translation
    #      of `cds_dna` must match `prot_seq` (X residues wildcarded,
    #      single trailing `*` tolerated). In strict mode (default),
    #      a mismatch on a selected function raises immediately. With
    #      --allow_validation_warnings, the row is dropped with a warning.
    #   5. Successful rows contribute one entry each to cds_list,
    #      cds_hashes (md5(cds_dna)), and keep_idx (used to slice
    #      `prot` after the loop). `cds_length` is derived later from
    #      `len(cds_dna)`.
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
    out = _build_output_frame(prot, keep_idx, cds_list, cds_hashes)
    print(f'  is_complete_cds: {int(out["is_complete_cds"].sum()):,} of {len(out):,} '
          f'({100 * out["is_complete_cds"].mean():.2f}%)')
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
