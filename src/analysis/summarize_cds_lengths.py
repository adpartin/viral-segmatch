"""Per-protein CDS completeness and length survey, one row per major protein.

Written to decide which of the 28 schema pairs are worth pursuing for per-site features. Those
features need every sequence at one length, so a pair is only viable when both of its proteins
have a dominant length. This reports what each protein's length distribution actually looks like
so that question can be answered before a run is attempted.

Every count here is of CDS DNA, never protein. The `protein` column names which gene the CDS
belongs to; the sequences counted are nucleotide, keyed on `cds_dna_hash`.

Sequence statistics count each distinct CDS once. Counting rows instead would let a heavily
sampled strain decide the answer on its own, since one sequence appears once per isolate carrying
it. Isolate statistics count isolates, and the two differ a lot: on human H3N2 2024, 5,346 isolates
carry only 815 distinct M1 sequences.

`min`, `max`, `median`, `mode` and `complete CDS at mode` describe COMPLETE sequences only. A sequence is complete when `is_complete_cds` holds, which is
`starts_with_m & has_terminal_stop & ~has_internal_stop`.

Screen on `frac isolates at mode`, not on `frac at mode`. The difference between them is the
DENOMINATOR, not sequences versus isolates. `frac at mode` divides by the complete sequences, so
it cannot see a gene whose records are mostly incomplete. `frac isolates at mode` divides by every
isolate carrying the gene, so it can.

PB1 on human H3N2 2024 is the case that matters. It reads 0.994 at mode over complete sequences
and 0.551 over isolates, because only 2,958 of its 5,346 isolates have a complete CDS at all.
Deduplication is not the cause: among isolates that DO have a complete PB1, 0.996 are at the mode.
`frac isolates complete` reports that first failure on its own, so the two isolate columns say
which of the two problems a gene has.

SCOPE. This reads the whole corpus, which spans every subtype and year. HA is 1,701 nt in H3N2 but
1,704 in H5N1 and 1,683 in H9/H7, so a corpus-wide mode is a mixture rather than a fact about any
one population. Read the output as a screen for which pairs look viable, NOT as a length to pin.
`summarize_cds_lengths` takes an already-filtered frame, so narrowing to one subtype or year later
is a matter of filtering before the call and passing a `population` label; the summary itself needs
no change.

Outputs (to `--out_dir`):
    cds_length_survey.csv   one row per protein per population, with the columns in `COLUMNS`

One run surveys one population, and the file accumulates them: a run adds its rows to whatever is
already there and replaces the rows of any population it recomputes, keyed on the `population`
column. Populations therefore share one file instead of each taking its own, which keeps a
comparison across populations a groupby rather than a join. Reading that file back needs
`keep_default_na=False, na_values=['']`, since the `protein` column holds the literal string `NA`.

CLI:
    python -m src.analysis.summarize_cds_lengths
    python -m src.analysis.summarize_cds_lengths --hn_subtype H3N2 --host Human --year 2024
    python -m src.analysis.summarize_cds_lengths --hn_subtype H3N2 --year_range 2021 2025
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.cds_utils import modal_length  # noqa: E402
from src.utils.config_hydra import get_function_short_name_map, get_virus_config_hydra  # noqa: E402
from src.utils.metadata_enrichment import attach_isolate_metadata, filter_by_metadata  # noqa: E402

COLUMNS = ['population', 'Segment ID', 'protein', 'isolates', 'unique CDS', 'complete CDS',
           'min', 'max', 'median', 'mode', 'complete CDS at mode', 'frac at mode',
           'frac isolates complete', 'frac isolates at mode', 'mode tie']


def population_label(hn_subtype, host, year, year_range) -> str:
    """Describe a metadata filter in one string, for the `population` column.

    Args:
      hn_subtype: subtype filter, or None.
      host: host filter, or None.
      year: year filter, or None.
      year_range: inclusive [min, max] year filter, or None.

    Returns:
      The filters joined by spaces, or 'all' when none was given.
    """
    parts = []
    for value in (hn_subtype, host, year):
        if value is not None:
            parts.extend(str(v) for v in value)
    if year_range is not None:
        parts.append(f'{year_range[0]}-{year_range[1]}')
    return ' '.join(parts) if parts else 'all'


def segment_number(canonical_segment: str) -> int:
    """Turn a canonical segment label into its segment number.

    Args:
      canonical_segment: label of the form `S1` ... `S8`.

    Returns:
      The segment number, 1 through 8.

    Raises:
      ValueError: the label is not `S` followed by digits.
    """
    text = str(canonical_segment)
    if not text.startswith('S') or not text[1:].isdigit():
        raise ValueError(f"canonical_segment must look like 'S1'; got {canonical_segment!r}.")
    return int(text[1:])


def summarize_cds_lengths(cds: pd.DataFrame, function_to_short: dict,
                          population: str = 'all') -> pd.DataFrame:
    """Summarize completeness and CDS length for every protein in one population.

    Takes an already-filtered frame so that a later per-subtype or per-year breakdown is a matter
    of filtering before the call. The `population` label is carried into the output so tables from
    several populations can be stacked and still be told apart.

    Sequence statistics count each distinct `cds_dna_hash` once. Isolate statistics are taken
    from the rows as given, because deduplicating collapses the isolates that share a sequence.

    Args:
      cds: rows from `cds_dna_final`, needing `assembly_id`, `function`, `canonical_segment`,
          `cds_dna_hash`, `cds_length` and `is_complete_cds`.
      function_to_short: full function name -> short protein name; a name absent from the map is
          kept in full.
      population: label describing what `cds` was filtered to.

    Returns:
      One row per protein, ordered by segment number, with the columns in `COLUMNS`.

    Raises:
      ValueError: a required column is missing, or a protein has no complete sequence.
    """
    required = {'assembly_id', 'function', 'canonical_segment', 'cds_dna_hash', 'cds_length',
                'is_complete_cds'}
    missing = required - set(cds.columns)
    if missing:
        raise ValueError(f"summarize_cds_lengths: missing columns {sorted(missing)}.")

    # One row per distinct sequence, before any sequence statistic is taken. Isolate counts are
    # taken from `cds` itself, because deduplicating collapses the isolates that share a sequence.
    unique = cds.drop_duplicates('cds_dna_hash')
    rows = []
    for function, group in unique.groupby('function'):
        records = cds[cds['function'] == function]
        n_isolates = records['assembly_id'].nunique()
        complete = group[group['is_complete_cds']]
        if complete.empty:
            raise ValueError(
                f"summarize_cds_lengths: {function!r} has no complete CDS in population "
                f"{population!r}, so its length statistics are undefined.")
        lengths = complete['cds_length']
        mode = modal_length(lengths)

        # Isolate shares, over EVERY isolate carrying this gene. `frac at mode` cannot see an
        # incomplete record, because its denominator is the complete sequences. These two can,
        # and the pair separates the two ways a gene fails: no complete CDS, or the wrong length.
        isolates_complete = records.loc[records['is_complete_cds'], 'assembly_id'].nunique()
        at_mode = records['is_complete_cds'] & (records['cds_length'] == mode['mode'])
        isolates_at_mode = records.loc[at_mode, 'assembly_id'].nunique()

        rows.append({
            'population': population,
            'Segment ID': segment_number(group['canonical_segment'].iloc[0]),
            'protein': function_to_short.get(function, function),
            'isolates': n_isolates,
            'unique CDS': len(group),
            'complete CDS': len(complete),
            'min': int(lengths.min()),
            'max': int(lengths.max()),
            'median': float(lengths.median()),
            'mode': mode['mode'],
            'complete CDS at mode': mode['n_at_mode'],
            'frac at mode': mode['frac_at_mode'],
            'frac isolates complete':
                isolates_complete / n_isolates if n_isolates else float('nan'),
            'frac isolates at mode': isolates_at_mode / n_isolates if n_isolates else float('nan'),
            'mode tie': mode['tied'],
        })
    table = pd.DataFrame(rows, columns=COLUMNS).sort_values('Segment ID').reset_index(drop=True)
    return table


def merge_population_rows(table: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Combine `table` with the rows already on disk, replacing the populations it carries.

    One survey covers one population, and the populations of interest accumulate over time, so
    they share a file keyed on the `population` column rather than each taking a file of their
    own. Re-running a population replaces its rows, so the file never carries a population twice.

    The read uses `keep_default_na=False, na_values=['']` because the `protein` column holds the
    literal string `NA` (Neuraminidase), which a default read parses as NaN and so drops.

    Args:
      table: the survey rows for the population just computed.
      out_path: CSV holding the populations surveyed so far; need not exist.

    Returns:
      The combined rows, ordered by population then segment, which makes the file content
      independent of the order the populations were run in.
    """
    if out_path.exists():
        existing = pd.read_csv(out_path, keep_default_na=False, na_values=[''])
        kept = existing[~existing['population'].isin(table['population'])]
        combined = pd.concat([kept, table], ignore_index=True)
    else:
        combined = table
    ordered = combined.sort_values(['population', 'Segment ID']).reset_index(drop=True)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--cds_final', type=Path,
                        default=PROJ / 'data/processed/flu/July_2025/cds_dna_final.parquet',
                        help='cds_dna_final.parquet to survey')
    parser.add_argument('--config_bundle', default='flu_ha_na',
                        help='bundle supplying the protein short names')
    parser.add_argument('--hn_subtype', nargs='+', default=None,
                        help='subtype filter, e.g. H3N2 (repeatable)')
    parser.add_argument('--host', nargs='+', default=None,
                        help='host filter, e.g. Human (repeatable)')
    parser.add_argument('--year', nargs='+', type=int, default=None,
                        help='year filter, e.g. 2024 (repeatable)')
    parser.add_argument('--year_range', nargs=2, type=int, default=None,
                        metavar=('MIN', 'MAX'),
                        help='inclusive year range; mutually exclusive with --year')
    parser.add_argument('--population', default=None,
                        help='label for the population; defaults to the filters applied')
    parser.add_argument('--out_dir', type=Path,
                        default=PROJ / 'results/flu/July_2025/cds_length_survey')
    args = parser.parse_args()

    config = get_virus_config_hydra(args.config_bundle, config_path=str(PROJ / 'conf'))
    function_to_short = get_function_short_name_map(config)

    cds = pd.read_parquet(args.cds_final, columns=[
        'assembly_id', 'function', 'canonical_segment', 'cds_dna_hash', 'cds_length',
        'is_complete_cds'])
    print(f"Read {len(cds):,} rows from {args.cds_final}")

    filtered = any(v is not None for v in
                   (args.hn_subtype, args.host, args.year, args.year_range))
    if filtered:
        # cds_dna_final carries no subtype, host or year, so they come from the isolate table.
        cds = filter_by_metadata(
            attach_isolate_metadata(cds, project_root=PROJ),
            hn_subtype=args.hn_subtype, host=args.host, year=args.year,
            year_range=args.year_range)
        if cds.empty:
            raise ValueError('No records match the requested metadata filters.')
        print(f"After metadata filtering: {len(cds):,} rows, "
              f"{cds['assembly_id'].nunique():,} isolates")

    population = args.population or population_label(
        args.hn_subtype, args.host, args.year, args.year_range)
    table = summarize_cds_lengths(cds, function_to_short, population=population)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / 'cds_length_survey.csv'
    combined = merge_population_rows(table, out_path)
    combined.to_csv(out_path, index=False)

    shown = table.copy()
    for column in ('frac at mode', 'frac isolates complete', 'frac isolates at mode'):
        shown[column] = shown[column].map(lambda v: f'{v:.3f}')
    print()
    print(shown.to_string(index=False))
    if not filtered:
        print("\nThis population spans every subtype, host and year. A mode taken across subtypes"
              "\nis a mixture, so read it as a screen rather than as a length to pin.")
    print("\nScreen on 'frac isolates at mode'. 'frac at mode' divides by the complete CDS, so it"
          "\ncannot see a gene whose records are mostly incomplete; 'frac isolates complete'"
          "\nreports that failure on its own. All counts are CDS DNA, not protein.")
    populations = combined['population'].nunique()
    print(f"\nWrote {out_path} ({len(combined)} rows, {populations} "
          f"population{'s' if populations != 1 else ''})")
    print('Done.')


if __name__ == '__main__':
    main()
