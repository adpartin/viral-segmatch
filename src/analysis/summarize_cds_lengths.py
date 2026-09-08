"""Per-protein CDS completeness and length survey, one row per major protein.

Written to decide which of the 28 schema pairs are worth pursuing for per-site features. Those
features need every sequence at one length, so a pair is only viable when both of its proteins
have a dominant length. This reports what each protein's length distribution actually looks like
so that question can be answered before a run is attempted.

Every statistic counts each distinct CDS sequence once, keyed on `cds_dna_hash`. Counting rows
instead would let a heavily sampled strain decide the answer on its own, since one sequence
appears once per isolate carrying it.

`min`, `max`, `median` and `mode` describe COMPLETE sequences only. A sequence is complete when
`is_complete_cds` holds, which is `starts_with_m & has_terminal_stop & ~has_internal_stop`.

Two columns answer the viability question, and they can disagree sharply. `frac at mode` is the
share of distinct COMPLETE sequences at the modal length. `frac isolates at mode` is the share of
ISOLATES whose CDS is complete and at that length, which is what a dataset is actually built from.
On human H3N2 2024, PB1 reads 0.942 by sequence but 0.551 by isolate, because 45% of its records
have no terminal stop and so are unusable however common they are. Read the isolate column before
concluding that a protein can be pinned.

`distinct lengths` answers neither. PB2 and PB1 both have 23 distinct lengths corpus-wide but sit
at 0.998 and 0.927, and HA has fewer distinct lengths than PB2 yet reaches only 0.689. A count of
lengths says nothing about whether one of them dominates.

SCOPE. This reads the whole corpus, which spans every subtype and year. HA is 1,701 nt in H3N2 but
1,704 in H5N1 and 1,683 in H9/H7, so a corpus-wide mode is a mixture rather than a fact about any
one population. Read the output as a screen for which pairs look viable, NOT as a length to pin.
`summarize_cds_lengths` takes an already-filtered frame, so narrowing to one subtype or year later
is a matter of filtering before the call and passing a `population` label; the summary itself needs
no change.

Outputs (to `--out_dir`):
    cds_length_survey.csv   one row per protein, with the columns in `COLUMNS`

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

COLUMNS = ['population', 'Segment ID', 'protein', 'isolates', 'unique seqs', 'complete seqs',
           'min', 'max', 'median', 'mode', 'seqs at mode', 'frac at mode',
           'frac isolates at mode', 'distinct lengths', 'mode tie']


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

        # The share of ISOLATES usable at the modal length, which is what a dataset is built
        # from. It can sit far below `frac at mode`: a length carried by few distinct sequences
        # may still cover many isolates, and an incomplete CDS is unusable however common it is.
        at_mode = records['is_complete_cds'] & (records['cds_length'] == mode['mode'])
        isolates_at_mode = records.loc[at_mode, 'assembly_id'].nunique()

        rows.append({
            'population': population,
            'Segment ID': segment_number(group['canonical_segment'].iloc[0]),
            'protein': function_to_short.get(function, function),
            'isolates': n_isolates,
            'unique seqs': len(group),
            'complete seqs': len(complete),
            'min': int(lengths.min()),
            'max': int(lengths.max()),
            'median': float(lengths.median()),
            'mode': mode['mode'],
            'seqs at mode': mode['n_at_mode'],
            'frac at mode': mode['frac_at_mode'],
            'frac isolates at mode': isolates_at_mode / n_isolates if n_isolates else float('nan'),
            'distinct lengths': int(lengths.nunique()),
            'mode tie': mode['tied'],
        })
    table = pd.DataFrame(rows, columns=COLUMNS).sort_values('Segment ID').reset_index(drop=True)
    return table


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
    table.to_csv(out_path, index=False)

    shown = table.copy()
    for column in ('frac at mode', 'frac isolates at mode'):
        shown[column] = shown[column].map(lambda v: f'{v:.3f}')
    print()
    print(shown.to_string(index=False))
    if not filtered:
        print("\nThis population spans every subtype, host and year. A mode taken across subtypes"
              "\nis a mixture, so read it as a screen rather than as a length to pin.")
    print("\n'frac at mode' counts distinct sequences; 'frac isolates at mode' counts isolates."
          "\nThe isolate column is the one a dataset is built from, and it can be far lower.")
    print(f"\nWrote {out_path}")
    print('Done.')


if __name__ == '__main__':
    main()
