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

`distinct lengths` is the column that answers the viability question fastest. A protein with a
handful of lengths has a dominant form; one with many is a mixture, and `min`/`max` alone will not
tell the two apart.

SCOPE. This reads the whole corpus, which spans every subtype and year. HA is 1,701 nt in H3N2 but
1,704 in H5N1 and 1,683 in H9/H7, so a corpus-wide mode is a mixture rather than a fact about any
one population. Read the output as a screen for which pairs look viable, NOT as a length to pin.
`summarize_cds_lengths` takes an already-filtered frame, so narrowing to one subtype or year later
is a matter of filtering before the call and passing a `population` label; the summary itself needs
no change.

Outputs (to `--out_dir`):
    cds_length_survey.csv   one row per protein: population, Segment ID, protein, unique seqs,
                            complete seqs, min, max, median, mode, seqs at mode, frac at mode,
                            distinct lengths, mode tie

CLI:
    python -m src.analysis.summarize_cds_lengths
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

COLUMNS = ['population', 'Segment ID', 'protein', 'unique seqs', 'complete seqs',
           'min', 'max', 'median', 'mode', 'seqs at mode', 'frac at mode',
           'distinct lengths', 'mode tie']


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

    Args:
      cds: rows from `cds_dna_final`, needing `function`, `canonical_segment`, `cds_dna_hash`,
          `cds_length` and `is_complete_cds`.
      function_to_short: full function name -> short protein name; a name absent from the map is
          kept in full.
      population: label describing what `cds` was filtered to.

    Returns:
      One row per protein, ordered by segment number, with the columns in `COLUMNS`.

    Raises:
      ValueError: a required column is missing, or a protein has no complete sequence.
    """
    required = {'function', 'canonical_segment', 'cds_dna_hash', 'cds_length', 'is_complete_cds'}
    missing = required - set(cds.columns)
    if missing:
        raise ValueError(f"summarize_cds_lengths: missing columns {sorted(missing)}.")

    # One row per distinct sequence, before any statistic is taken.
    unique = cds.drop_duplicates('cds_dna_hash')
    rows = []
    for function, group in unique.groupby('function'):
        complete = group[group['is_complete_cds']]
        if complete.empty:
            raise ValueError(
                f"summarize_cds_lengths: {function!r} has no complete CDS in population "
                f"{population!r}, so its length statistics are undefined.")
        lengths = complete['cds_length']
        mode = modal_length(lengths)
        rows.append({
            'population': population,
            'Segment ID': segment_number(group['canonical_segment'].iloc[0]),
            'protein': function_to_short.get(function, function),
            'unique seqs': len(group),
            'complete seqs': len(complete),
            'min': int(lengths.min()),
            'max': int(lengths.max()),
            'median': float(lengths.median()),
            'mode': mode['mode'],
            'seqs at mode': mode['n_at_mode'],
            'frac at mode': mode['frac_at_mode'],
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
    parser.add_argument('--population', default='all',
                        help='label for the population being surveyed')
    parser.add_argument('--out_dir', type=Path,
                        default=PROJ / 'results/flu/July_2025/cds_length_survey')
    args = parser.parse_args()

    config = get_virus_config_hydra(args.config_bundle, config_path=str(PROJ / 'conf'))
    function_to_short = get_function_short_name_map(config)

    cds = pd.read_parquet(args.cds_final, columns=[
        'function', 'canonical_segment', 'cds_dna_hash', 'cds_length', 'is_complete_cds'])
    print(f"Read {len(cds):,} rows from {args.cds_final}")

    table = summarize_cds_lengths(cds, function_to_short, population=args.population)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / 'cds_length_survey.csv'
    table.to_csv(out_path, index=False)

    shown = table.copy()
    shown['frac at mode'] = shown['frac at mode'].map(lambda v: f'{v:.3f}')
    print()
    print(shown.to_string(index=False))
    print(f"\nPopulation {args.population!r} spans every subtype and year unless it was filtered "
          f"first.\nA mode taken across subtypes is a mixture, so read this as a screen for which "
          f"proteins\nhave a dominant length, not as a length to pin.")
    print(f"\nWrote {out_path}")
    print('Done.')


if __name__ == '__main__':
    main()
