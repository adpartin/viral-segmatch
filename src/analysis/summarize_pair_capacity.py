"""How many positive pairs each schema pair can supply, on one shared isolate population.

Written to choose which schema pairs to train, after `summarize_cds_lengths` has said which
proteins can be pinned. That survey works per protein; this one works per pair, which is the unit
an experiment is actually built on.

Every pair is built on the same COMMON COHORT: the isolates carrying a complete CDS at the pinned
length for every protein under comparison. Holding the isolates fixed means host, subtype and year
are identical across pairs by construction, so a difference between two pairs is a difference
between the proteins rather than between the populations.

A common cohort does NOT give equal positive counts. Two isolates that share both proteins of a
pair collapse into one positive, and how often that happens depends on the proteins. On human H3N2
2024 the fifteen pairs over the six pinned proteins run from 2,293 to 3,723 positives despite
drawing on the same isolates.

`hopcroft_karp` reports how many positives survive the unique-sequence constraint, where no slot-A
sequence and no slot-B sequence is used twice. That is a maximum matching, so it is the largest
such set; the sequential-dedup selectors in `_positive_pair_selection` retain fewer.

`isolate_jaccard` records how comparable two schema pairs really are. Each matching is solved on
its own bigraph, so two pairs sharing a protein do not keep the same isolates. On human H3N2 2024,
PA-NP and NP-NA overlap on only 0.470 of the isolates they retain.

`Pair ID` names a pair by its two segment numbers, so PB2-HA is `1-4`. `ID` is a row counter over
the table as sorted, which is by descending matched count, so it is a rank rather than a stable
identifier and it moves when the population changes.

Outputs (to `--out_dir`):
    pair_capacity.csv           one row per schema pair, with the columns in `CAPACITY_COLUMNS`
    pair_isolate_overlap.csv    one row per pair of schema pairs: shared isolates and Jaccard

CLI:
    python -m src.analysis.summarize_pair_capacity --hn_subtype H3N2 --host Human --year 2024
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.summarize_cds_lengths import segment_number  # noqa: E402
from src.datasets._pair_helpers import filter_complete_cds_at_pinned_length  # noqa: E402
from src.datasets._positive_pair_selection import select_positive_pairs  # noqa: E402
from src.datasets.dataset_pairs_cc import build_frontend  # noqa: E402
from src.datasets.dataset_segment_pairs_v2 import create_positive_pairs_v2  # noqa: E402
from src.utils import schema  # noqa: E402
from src.utils.config_hydra import (  # noqa: E402
    get_function_short_name_map,
    get_virus_config_hydra,
)

DEFAULT_PROTEINS = ['PB2', 'PA', 'HA', 'NP', 'NA', 'M1']

CAPACITY_COLUMNS = ['ID', 'Pair ID', 'pair', 'population', 'cohort isolates', 'positives',
                    'distinct A', 'distinct B', 'matched', 'matched share']
OVERLAP_COLUMNS = ['pair A', 'pair B', 'isolates A', 'isolates B', 'shared', 'isolate jaccard']


def common_isolate_cohort(cds: pd.DataFrame, proteins: list, function_to_short: dict) -> set:
    """Isolates carrying a record for every one of `proteins`.

    Args:
      cds: rows already filtered to usable records, needing `assembly_id` and `function`.
      proteins: short protein names every isolate must carry.
      function_to_short: full function name -> short protein name.

    Returns:
      The `assembly_id` values present for all of `proteins`.

    Raises:
      ValueError: a requested protein appears in no row.
    """
    short = cds['function'].map(function_to_short)
    absent = [p for p in proteins if p not in set(short)]
    if absent:
        raise ValueError(f"common_isolate_cohort: no rows for {absent}.")

    carried = cds.assign(short=short).groupby('assembly_id')['short'].agg(set)
    wanted = set(proteins)
    return set(carried[carried.map(lambda held: wanted <= held)].index)


def segment_numbers(cds: pd.DataFrame, function_to_short: dict) -> dict:
    """Map each short protein name to its segment number.

    Args:
      cds: rows carrying `function` and `canonical_segment`.
      function_to_short: full function name -> short protein name.

    Returns:
      Short protein name -> segment number, 1 through 8.

    Raises:
      ValueError: one protein carries more than one `canonical_segment`.
    """
    seen = {}
    for function, group in cds.groupby('function'):
        labels = set(group['canonical_segment'])
        if len(labels) != 1:
            raise ValueError(
                f"segment_numbers: {function!r} spans several segments {sorted(labels)}.")
        seen[function_to_short.get(function, function)] = segment_number(labels.pop())
    return seen


def isolate_overlap(retained_isolates: dict) -> pd.DataFrame:
    """Shared-isolate counts and Jaccard for every combination of two schema pairs.

    Args:
      retained_isolates: pair label -> the `assembly_id` values its matching kept.

    Returns:
      One row per unordered combination, with the columns in `OVERLAP_COLUMNS`, ordered by
      descending Jaccard.
    """
    rows = []
    for (label_a, isolates_a), (label_b, isolates_b) in itertools.combinations(
            sorted(retained_isolates.items()), 2):
        shared = len(isolates_a & isolates_b)
        union = len(isolates_a | isolates_b)
        rows.append({
            'pair A': label_a,
            'pair B': label_b,
            'isolates A': len(isolates_a),
            'isolates B': len(isolates_b),
            'shared': shared,
            'isolate jaccard': shared / union if union else float('nan'),
        })
    table = pd.DataFrame(rows, columns=OVERLAP_COLUMNS)
    return table.sort_values('isolate jaccard', ascending=False).reset_index(drop=True)


def summarize_pair_capacity(cohort: pd.DataFrame, proteins: list, function_to_short: dict,
                            pair_key_alphabet: str, population: str = 'all',
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Positive and matched-positive counts for every schema pair over `proteins`.

    Args:
      cohort: front-end rows restricted to the common isolate cohort.
      proteins: short protein names to pair up, taken two at a time.
      function_to_short: full function name -> short protein name.
      pair_key_alphabet: alphabet the positive dedup keys on, e.g. `nt_cds`.
      population: label describing what `cohort` was filtered to.

    Returns:
      The capacity table, ordered by descending matched count, and the isolate-overlap table.
    """
    full_of = {short: full for full, short in function_to_short.items()}
    hash_col_a, hash_col_b = schema.hash_col_ab(pair_key_alphabet)
    n_cohort = cohort['assembly_id'].nunique()
    segment_of = segment_numbers(cohort, function_to_short)

    rows = []
    retained_isolates = {}
    for protein_a, protein_b in itertools.combinations(proteins, 2):
        label = f'{protein_a}-{protein_b}'
        positives, _ = create_positive_pairs_v2(
            cohort, schema_pair=(full_of[protein_a], full_of[protein_b]),
            pair_key_alphabet=pair_key_alphabet)
        matched, _ = select_positive_pairs(
            positives, 'hopcroft_karp', hash_col_a, hash_col_b)

        retained_isolates[label] = set(matched['assembly_id_a'])
        rows.append({
            # Segment numbers follow the pair order, so `Pair ID` and `pair` always agree.
            'Pair ID': f'{segment_of[protein_a]}-{segment_of[protein_b]}',
            'pair': label,
            'population': population,
            'cohort isolates': n_cohort,
            'positives': len(positives),
            'distinct A': int(positives[hash_col_a].nunique()),
            'distinct B': int(positives[hash_col_b].nunique()),
            'matched': len(matched),
            'matched share': len(matched) / len(positives) if len(positives) else float('nan'),
        })

    table = pd.DataFrame(rows).sort_values('matched', ascending=False).reset_index(drop=True)
    table.insert(0, 'ID', range(1, len(table) + 1))
    return table[CAPACITY_COLUMNS], isolate_overlap(retained_isolates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config_bundle', default='flu_ha_na_h3n2_2024_random_cv4_pinned_length',
                        help='bundle supplying the pinned lengths and the pair_key alphabet')
    parser.add_argument('--proteins', nargs='+', default=DEFAULT_PROTEINS,
                        help='short protein names to pair up, taken two at a time')
    parser.add_argument('--hn_subtype', nargs='+', default=['H3N2'])
    parser.add_argument('--host', nargs='+', default=['Human'])
    parser.add_argument('--year', nargs='+', type=int, default=None)
    parser.add_argument('--year_range', nargs=2, type=int, default=None, metavar=('MIN', 'MAX'))
    parser.add_argument('--population', default=None,
                        help='label for the population; defaults to the filters applied')
    parser.add_argument('--out_dir', type=Path,
                        default=PROJ / 'results/flu/July_2025/pair_capacity')
    args = parser.parse_args()

    config = get_virus_config_hydra(args.config_bundle, config_path=str(PROJ / 'conf'))
    # build_frontend reads its metadata filters off the config, so set them there.
    config.dataset.hn_subtype = args.hn_subtype
    config.dataset.host = args.host
    config.dataset.year = args.year
    config.dataset.year_range = args.year_range

    function_to_short = get_function_short_name_map(config)
    full_of = {short: full for full, short in function_to_short.items()}
    unknown = [p for p in args.proteins if p not in full_of]
    if unknown:
        raise ValueError(f"--proteins not in the virus config: {unknown}.")

    pins = {str(k): int(v['nt']) for k, v in dict(config.virus.cds_length).items()}
    unpinned = [p for p in args.proteins if p not in pins]
    if unpinned:
        raise ValueError(
            f"{unpinned} have no pinned length in conf/virus/{config.virus.virus_name}.yaml. "
            f"Per-site features need one length per protein; pin them for this population first.")

    pair_key_alphabet = str(config.dataset.split_strategy.pair_key_alphabet)
    prot_path = PROJ / f'data/processed/{config.virus.virus_name}/{config.virus.data_version}'
    cds_path = prot_path / 'cds_dna_final.parquet'

    # One front-end for every protein at once, so all pairs see the same isolates.
    frontend = build_frontend(config, prot_path / 'protein_final.parquet',
                              tuple(full_of[p] for p in args.proteins), cds_final_path=cds_path)
    kept, _ = filter_complete_cds_at_pinned_length(frontend, cds_path, pins, function_to_short)

    cohort_ids = common_isolate_cohort(kept, args.proteins, function_to_short)
    cohort = kept[kept['assembly_id'].isin(cohort_ids)]
    print(f"\nCommon cohort: {len(cohort_ids):,} of {kept['assembly_id'].nunique():,} isolates "
          f"carry all {len(args.proteins)} proteins at their pinned length")

    population = args.population or ' '.join(
        str(v) for values in (args.hn_subtype, args.host, args.year) if values for v in values)
    capacity, overlap = summarize_pair_capacity(
        cohort, args.proteins, function_to_short, pair_key_alphabet, population=population or 'all')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    capacity_path = args.out_dir / 'pair_capacity.csv'
    overlap_path = args.out_dir / 'pair_isolate_overlap.csv'
    capacity.to_csv(capacity_path, index=False)
    overlap.to_csv(overlap_path, index=False)

    shown = capacity.copy()
    shown['matched share'] = shown['matched share'].map('{:.1%}'.format)
    print()
    print(shown.to_string(index=False))
    print(f"\nMatched positives: min {capacity['matched'].min():,}, "
          f"median {int(capacity['matched'].median()):,}, max {capacity['matched'].max():,}. "
          f"Equalizing the count across every pair would cap it at {capacity['matched'].min():,}.")
    print(f"Isolate overlap between matchings: median Jaccard "
          f"{overlap['isolate jaccard'].median():.3f}. Two pairs are less comparable than a shared "
          f"cohort suggests, because each matching keeps its own isolates.")
    print(f"\nWrote {capacity_path}")
    print(f"Wrote {overlap_path}")
    print('Done.')


if __name__ == '__main__':
    main()
