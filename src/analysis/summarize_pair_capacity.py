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

`HK matched` is how many positives survive the unique-sequence constraint, where no slot-A sequence
and no slot-B sequence is used twice. HK is Hopcroft-Karp, which returns a maximum matching, so it
is the largest such set; the sequential-dedup selectors in `_positive_pair_selection` retain fewer.
This script runs only that selector, so the column needs no qualifier beyond naming it.

`isolate_jaccard` records how comparable two schema pairs really are. Each matching is solved on
its own bigraph, so two pairs sharing a protein do not keep the same isolates. On human H3N2 2024,
PA-NP and NP-NA overlap on only 0.470 of the isolates they retain.

`Pair ID` names a pair by its two segment numbers, so PB2-HA is `1-4`. `ID` is a row counter over
the table as sorted, which is by descending `HK matched`, so it is a rank rather than a stable
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
    canonical_protein_pairs,
    get_function_short_name_map,
    get_virus_config_hydra,
)

CAPACITY_COLUMNS = ['ID', 'Pair ID', 'pair', 'population', 'eligible isolates', 'positives',
                    'Unique slot-A', 'Unique slot-B', 'HK matched', 'HK share', 'min-count sample']
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


def summarize_pair_capacity(kept: pd.DataFrame, proteins: list, function_to_short: dict,
                            canonical_order: list, pair_key_alphabet: str,
                            population: str = 'all', cohort_mode: str = 'pair',
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Positive and matched-positive counts for every schema pair over `proteins`.

    Args:
      kept: front-end rows already filtered to the population and to complete CDS at the pinned
          length, before any cohort is taken.
      proteins: short protein names to pair up, taken two at a time.
      function_to_short: full function name -> short protein name.
      canonical_order: short names in canonical order, fixing each pair's slot A / slot B.
      pair_key_alphabet: alphabet the positive dedup keys on, e.g. `nt_cds`.
      population: label describing what `kept` was filtered to.
      cohort_mode: `pair` builds each pair from the isolates carrying its own two proteins;
          `common` builds every pair from the isolates carrying all of `proteins`.

    Returns:
      The capacity table, ordered by descending `HK matched`, and the isolate-overlap table.

    Raises:
      ValueError: `cohort_mode` is not `pair` or `common`.
    """
    if cohort_mode not in ('pair', 'common'):
        raise ValueError(f"cohort_mode must be 'pair' or 'common'; got {cohort_mode!r}.")
    full_of = {short: full for full, short in function_to_short.items()}
    hash_col_a, hash_col_b = schema.hash_col_ab(pair_key_alphabet)
    segment_of = segment_numbers(kept, function_to_short)

    # Under `common` every pair draws on the same isolates, which costs a pair the isolates that
    # are missing a protein it does not contain. On human H3N2 2024 requiring all 8 proteins
    # leaves 2,922 isolates against HA-NA's own 5,173, because only 2,945 have a complete PB1.
    if cohort_mode == 'common':
        shared = common_isolate_cohort(kept, proteins, function_to_short)

    rows = []
    retained_isolates = {}
    for protein_a, protein_b in canonical_protein_pairs(proteins, canonical_order):
        label = f'{protein_a}-{protein_b}'
        eligible = (shared if cohort_mode == 'common'
                    else common_isolate_cohort(kept, [protein_a, protein_b], function_to_short))
        cohort = kept[kept['assembly_id'].isin(eligible)]
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
            'eligible isolates': len(eligible),
            'positives': len(positives),
            'Unique slot-A': int(positives[hash_col_a].nunique()),
            'Unique slot-B': int(positives[hash_col_b].nunique()),
            'HK matched': len(matched),
            'HK share': len(matched) / len(positives) if len(positives) else float('nan'),
        })

    table = pd.DataFrame(rows).sort_values('HK matched', ascending=False).reset_index(drop=True)
    # The smallest HK count is the largest sample every pair could supply. Experiment 3 of
    # docs/plans/2026-09-14_cross_year_importance_all_pairs_alignment_plan.md trains on each
    # pair's own HK population instead, so this column reports the floor rather than what is run.
    table['min-count sample'] = table['HK matched'].min()
    table.insert(0, 'ID', range(1, len(table) + 1))
    return table[CAPACITY_COLUMNS], isolate_overlap(retained_isolates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config_bundle',
                        default='flu_8_major_proteins_human_h3n2_2024_pinned_length',
                        help='bundle supplying the pinned lengths and the pair_key alphabet')
    parser.add_argument('--proteins', nargs='+', default=None,
                        help='short protein names to pair up, taken two at a time; '
                             'default: the virus config\'s selected_functions')
    parser.add_argument('--hn_subtype', nargs='+', default=None)
    parser.add_argument('--host', nargs='+', default=None)
    parser.add_argument('--year', nargs='+', type=int, default=None)
    parser.add_argument('--year_range', nargs=2, type=int, default=None, metavar=('MIN', 'MAX'))
    parser.add_argument('--population', default=None,
                        help='label for the population; defaults to the filters applied')
    parser.add_argument('--cohort', choices=['pair', 'common'], default='pair',
                        help="'pair' builds each schema pair from the isolates carrying its own "
                             "two proteins; 'common' builds every pair from the isolates carrying "
                             "all --proteins")
    parser.add_argument('--out_dir', type=Path,
                        default=PROJ / 'results/flu/July_2025/pair_capacity')
    args = parser.parse_args()

    config = get_virus_config_hydra(args.config_bundle, config_path=str(PROJ / 'conf'))
    # build_frontend reads its metadata filters off the config, so set them there.
    # Only a filter given on the command line overrides the bundle, so a bundle can carry the
    # population it was written for. Setting them unconditionally cleared the bundle's year and
    # made check_cds_length see every year at once.
    for name in ('hn_subtype', 'host', 'year', 'year_range'):
        given = getattr(args, name)
        if given is not None:
            setattr(config.dataset, name, given)

    function_to_short = get_function_short_name_map(config)
    canonical_order = [function_to_short[f] for f in config.virus.protein_order]
    # The proteins default to the ones the virus config selects for modelling, which needs the
    # config, so it cannot be an argparse default.
    if args.proteins is None:
        args.proteins = [function_to_short[f] for f in config.virus.selected_functions]
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

    n_population = kept['assembly_id'].nunique()
    if args.cohort == 'common':
        shared = common_isolate_cohort(kept, args.proteins, function_to_short)
        print(f"\nCommon cohort: {len(shared):,} of {n_population:,} isolates carry all "
              f"{len(args.proteins)} proteins at their pinned length")
    else:
        print(f"\nPair-specific cohorts over {n_population:,} isolates: each pair keeps the "
              f"isolates carrying its own two proteins at their pinned length")

    population = args.population or ' '.join(
        str(v) for values in (config.dataset.hn_subtype, config.dataset.host, config.dataset.year)
        if values for v in values)
    capacity, overlap = summarize_pair_capacity(
        kept, args.proteins, function_to_short, canonical_order, pair_key_alphabet,
        population=population or 'all', cohort_mode=args.cohort)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    capacity_path = args.out_dir / 'pair_capacity.csv'
    overlap_path = args.out_dir / 'pair_isolate_overlap.csv'
    capacity.to_csv(capacity_path, index=False)
    overlap.to_csv(overlap_path, index=False)

    shown = capacity.copy()
    shown['HK share'] = shown['HK share'].map('{:.1%}'.format)
    print()
    print(shown.to_string(index=False))
    print(f"\nHopcroft-Karp matched positives: min {capacity['HK matched'].min():,}, "
          f"median {int(capacity['HK matched'].median()):,}, "
          f"max {capacity['HK matched'].max():,}. Equalizing the count across every pair would "
          f"cap it at {capacity['HK matched'].min():,}.")
    print(f"Isolate overlap between matchings: median Jaccard "
          f"{overlap['isolate jaccard'].median():.3f}. Two pairs are less comparable than a shared "
          f"cohort suggests, because each matching keeps its own isolates.")
    print(f"\nWrote {capacity_path}")
    print(f"Wrote {overlap_path}")
    print('Done.')


if __name__ == '__main__':
    main()
