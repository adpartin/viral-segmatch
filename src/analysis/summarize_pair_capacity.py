"""How many positive pairs each schema pair can supply, and how sequence-unique they are.

Run after `summarize_cds_lengths` has said which proteins can be pinned. That survey works per
protein; this one works per pair, which is the unit an experiment is built on.

How?

- Each pair keeps the isolates carrying its own two proteins as a complete CDS at the pinned
  length. `--cohort common` instead gives every pair the isolates carrying all `--proteins`, which
  costs a pair the isolates missing a protein it does not contain.
- Pairs are enumerated in canonical protein order, so `Pair ID` reads as segment numbers: PB2-HA
  is `1-4`.
- `Unique positives` counts observed same-isolate pairs after deduplicating on the `nt_cds` pair
  key.
- `HK selected` counts the positives left once no slot-A and no slot-B sequence is used twice.
  Hopcroft-Karp returns a maximum matching, so it is the largest such set; the sequential-dedup
  selectors in `_positive_pair_selection` retain fewer. One positive per unique sequence also
  means `HK selected` cannot exceed the smaller of `Unique slot-A` and `Unique slot-B`.
- `pair_sequence_reuse.csv` summarizes how often each unique sequence recurs across a pair's
  positives, two rows per pair and one per slot. Reuse says how concentrated the positives are and
  helps explain a low `HK share`, since many edges compete for the one a sequence can keep. How
  close the matching comes to its ceiling depends on the whole bigraph, not on reuse alone.
- `pair_isolate_overlap.csv` records how far two pairs' retained isolates agree, because each
  matching is solved on its own bigraph and keeps its own isolates. `shares protein` marks the
  combinations whose two schema pairs have a protein in common.
- `pair_capacity_matrix.csv` and its heatmap lay `HK selected` out protein by protein.

CLI:
    python -m src.analysis.summarize_pair_capacity
    python -m src.analysis.summarize_pair_capacity --proteins HA NA PB2 PA --cohort common

Notes:

- The default bundle carries pins for all 8 major proteins, two of which are not in
  `conf/virus/flu.yaml`. A bundle without a pin for every protein passed raises, naming the ones
  it lacks.
- A metadata filter given on the command line overrides the bundle; otherwise the bundle's own
  population stands.
- `min-count sample` is the smallest `HK selected` in the table, so it moves when the set of pairs
  changes. It reports the floor, not what any experiment trains on.
- `ID` is a row counter over the table as sorted, which is by descending `HK selected`. It is a
  rank rather than a stable identifier and it moves when the population changes.
- Reuse is counted after the pair-key dedup, so a sequence's reuse count is the number of unique
  partner sequences it was observed with, not the number of isolates it occurs in.

Outputs (to `--out_dir`):
    pair_capacity.csv            one row per schema pair, with the columns in `CAPACITY_COLUMNS`
    pair_capacity_by_segment.csv the same rows in segment order rather than by matched count
    pair_sequence_reuse.csv      two rows per schema pair, one per slot, with `REUSE_COLUMNS`
    pair_isolate_overlap.csv     one row per pair of schema pairs: shared isolates and Jaccard
    pair_capacity_matrix.csv     `HK selected` as a symmetric protein-by-protein table
    pair_capacity_matrix.png     that table as a heatmap
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
from src.utils.metadata_enrichment import population_label  # noqa: E402
from src.utils.plot_utils import savefig, setup_plot_style  # noqa: E402

CAPACITY_COLUMNS = ['ID', 'Pair ID', 'Schema pair', 'population', 'Eligible isolates',
                    'Unique positives', 'Unique slot-A', 'Unique slot-B', 'HK selected',
                    'HK share', 'min-count sample']
OVERLAP_COLUMNS = ['pair A', 'pair B', 'shares protein', 'isolates A', 'isolates B', 'shared',
                   'isolate jaccard']
REUSE_COLUMNS = ['Schema pair', 'slot', 'protein', 'Unique positives', 'unique sequences',
                 'reuse mean', 'reuse median', 'reuse p90', 'reuse max', 'singleton share']


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
      retained_isolates: (protein A, protein B) -> the `assembly_id` values its matching kept.

    Returns:
      One row per unordered combination, with the columns in `OVERLAP_COLUMNS`, ordered by
      descending Jaccard. `shares protein` is True where the two schema pairs have a protein in
      common, which is the subset whose matchings draw on overlapping sequences.
    """
    rows = []
    for (pair_a, isolates_a), (pair_b, isolates_b) in itertools.combinations(
            sorted(retained_isolates.items()), 2):
        shared = len(isolates_a & isolates_b)
        union = len(isolates_a | isolates_b)
        rows.append({
            'pair A': '-'.join(pair_a),
            'pair B': '-'.join(pair_b),
            'shares protein': bool(set(pair_a) & set(pair_b)),
            'isolates A': len(isolates_a),
            'isolates B': len(isolates_b),
            'shared': shared,
            'isolate jaccard': shared / union if union else float('nan'),
        })
    table = pd.DataFrame(rows, columns=OVERLAP_COLUMNS)
    return table.sort_values('isolate jaccard', ascending=False).reset_index(drop=True)


def sequence_reuse(positives: pd.DataFrame, hash_col: str, label: str, slot: str,
                   protein: str) -> dict:
    """How often each unique sequence in one slot recurs across a pair's positives.

    `positives` is already deduplicated on the pair key, so one row is one unique pair of
    sequences and a sequence's count is the number of unique partner sequences it was observed
    with. That is smaller than the number of isolates it occurs in wherever one sequence pair
    recurs across isolates.

    Args:
      positives: the pair's deduplicated positive pairs, before matching.
      hash_col: the slot's sequence-hash column in `positives`.
      label: the pair label, e.g. `HA-NA`.
      slot: `A` or `B`.
      protein: the short protein name filling the slot.

    Returns:
      One row of `REUSE_COLUMNS`.

    Raises:
      ValueError: `positives` is empty, so there is no distribution to describe.
    """
    per_sequence = positives[hash_col].value_counts()
    if per_sequence.empty:
        raise ValueError(f"sequence_reuse: {label} slot {slot} has no positives.")
    return {
        'Schema pair': label,
        'slot': slot,
        'protein': protein,
        'Unique positives': len(positives),
        'unique sequences': len(per_sequence),
        'reuse mean': per_sequence.mean(),
        'reuse median': per_sequence.median(),
        'reuse p90': per_sequence.quantile(0.9),
        'reuse max': int(per_sequence.max()),
        'singleton share': float((per_sequence == 1).mean()),
    }


def sort_by_segment(capacity: pd.DataFrame) -> pd.DataFrame:
    """The capacity table reordered by segment number instead of by matched count.

    Args:
      capacity: the table `summarize_pair_capacity` returns.

    Returns:
      The same rows ordered by the two segment numbers in `Pair ID`. `ID` still carries the
      selected-count rank, so it runs out of order in this view.
    """
    segment_numbers = capacity['Pair ID'].str.split('-').map(
        lambda parts: tuple(int(number) for number in parts))
    ordered = capacity.assign(segment_key=segment_numbers).sort_values('segment_key')
    return ordered.drop(columns='segment_key').reset_index(drop=True)


def hk_selected_matrix(capacity: pd.DataFrame, proteins: list,
                       canonical_order: list) -> pd.DataFrame:
    """`HK selected` laid out as a symmetric protein-by-protein table.

    Args:
      capacity: the table `summarize_pair_capacity` returns.
      proteins: short protein names indexing the rows and columns.
      canonical_order: short names in canonical order, fixing the row and column order.

    Returns:
      A square table of selected counts in canonical order. The diagonal is left empty, because a
      protein is not paired with itself.

    Raises:
      KeyError: `capacity` has no row for a pair drawn from `proteins`.
    """
    selected_of = dict(zip(capacity['Schema pair'], capacity['HK selected']))
    ordered = sorted(set(proteins), key=canonical_order.index)
    matrix = pd.DataFrame(float('nan'), index=ordered, columns=ordered)
    # Labels are rebuilt the way summarize_pair_capacity built them rather than split on '-',
    # because several short protein names contain a hyphen (PA-X, PB1-F2).
    for protein_a, protein_b in canonical_protein_pairs(proteins, canonical_order):
        selected_count = selected_of[f'{protein_a}-{protein_b}']
        matrix.loc[protein_a, protein_b] = selected_count
        matrix.loc[protein_b, protein_a] = selected_count
    return matrix


def plot_hk_selected_matrix(matrix: pd.DataFrame, out_path: Path, population: str) -> None:
    """Draw the selected-count matrix as an annotated heatmap.

    Args:
      matrix: the square table `hk_selected_matrix` returns.
      out_path: PNG file to write.
      population: population label for the title.

    Returns:
      None. Writes `out_path`.
    """
    setup_plot_style()
    # Small on purpose. The matrix is nearly square, so at the width GitHub gives a figure it
    # would otherwise run about 750 px tall and crowd out the table beside it.
    fig, ax = plt.subplots(figsize=(6.5, 5.3))
    sns.heatmap(matrix, annot=True, fmt='.0f', annot_kws={'size': 9}, cmap='YlGnBu',
                mask=matrix.isna(), square=True, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Hopcroft-Karp selected positives'})
    ax.set_title(f'Pair capacity, {population}', fontsize=12)
    ax.tick_params(labelsize=10)
    # setup_plot_style turns axes.grid on, which draws a line through every cell of a heatmap.
    ax.grid(False)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    for position in range(len(matrix)):
        ax.text(position + 0.5, position + 0.5, '--', ha='center', va='center', color='gray',
                fontsize=9)
    fig.tight_layout()
    savefig(out_path, dpi=110)


def summarize_pair_capacity(kept: pd.DataFrame, proteins: list, function_to_short: dict,
                            canonical_order: list, pair_key_alphabet: str,
                            population: str = 'all', cohort_mode: str = 'pair',
                            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
      The capacity table ordered by descending `HK selected`, the isolate-overlap table, and the
      sequence-reuse table.

    Raises:
      ValueError: `cohort_mode` is not `pair` or `common`.
    """
    if cohort_mode not in ('pair', 'common'):
        raise ValueError(f"cohort_mode must be 'pair' or 'common'; got {cohort_mode!r}.")
    full_of = {short: full for full, short in function_to_short.items()}
    hash_col_a, hash_col_b = schema.hash_col_ab(pair_key_alphabet)
    segment_of = segment_numbers(kept, function_to_short)

    # Under `common` every pair draws on the same isolates, which costs a pair the isolates that
    # are missing a protein it does not contain. On Human-H3N2-2024 requiring all 8 proteins
    # leaves 2,922 isolates against HA-NA's own 5,173, because only 2,945 have a complete PB1.
    if cohort_mode == 'common':
        shared = common_isolate_cohort(kept, proteins, function_to_short)

    rows = []
    reuse_rows = []
    retained_isolates = {}
    for protein_a, protein_b in canonical_protein_pairs(proteins, canonical_order):
        label = f'{protein_a}-{protein_b}'
        eligible = (shared if cohort_mode == 'common'
                    else common_isolate_cohort(kept, [protein_a, protein_b], function_to_short))
        cohort = kept[kept['assembly_id'].isin(eligible)]
        positives, _ = create_positive_pairs_v2(
            cohort, schema_pair=(full_of[protein_a], full_of[protein_b]),
            pair_key_alphabet=pair_key_alphabet)
        selected, _ = select_positive_pairs(
            positives, 'hopcroft_karp', hash_col_a, hash_col_b)

        retained_isolates[(protein_a, protein_b)] = set(selected['assembly_id_a'])
        reuse_rows.append(sequence_reuse(positives, hash_col_a, label, 'A', protein_a))
        reuse_rows.append(sequence_reuse(positives, hash_col_b, label, 'B', protein_b))
        rows.append({
            # Segment numbers follow the pair order, so `Pair ID` and `Schema pair` always agree.
            'Pair ID': f'{segment_of[protein_a]}-{segment_of[protein_b]}',
            'Schema pair': label,
            'population': population,
            'Eligible isolates': len(eligible),
            'Unique positives': len(positives),
            'Unique slot-A': int(positives[hash_col_a].nunique()),
            'Unique slot-B': int(positives[hash_col_b].nunique()),
            'HK selected': len(selected),
            'HK share': len(selected) / len(positives) if len(positives) else float('nan'),
        })

    table = pd.DataFrame(rows).sort_values('HK selected', ascending=False).reset_index(drop=True)
    # The smallest HK count is the largest sample every pair could supply. Experiment 3 of
    # docs/plans/2026-09-14_codon_site_features_plan.md trains on each pair's own `HK selected`
    # positives instead, so this column reports the floor rather than what is run.
    table['min-count sample'] = table['HK selected'].min()
    table.insert(0, 'ID', range(1, len(table) + 1))
    reuse = pd.DataFrame(reuse_rows, columns=REUSE_COLUMNS)
    return table[CAPACITY_COLUMNS], isolate_overlap(retained_isolates), reuse


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

    # One front-end for every protein at once, so all pairs are drawn from the same
    # metadata-filtered source population before their own cohorts are taken.
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

    # Built from the config rather than from `args`, so a bundle's own filters are named even when
    # nothing was passed on the command line. `population_label` returns 'all' for no filter, so
    # the result is never empty.
    # `geo_location` and `passage` are passed so that a bundle setting either raises here rather
    # than writing a label that does not say so; `build_frontend` forwards both to the filter.
    population = args.population or population_label(
        host=config.dataset.host, hn_subtype=config.dataset.hn_subtype,
        year=config.dataset.year, year_range=config.dataset.year_range,
        geo_location=getattr(config.dataset, 'geo_location', None),
        passage=getattr(config.dataset, 'passage', None))
    capacity, overlap, reuse = summarize_pair_capacity(
        kept, args.proteins, function_to_short, canonical_order, pair_key_alphabet,
        population=population, cohort_mode=args.cohort)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    capacity_path = args.out_dir / 'pair_capacity.csv'
    segment_path = args.out_dir / 'pair_capacity_by_segment.csv'
    reuse_path = args.out_dir / 'pair_sequence_reuse.csv'
    overlap_path = args.out_dir / 'pair_isolate_overlap.csv'
    matrix_path = args.out_dir / 'pair_capacity_matrix.csv'
    figure_path = args.out_dir / 'pair_capacity_matrix.png'
    capacity.to_csv(capacity_path, index=False)
    sort_by_segment(capacity).to_csv(segment_path, index=False)
    reuse.to_csv(reuse_path, index=False)
    overlap.to_csv(overlap_path, index=False)
    matrix = hk_selected_matrix(capacity, args.proteins, canonical_order)
    matrix.to_csv(matrix_path)
    plot_hk_selected_matrix(matrix, figure_path, population)

    shown = capacity.copy()
    shown['HK share'] = shown['HK share'].map('{:.1%}'.format)
    print()
    print(shown.to_string(index=False))
    print(f"\nHopcroft-Karp selected positives: min {capacity['HK selected'].min():,}, "
          f"median {int(capacity['HK selected'].median()):,}, "
          f"max {capacity['HK selected'].max():,}. Equalizing the count across every pair would "
          f"cap it at {capacity['HK selected'].min():,}.")
    with_shared_protein = overlap[overlap['shares protein']]
    print(f"Isolate overlap between matchings: median Jaccard "
          f"{overlap['isolate jaccard'].median():.3f} over all {len(overlap):,} combinations, and "
          f"{with_shared_protein['isolate jaccard'].median():.3f} over the "
          f"{len(with_shared_protein):,} whose schema pairs share a protein. Two pairs are less "
          f"comparable than a shared cohort suggests, because each matching keeps its own "
          f"isolates.")
    # The mean rather than the median, because the distribution is long-tailed: most sequences
    # are observed once, so the median is 1 for nearly every pair and slot.
    heaviest = reuse.loc[reuse['reuse mean'].idxmax()]
    print(f"Sequence reuse before matching is heaviest for {heaviest['protein']} in "
          f"{heaviest['Schema pair']}: {heaviest['unique sequences']:,} unique sequences over "
          f"{heaviest['Unique positives']:,} positives, so each pairs with "
          f"{heaviest['reuse mean']:.1f} unique partner sequences on average and the widest-used "
          f"one pairs with {heaviest['reuse max']:,}.")
    for written in (capacity_path, segment_path, reuse_path, overlap_path, matrix_path,
                    figure_path):
        print(f"Wrote {written}")
    print('Done.')


if __name__ == '__main__':
    main()
