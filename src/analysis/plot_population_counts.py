"""Count the isolates of one population by year, H/N subtype and host group, and plot them.

The corpus isolates are the `assembly_id` values of a Stage 1 table. Their host group, subtype and
year come from `metadata_enrichment.attach_isolate_metadata`, and the population is kept with
`metadata_enrichment.filter_by_metadata`, the filter `align_proteins` uses. Every count is of
isolates, before any per-protein CDS completeness filter and before sequences are deduplicated.

Outputs under <out_dir>/<population>/:
    isolate_counts.csv                isolates per (host, hn_subtype, year); every figure is drawn
                                      from this table
    isolates_per_year.png             one bar per year, one color
    isolates_per_year_by_subtype.png  one bar per year, stacked by H/N subtype
    isolates_per_year_by_host.png     one bar per year, stacked by host group
    host_subtype_heatmap.png          isolates per host group and H/N subtype, log color scale
    population_counts_manifest.json   filters, inputs, counts and code state

Subtype and host colors are fixed in `src/utils/plot_config.py`, so a subtype or host has the
same color in every figure.

Example:
    python -m src.analysis.plot_population_counts \\
        --host Human Swine Chicken Duck --hn_subtype H3N2 H1N1 H5N1 --year_range 2010 2025
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import StrMethodFormatter

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.experiment_utils import file_sha256, get_git_info  # noqa: E402
from src.utils.metadata_enrichment import (  # noqa: E402
    attach_isolate_metadata,
    filter_by_metadata,
    population_label,
)
from src.utils.plot_config import HOST_COLORS, SUBTYPE_COLORS  # noqa: E402
from src.utils.plot_utils import annotated_heatmap, savefig, setup_plot_style  # noqa: E402

OUTPUT_NAMES = (
    'isolate_counts.csv', 'isolates_per_year.png', 'isolates_per_year_by_subtype.png',
    'isolates_per_year_by_host.png', 'host_subtype_heatmap.png',
    'population_counts_manifest.json',
)
# The one-color year figure uses a color that no subtype or host has.
YEAR_BAR_COLOR = '#4682b4'  # steel blue
METADATA_COLUMNS = ['host', 'hn_subtype', 'year']


def select_population(corpus: pd.DataFrame, metadata_file: Path, filters: dict) -> pd.DataFrame:
    """Return one row per isolate in the population, with its host group, subtype and year.

    Args:
        corpus: One row per corpus isolate, with `assembly_id`.
        metadata_file: Parsed isolate metadata read by `attach_isolate_metadata`.
        filters: Keyword arguments for `filter_by_metadata` (host, hn_subtype, year, year_range).

    Returns:
        The population, with `assembly_id`, `host`, `hn_subtype` and `year`.

    Raises:
        ValueError: The filter keeps no isolate, an isolate appears twice, or an isolate lacks
            host, subtype or year.
    """
    with_metadata = attach_isolate_metadata(corpus, metadata_file=metadata_file)
    population = filter_by_metadata(with_metadata, **filters)
    population = population[['assembly_id'] + METADATA_COLUMNS]

    if population.empty:
        raise ValueError('No isolate matches the requested population.')
    if not population['assembly_id'].is_unique:
        raise ValueError('An isolate appears more than once after the metadata join.')
    # groupby would silently drop an isolate with a missing value, so the counts would no
    # longer add up to the population.
    missing = population[METADATA_COLUMNS].isna().any(axis=1)
    if missing.any():
        raise ValueError(f'{missing.sum():,} isolates lack host, subtype or year; filter on all '
                         'three so every counted isolate has a value.')
    return population


def count_isolates(population: pd.DataFrame) -> pd.DataFrame:
    """Count isolates per host group, subtype and year.

    Args:
        population: One row per isolate, with `host`, `hn_subtype` and `year`.

    Returns:
        One row per (host, hn_subtype, year) present, with its `isolates` count.

    Raises:
        ValueError: The counts do not add up to the population size.
    """
    counts = population.groupby(METADATA_COLUMNS).size().reset_index(name='isolates')
    counts['year'] = counts['year'].astype(int)
    if counts['isolates'].sum() != len(population):
        raise ValueError('Isolate counts do not add up to the population size.')
    return counts


def ordered_categories(present, colors: dict) -> list:
    """List the categories present in the order of their color map.

    Args:
        present: Category values found in the data.
        colors: A color map from `plot_config`, whose key order is the display order.

    Returns:
        The present categories, in color-map order.

    Raises:
        ValueError: A present category has no color.
    """
    missing = sorted(set(present) - set(colors))
    if missing:
        raise ValueError(f'No color for {missing}; add them to src/utils/plot_config.py.')
    ordered = [category for category in colors if category in set(present)]
    return ordered


def plot_isolates_per_year(counts: pd.DataFrame, out_png: Path, title: str,
                           stack_by: str | None = None, colors: dict | None = None,
                           legend_title: str = '') -> None:
    """Draw one bar per year with its total above it, in one color or stacked by a category.

    Every year from the first to the last in `counts` gets a bar, so a year with no isolates
    shows as an empty slot rather than disappearing.

    Args:
        counts: Output of `count_isolates`.
        out_png: Where to save the figure.
        title: Figure title.
        stack_by: `hn_subtype` or `host` to stack the bars by that column; None for one color.
        colors: The color map for `stack_by`; required when `stack_by` is set.
        legend_title: Title of the legend when the bars are stacked.

    Returns:
        None.
    """
    years = list(range(counts['year'].min(), counts['year'].max() + 1))
    fig, ax = plt.subplots(figsize=(12, 5))

    if stack_by is None:
        totals = counts.groupby('year')['isolates'].sum().reindex(years, fill_value=0)
        ax.bar(years, totals, color=YEAR_BAR_COLOR, edgecolor='white', linewidth=0.5)
    else:
        per_year = counts.pivot_table(index='year', columns=stack_by, values='isolates',
                                      aggfunc='sum', fill_value=0)
        per_year = per_year.reindex(years, fill_value=0)
        # Each category's bars start where the categories below it end.
        bottom = np.zeros(len(years))
        for category in ordered_categories(per_year.columns, colors):
            heights = per_year[category].to_numpy()
            ax.bar(years, heights, bottom=bottom, color=colors[category], label=category,
                   edgecolor='white', linewidth=0.5)
            bottom = bottom + heights
        ax.legend(title=legend_title, loc='upper left')
        totals = per_year.sum(axis=1)

    for year, total in zip(years, totals):
        ax.text(year, total, f'{total:,}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(years)
    ax.set_xticklabels(years)
    ax.set_xlabel('Year')
    ax.set_ylabel('Isolates')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.set_title(title)
    ax.grid(False, axis='x')
    ax.set_axisbelow(True)
    savefig(out_png, dpi=200)


def plot_host_subtype_heatmap(counts: pd.DataFrame, out_png: Path) -> None:
    """Draw isolates per host group and subtype on a log color scale, in cells wider than tall.

    The log scale keeps small groups visible next to the largest one. A cell with no isolates
    cannot be placed on it, so it is drawn gray and labelled 0.

    Args:
        counts: Output of `count_isolates`.
        out_png: Where to save the figure.

    Returns:
        None.
    """
    hosts = ordered_categories(counts['host'].unique(), HOST_COLORS)
    subtypes = ordered_categories(counts['hn_subtype'].unique(), SUBTYPE_COLORS)
    table = counts.pivot_table(index='host', columns='hn_subtype', values='isolates',
                               aggfunc='sum', fill_value=0)
    table = table.reindex(index=hosts, columns=subtypes, fill_value=0)
    cell_height = 0.6  # cell height as a fraction of cell width

    # The default style turns the axes grid on, which would draw lines through the cells.
    with plt.rc_context({'axes.grid': False, 'xtick.labelsize': 14, 'axes.labelsize': 15}):
        annotated_heatmap(
            table.to_numpy(),
            row_labels=hosts,
            col_labels=subtypes,
            out_png=out_png,
            title='Host x H/N Subtype Heatmap',
            xlabel='H/N subtype',
            ylabel='Host',
            cbar_label='Isolates (log scale)',
            cmap='Blues',
            norm=LogNorm(vmin=1, vmax=table.to_numpy().max()),
            value_fmt='{:,.0f}',
            aspect=cell_height,
            row_fontsize=14,
            cell_fontsize=16,
            title_fontsize=16,
            figsize=(1.6 * len(subtypes) + 3, 1.6 * cell_height * len(hosts) + 1.5),
            dpi=200,
        )


def main() -> None:
    """Count one population's isolates and write the table, figures and manifest.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--host', nargs='+', default=None, help='host groups, e.g. Human Swine')
    parser.add_argument('--hn_subtype', nargs='+', default=None, help='e.g. H3N2 H1N1')
    years = parser.add_mutually_exclusive_group()
    years.add_argument('--year', nargs='+', type=int, default=None)
    years.add_argument('--year_range', nargs=2, type=int, default=None, metavar=('MIN', 'MAX'))
    parser.add_argument('--corpus_file', type=Path,
                        default=PROJ / 'data/processed/flu/July_2025/protein_final.parquet')
    parser.add_argument('--metadata_file', type=Path,
                        default=PROJ / 'data/processed/flu/metadata_eda/flu_genomes_metadata_parsed.csv')
    parser.add_argument('--out_dir', type=Path,
                        default=PROJ / 'results/flu/July_2025/population_counts')
    parser.add_argument('--overwrite', action='store_true',
                        help='replace existing outputs for this population')
    args = parser.parse_args()

    filters = {'host': args.host, 'hn_subtype': args.hn_subtype,
               'year': args.year, 'year_range': args.year_range}
    label = population_label(**filters)
    out_dir = args.out_dir / label
    existing = [name for name in OUTPUT_NAMES if (out_dir / name).exists()]
    if existing and not args.overwrite:
        parser.error(f'Outputs already exist in {out_dir}: {existing}; pass --overwrite.')

    corpus = pd.read_parquet(args.corpus_file, columns=['assembly_id']).drop_duplicates()
    population = select_population(corpus, args.metadata_file, filters)
    counts = count_isolates(population)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out_dir / 'isolate_counts.csv', index=False)
    setup_plot_style(use_seaborn_palette=False)
    plot_isolates_per_year(counts, out_dir / 'isolates_per_year.png',
                           title='Isolates per year')
    plot_isolates_per_year(counts, out_dir / 'isolates_per_year_by_subtype.png',
                           title='Isolates per year by H/N subtype',
                           stack_by='hn_subtype', colors=SUBTYPE_COLORS,
                           legend_title='H/N subtype')
    plot_isolates_per_year(counts, out_dir / 'isolates_per_year_by_host.png',
                           title='Isolates per year by host',
                           stack_by='host', colors=HOST_COLORS, legend_title='Host')
    plot_host_subtype_heatmap(counts, out_dir / 'host_subtype_heatmap.png')

    manifest = {
        'stage': 'population_counts',
        'population': label,
        'filters': filters,
        'inputs': {
            'corpus_file': str(args.corpus_file.resolve()),
            'corpus_file_sha256': file_sha256(args.corpus_file),
            'metadata_file': str(args.metadata_file.resolve()),
            'metadata_file_sha256': file_sha256(args.metadata_file),
        },
        'counts': {
            'corpus_isolates': len(corpus),
            'population_isolates': len(population),
            'first_year': int(counts['year'].min()),
            'last_year': int(counts['year'].max()),
        },
        'code': get_git_info(),
        'outputs': list(OUTPUT_NAMES),
    }
    manifest_path = out_dir / 'population_counts_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{label}: {len(population):,} of {len(corpus):,} corpus isolates.')
    print(f'Wrote outputs to {out_dir}')
    print('Done.')


if __name__ == '__main__':
    main()
