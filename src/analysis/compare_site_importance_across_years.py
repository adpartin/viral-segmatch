"""Compare per-site importance between two populations of the same schema pair.

Step 1 of docs/plans/2026-09-14_cross_year_importance_all_pairs_alignment_plan.md asks whether
models fitted on separate years lean on the same codon sites. Two importance maps are comparable
only when both proteins keep the same pinned CDS length in both populations, because a site index
is a position in that fixed coordinate system. The script checks that before comparing anything.

How many of each year's top N sites the other year also ranks that high. Spearman over all sites
is not reported, because most sites carry zero gain in both years, tie at the same rank, and agree
for free.

The count alone says nothing without a draw to compare it against. A site can only enter a top-N
list if it varies, so the eligible set is the sites with more than one observed value, taken from
the sequences rather than from the fits. Drawing N sites independently from each year's eligible
set gives

    expected = |V_left AND V_right| * (N / |V_left|) * (N / |V_right|)

because a site has to be eligible in both years to appear in both lists. This is a descriptive
baseline, not a significance test: it treats eligible sites as equally likely to be picked, which
correlated sites violate.

A low overlap does not by itself show that the sequence signal changed. Correlated sites stand in
for each other, so two fits can reach the same accuracy from different members of one group.

Outputs (to `--out_dir`):
    site_importance_comparison.csv   one row per protein and top-N cut, with the three measures

CLI:
    python -m src.analysis.compare_site_importance_across_years \
        --left  results/.../dataset_ha_na_human_h3n2_2024_..._n1698_seed42/site_importance/site_importance_codon.csv \
        --right results/.../dataset_ha_na_human_h3n2_2025_..._n1337_seed42/site_importance/site_importance_codon.csv \
        --left_label 2024 --right_label 2025 --out_dir results/.../cross_year
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis._importance_helpers import (  # noqa: E402
    IMPORTANCE_MEASURES,
    load_importance,
    share_column,
)

COMPARISON_COLUMNS = ['protein', 'left', 'right', 'sites', 'top_n', 'shared', 'shared_frac',
                      'eligible_left', 'eligible_right', 'eligible_both', 'expected',
                      'enrichment', 'left_share', 'right_share']

# The combined row ranks both proteins together, so its top N is the N best of every site rather
# than N of each protein. It is reported beside them because the two answer different questions.
COMBINED = 'combined'


def join_on_site(left: pd.DataFrame, right: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Join two importance tables on (protein, site), keeping one share column from each.

    Args:
      left: importance table for the first population.
      right: importance table for the second.
      measure: one of `IMPORTANCE_MEASURES`.

    Returns:
      One row per (protein, site) with `share_left` and `share_right`.

    Raises:
      ValueError: the two tables do not cover the same sites, so the coordinates differ.
    """
    share = share_column(measure)
    keep = ['protein', 'site', share, 'n_values']
    joined = left[keep].merge(right[keep], on=['protein', 'site'],
                              suffixes=('_left', '_right'), how='outer', indicator=True)
    unmatched = joined[joined['_merge'] != 'both']
    if not unmatched.empty:
        counts = unmatched['protein'].value_counts().to_dict()
        raise ValueError(
            f"the two tables do not cover the same sites, so their coordinates differ: "
            f"{len(unmatched)} unmatched rows across {counts}. Both populations must use the "
            f"same pinned CDS length for every protein.")
    return joined.drop(columns='_merge').rename(
        columns={f'{share}_left': 'share_left', f'{share}_right': 'share_right'})


def compare_protein(sites: pd.DataFrame, top_n: int) -> dict:
    """Shared top-`top_n` sites, and how many a pair of independent draws would share.

    Args:
      sites: one protein's rows from `join_on_site`, or every row for the combined ranking.
      top_n: how many highest-importance sites each year contributes.

    Returns:
      The shared count and fraction, the eligible-set sizes, the expected shared count under
      independent draws, the enrichment over it, and each year's share of gain.
    """
    left_top = set(map(tuple, sites.nlargest(top_n, 'share_left')[['protein', 'site']].values))
    right_top = set(map(tuple, sites.nlargest(top_n, 'share_right')[['protein', 'site']].values))
    shared = len(left_top & right_top)

    # A site can only be picked if it varies, which the sequences fix before any model is fitted.
    varies_left = sites['n_values_left'] > 1
    varies_right = sites['n_values_right'] > 1
    n_left, n_right = int(varies_left.sum()), int(varies_right.sum())
    n_both = int((varies_left & varies_right).sum())
    expected = n_both * (top_n / n_left) * (top_n / n_right) if n_left and n_right else float('nan')

    return {
        'sites': len(sites),
        'top_n': top_n,
        'shared': shared,
        'shared_frac': shared / top_n,
        'eligible_left': n_left,
        'eligible_right': n_right,
        'eligible_both': n_both,
        'expected': expected,
        'enrichment': shared / expected if expected else float('nan'),
        'left_share': sites['share_left'].sum(),
        'right_share': sites['share_right'].sum(),
    }


def compare_importance(left: pd.DataFrame, right: pd.DataFrame, measure: str,
                       top_ns: list, left_label: str, right_label: str) -> pd.DataFrame:
    """Compare two importance tables, one row per protein and top-N cut.

    A `combined` row accompanies the per-protein rows. It ranks every site together, so its top N
    is the N best overall rather than N of each protein, and a shift in which protein carries the
    gain costs it shared sites on its own.

    Args:
      left: importance table for the first population.
      right: importance table for the second.
      measure: one of `IMPORTANCE_MEASURES`.
      top_ns: the top-N cuts to report.
      left_label: name for the first population, e.g. a year.
      right_label: name for the second.

    Returns:
      The comparison table, with the columns in `COMPARISON_COLUMNS`.
    """
    joined = join_on_site(left, right, measure)
    groups = [(COMBINED, joined)] + list(joined.groupby('protein'))
    rows = []
    for protein, sites in groups:
        for top_n in top_ns:
            row = {'protein': protein, 'left': left_label, 'right': right_label}
            row.update(compare_protein(sites, top_n))
            rows.append(row)
    table = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    return table.sort_values(['protein', 'top_n']).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--left', type=Path, required=True,
                   help='site_importance_{unit}.csv for the first population')
    p.add_argument('--right', type=Path, required=True,
                   help='site_importance_{unit}.csv for the second population')
    p.add_argument('--left_label', default='left', help='name for the first population')
    p.add_argument('--right_label', default='right', help='name for the second')
    p.add_argument('--measure', default='gain', choices=list(IMPORTANCE_MEASURES))
    p.add_argument('--top_n', type=int, nargs='+', default=list(range(10, 51, 5)),
                   help='top-N cuts to report')
    p.add_argument('--out_dir', type=Path, required=True)
    args = p.parse_args()

    table = compare_importance(load_importance(args.left), load_importance(args.right),
                               args.measure, args.top_n, args.left_label, args.right_label)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / 'site_importance_comparison.csv'
    table.to_csv(out_csv, index=False)

    shown = table[['protein', 'top_n', 'shared', 'shared_frac', 'expected', 'enrichment']].copy()
    shown['shared_frac'] = shown['shared_frac'].map(lambda v: f'{v:.0%}')
    shown['expected'] = shown['expected'].map(lambda v: f'{v:.2f}')
    shown['enrichment'] = shown['enrichment'].map(lambda v: f'{v:.1f}x')
    print(f"\n{args.measure} importance, {args.left_label} against {args.right_label}:")
    print(shown.to_string(index=False))
    print(f"\nWrote {out_csv}")
    print('Done.')


if __name__ == '__main__':
    main()
