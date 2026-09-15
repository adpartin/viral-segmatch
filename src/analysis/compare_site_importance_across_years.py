"""Compare per-site importance between two populations of the same schema pair.

Step 1 of docs/plans/2026-09-14_cross_year_importance_all_pairs_alignment_plan.md asks whether
models fitted on separate years lean on the same codon sites. Two importance maps are comparable
only when both proteins keep the same pinned CDS length in both populations, because a site index
is a position in that fixed coordinate system. The script checks that before comparing anything.

Three measures, because they answer different questions. Spearman over all sites asks whether the
whole ranking moved. Overlap and Jaccard over the top N ask whether the sites a reader would quote
moved. The share of gain per protein asks whether one side of the pair took over.

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

COMPARISON_COLUMNS = ['protein', 'left', 'right', 'sites', 'spearman', 'top_n',
                      'overlap', 'jaccard', 'left_share', 'right_share']


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
    keep = ['protein', 'site', share]
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
    """Spearman over all sites, plus overlap and Jaccard over the top `top_n` of each side.

    Args:
      sites: one protein's rows from `join_on_site`.
      top_n: how many highest-importance sites each side contributes.

    Returns:
      `spearman`, `overlap`, `jaccard`, `left_share`, `right_share` and the site count.
    """
    left_top = set(sites.nlargest(top_n, 'share_left')['site'])
    right_top = set(sites.nlargest(top_n, 'share_right')['site'])
    shared = left_top & right_top
    union = left_top | right_top
    return {
        'sites': len(sites),
        'spearman': sites['share_left'].corr(sites['share_right'], method='spearman'),
        'top_n': top_n,
        'overlap': len(shared),
        'jaccard': len(shared) / len(union) if union else float('nan'),
        'left_share': sites['share_left'].sum(),
        'right_share': sites['share_right'].sum(),
    }


def compare_importance(left: pd.DataFrame, right: pd.DataFrame, measure: str,
                       top_ns: list, left_label: str, right_label: str) -> pd.DataFrame:
    """Compare two importance tables, one row per protein and top-N cut.

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
    rows = []
    for protein, sites in joined.groupby('protein'):
        for top_n in top_ns:
            row = {'protein': protein, 'left': left_label, 'right': right_label}
            row.update(compare_protein(sites, top_n))
            rows.append(row)
    table = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    # Shares are read per protein, so they sum to 1 across the pair for each population.
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
    p.add_argument('--top_n', type=int, nargs='+', default=[12, 25],
                   help='top-N cuts for overlap and Jaccard')
    p.add_argument('--out_dir', type=Path, required=True)
    args = p.parse_args()

    table = compare_importance(load_importance(args.left), load_importance(args.right),
                               args.measure, args.top_n, args.left_label, args.right_label)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / 'site_importance_comparison.csv'
    table.to_csv(out_csv, index=False)

    shown = table.copy()
    for column in ('spearman', 'jaccard', 'left_share', 'right_share'):
        shown[column] = shown[column].map(lambda v: f'{v:.3f}')
    print(f"\n{args.measure} importance, {args.left_label} against {args.right_label}:")
    print(shown.to_string(index=False))
    print(f"\nWrote {out_csv}")
    print('Done.')


if __name__ == '__main__':
    main()
