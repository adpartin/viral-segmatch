"""Plot ONE importance measure along the CDS, from the saved importance table.

`plot_site_importance.py` computes gain, split count, SHAP and permutation, then draws gain and
SHAP on the same axes. That overlay is right when comparing the measures is the point. It is wrong
when a report has settled on one measure, because the reader is then invited to compare curves that
answer different questions.

This reads the table that script already wrote and draws a single measure. It loads no models and
recomputes nothing, so it is cheap to re-run while choosing a figure.

Outputs (to `--out_dir`, by default the directory holding the importance CSV):
    site_importance_{unit}_{measure}_trace.png   one panel per protein, importance along the CDS,
                                                 with the top sites labelled

CLI:
    python -m src.analysis.plot_site_importance_trace \\
        --importance_csv results/flu/July_2025/dataset_ha_na_h3n2_2024_random_cv4_pinned_length/site_importance/site_importance_codon.csv \\
        --unit codon --measure gain
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis._importance_helpers import (  # noqa: E402
    IMPORTANCE_MEASURES,
    load_importance,
    plot_importance_trace,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--importance_csv', type=Path, required=True,
                        help='site_importance_{unit}.csv from plot_site_importance.py')
    parser.add_argument('--unit', default='codon', choices=['nt', 'codon', 'aa'],
                        help='site unit the table was built on; used for the axis label')
    parser.add_argument('--measure', default='gain', choices=list(IMPORTANCE_MEASURES),
                        help='which importance measure to draw')
    parser.add_argument('--n_label', type=int, default=5,
                        help='how many top sites to annotate per protein')
    parser.add_argument('--out_dir', type=Path, default=None)
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()

    importance = load_importance(args.importance_csv)
    out_dir = args.out_dir or args.importance_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'site_importance_{args.unit}_{args.measure}_trace.png'

    # The run is named by the dataset directory the table sits under, so the figure carries
    # the same identification as the other site_importance outputs beside it.
    run_name = args.importance_csv.resolve().parent.parent.name
    n_folds = int(importance['folds_used'].max()) if 'folds_used' in importance else 0
    caption = f"{run_name}  |  unit={args.unit}, measure={args.measure}"
    if n_folds:
        caption += f", mean of {n_folds} folds"
    written = plot_importance_trace(importance, args.measure, args.unit, out_path,
                                    dpi=args.dpi, n_label=args.n_label, title=caption)

    share = f'{args.measure}_frac'
    top = importance.nlargest(10, share)[['protein', 'site', share]]
    print(f"Top 10 sites by {args.measure}:")
    print(top.to_string(index=False))
    print(f"\nWrote {written}")
    print('Done.')


if __name__ == '__main__':
    main()
