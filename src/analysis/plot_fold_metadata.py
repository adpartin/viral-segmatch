"""Isolate-metadata distribution across the three splits of one CV fold.

ONE figure per invocation: `--field` picks the metadata axis, `--style` how it is drawn.
`bars` stacks train, val and test as three horizontal bar charts and suits a categorical axis;
`panels` stacks them as three histograms over a shared x-axis, which is what makes a SHIFT
between the splits legible (a test split concentrated in later years sits visibly apart) and
needs an ordered axis. Comparing two folds (or the 2D-CD arm against its random arm) is two runs
of the same command with a different `--fold_dir`.

Counts are over the ISOLATES carrying that split's positive pairs, not over rows: a pair's two
slots come from one isolate (`assembly_id_a == assembly_id_b` for every positive), and negatives
recombine two isolates that the positives already contribute, so rows would count the same
isolates repeatedly and with a weight set by the negative sampler.

Reuses `visualize_dataset_stats.plot_distribution_by_split` (the three-subplot horizontal barplot)
and `_pair_helpers.get_metadata_distributions` (per-isolate value counts); this module only
assembles the per-split isolate sets a Stage-3 fold directory implies.

CLI:
    python -m src.analysis.plot_fold_metadata \\
        --fold_dir data/datasets/flu/July_2025/runs/dataset_cc_nt_cds_cm0_h3n2_t099/fold_0 \\
        --field year
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.visualize_dataset_stats import plot_distribution_by_split  # noqa: E402
from src.datasets._pair_helpers import get_metadata_distributions  # noqa: E402
from src.utils.metadata_enrichment import attach_isolate_metadata  # noqa: E402
from src.utils.plot_utils import histogram_panels  # noqa: E402

_SPLITS = ('train', 'val', 'test')
_FIELDS = ('host', 'year', 'hn_subtype', 'geo_location_clean', 'passage')
_NUMERIC_FIELDS = ('year',)   # the only axis with an order to overlay along

# Same colors the fold UMAPs give the splits, so the two figure families read together.
_SPLIT_COLORS = {'train': '#4c72b0', 'val': '#dd8452', 'test': '#d43d51'}


def fold_isolates_by_split(fold_dir: Path) -> dict:
    """The isolate ids behind each split's positive pairs.

    Args:
        fold_dir: a `fold_k` directory holding `{train,val,test}_pairs.csv`.

    Returns:
        {split_name: set of assembly_id}.

    Raises:
        SystemExit: if a split CSV is missing.
    """
    out = {}
    for name in _SPLITS:
        path = fold_dir / f'{name}_pairs.csv'
        if not path.exists():
            raise SystemExit(f'ERROR: no {name} split at {path}.')
        df = pd.read_csv(path, usecols=['label', 'assembly_id_a'], dtype=str,
                         keep_default_na=False, na_values=[], low_memory=False)
        out[name] = set(df.loc[df['label'] == '1', 'assembly_id_a'])
    return out


def main() -> None:
    """Parse the CLI, build the per-split metadata distributions, and write the PNG."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--fold_dir', type=Path, required=True, help='a fold_k dir of a Stage-3 dataset.')
    p.add_argument('--field', choices=_FIELDS, default='year', help='metadata axis (default year).')
    p.add_argument('--style', choices=('bars', 'panels'), default='bars',
                   help="'bars' = one horizontal barplot per split; 'panels' = one histogram per "
                        f'split over a shared x-axis, for {"/".join(_NUMERIC_FIELDS)} only.')
    p.add_argument('--top_n', type=int, default=20,
                   help='bars: values to show per split (default 20); panels show every value.')
    p.add_argument('--xmin', type=float, default=None, help='panels: x-axis lower bound.')
    p.add_argument('--xmax', type=float, default=None, help='panels: x-axis upper bound.')
    p.add_argument('--xtick_step', type=float, default=None,
                   help='panels: x tick spacing (e.g. 5 for a year axis).')
    p.add_argument('--out_png', type=Path, default=None, help='default: <fold_dir>/figures/<name>.png')
    p.add_argument('--title', default=None, help='default: derived from the dataset, fold and field.')
    args = p.parse_args()

    if args.style == 'panels' and args.field not in _NUMERIC_FIELDS:
        raise SystemExit(f'--style panels needs an ordered axis ({"/".join(_NUMERIC_FIELDS)}); '
                         f'{args.field} is categorical, so use --style bars.')

    isolates = fold_isolates_by_split(args.fold_dir)
    print('isolates (pos only): ' + ' '.join(f'{n}={len(s):,}' for n, s in isolates.items()))

    # One metadata row per isolate in the fold, so the counts are per isolate, not per row.
    all_ids = sorted(set().union(*isolates.values()))
    meta = attach_isolate_metadata(pd.DataFrame({'assembly_id': all_ids}), project_root=PROJ)

    out_png = args.out_png or (args.fold_dir / 'figures'
                               / f'metadata_{args.field}_by_split_{args.style}.png')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    title = args.title or (f'{args.fold_dir.parent.name} · {args.fold_dir.name}\n'
                           f'{args.field} across splits, pos only')

    if args.style == 'panels':
        by_split = {n: pd.to_numeric(meta.loc[meta['assembly_id'].isin(s), args.field],
                                     errors='coerce').dropna().to_numpy()
                    for n, s in isolates.items()}
        stats = histogram_panels(by_split, out_png=out_png, title=title, xlabel=args.field,
                                 xlim=(args.xmin, args.xmax), xtick_step=args.xtick_step,
                                 label_colors=_SPLIT_COLORS)
        print(f"wrote {out_png} ({stats['bins']} bins)")
        return

    total = sum(len(s) for s in isolates.values())
    distributions = {n: get_metadata_distributions(meta, s) for n, s in isolates.items()}
    split_sizes = {n: {'isolates': len(s), 'isolate_share': len(s) / total if total else 0.0}
                   for n, s in isolates.items()}
    plot_distribution_by_split(distributions, args.field, title, 'isolates', out_png,
                               top_n=args.top_n, split_sizes=split_sizes)


if __name__ == '__main__':
    main()
