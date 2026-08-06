"""Per-CC single-side-cluster composition bars (one figure per slot) from build_cc_structure.

Reads `cc_cluster_composition.csv` + `cc_sizes.csv` from a `cc_{source}/{pair}/tXXX/` artifact
dir and draws, for each slot, the top-N CCs as stacked bars (top clusters + 'other'), so a
hub-dominated CC reads as one tall solid block and a diffuse CC as a short top block over a
large gray 'other'. Recomputes nothing; reuses `plot_utils.stacked_composition_barplot`.

The 2D sibling of the CC-size barplot (`plot_cc_sizes.py`): that shows how *big* each CC is,
this shows what each CC is *made of* in single-side clusters. Shares the title convention
(`plot_utils.barplot_title`) so the two read side by side.

CLI:
    python -m src.analysis.plot_cc_composition \\
        --cc_dir data/processed/flu/July_2025/cc_nt_cds_ood/HA-NA/t095 \\
        --pair HA-NA --alphabet nt_cds --threshold_id t095 [--top_n 12]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.plot_utils import barplot_title, stacked_composition_barplot  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cc_dir', type=Path, required=True,
                   help='cc_{source}/{pair}/tXXX/ dir holding cc_cluster_composition.csv + cc_sizes.csv')
    p.add_argument('--pair', required=True, help='slot pair, e.g. HA-NA (slot a = left, b = right)')
    p.add_argument('--alphabet', required=True, help='aa / nt_cds / nt_ctg (title only)')
    p.add_argument('--threshold_id', required=True, help='tXXX (title only)')
    p.add_argument('--top_n', type=int, default=12, help='number of largest CCs to draw (default 12)')
    p.add_argument('--note', default='', help='optional extra title line (e.g. "fragmented (1.6% dropped)")')
    p.add_argument('--normalize', action='store_true',
                   help='scale every bar to 100%% (share-of-CC) instead of real pair counts; default off '
                        '-- real heights match the CC-size barplot so the mega-CC towers and its colored '
                        'segments show whether it is one hub cluster (rigid) or spread (fragmentable)')
    args = p.parse_args()

    comp = pd.read_csv(args.cc_dir / 'cc_cluster_composition.csv', keep_default_na=False, na_values=[''])
    sizes = pd.read_csv(args.cc_dir / 'cc_sizes.csv', keep_default_na=False, na_values=[''])
    top_ccs = sizes.sort_values('n_pairs', ascending=False).head(args.top_n)['cc_id'].tolist()
    summ = json.loads((args.cc_dir / 'cc_summary.json').read_text())
    sa, sb = summ['slot_a'], summ['slot_b']  # robust to hyphenated short names (PB1-F2, PA-X)

    ylabel = 'share of CC' if args.normalize else 'positive pairs in CC'
    n_pairs = int(sizes['n_pairs'].sum())
    largest_pct = 100.0 * sizes['n_pairs'].max() / n_pairs if n_pairs else 0.0
    stats = (f'top {len(top_ccs)} of {len(sizes):,} CCs  ·  {n_pairs:,} positive pairs  ·  '
             f'largest CC {largest_pct:.1f}%')
    for slot, sname in (('a', sa), ('b', sb)):
        cs = comp[comp['slot'] == slot]
        out = (args.cc_dir / 'figures' /
               f'cc_cluster_composition_{sname}_{args.pair}_{args.alphabet}_{args.threshold_id}.png')

        title = barplot_title(
            args.pair, args.alphabet, args.threshold_id, stats,
            descriptor=f'{sname}-side cluster composition per CC', note=args.note
        )

        stacked_composition_barplot(
            cs, item_col='cc_id', category_col='cluster_id', value_col='n_pairs',
            item_order=top_ccs, item_labels=[f'CC{i + 1}' for i in range(len(top_ccs))],
            out_png=out, normalize=args.normalize, title=title, rotation=0,
            xlabel='connected component (sorted by size; not true CC ids)',
            ylabel=ylabel
        )

        print(f'  wrote {out}')


if __name__ == '__main__':
    main()
