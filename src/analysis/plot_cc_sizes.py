"""Operational 2D-CD CC-size barplot: positive pairs per bipartite connected component.

The 2D sibling of the 1D cluster-size barplot (`plot_clusters.py --plots barplot`).
Where that bars single-segment clusters by unique-sequence count for one slot, this
bars bipartite CCs (2D-CD atoms) by the number of positive pairs they carry, for the
slot PAIR. Reads `cc_pair_sizes.csv` emitted by `src/datasets/dataset_pairs_cc.py`
(one row per CC: cc_id, n_pairs), so the counts are exactly what the splitter routes
-- the operational universe, which for nt_cds keeps the codon variants the analysis
protein-dedup path (bigraph_pair_feasibility.py) collapses. Recomputes nothing;
delegates drawing to `src.utils.plot_utils.size_barplot`.

The mega-CC's dominance -- the tall first bar, and the largest-CC % in the title --
is the 2D-CD K-fold feasibility signal: whole CCs route to one fold, so a mega-CC
that towers over ~1/K of the pairs makes an 80/10/10 split infeasible without a cut
(docs/methods/glossary.md; the cut is `_megacc_cut.apply_drop_budget_cut`).

CLI:
    python -m src.analysis.plot_cc_sizes \\
        --run_dir data/datasets/flu/July_2025/runs/dataset_cc_nt_cds_ood_t097 \\
        --pair_label HA-NA --alphabet nt_cds --threshold_id t097 [--top_n 20]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.clustering_utils import threshold_decimal  # noqa: E402
from src.utils.plot_utils import size_barplot  # noqa: E402


def plot_cc_size_barplot(
    cc_pair_sizes_csv: Path,
    *,
    pair_label: str,
    alphabet: str,
    threshold_id: str,
    top_n: int,
    out_png: Path,
    ) -> dict:
    """Top-N CC-size barplot (positive pairs per CC) from a run's cc_pair_sizes.csv.

    Args:
        cc_pair_sizes_csv: `cc_pair_sizes.csv` (cc_id, n_pairs) from dataset_pairs_cc.
        pair_label: slot pair, e.g. 'HA-NA' -- title only.
        alphabet: 'aa' / 'nt_cds' / 'nt_ctg' -- title only.
        threshold_id: 'tXXX' -- title only.
        top_n: number of largest CCs to draw.
        out_png: output PNG path.

    Returns:
        {'n_ccs', 'n_pairs', 'largest_pct'}.
    """
    sizes = (pd.read_csv(cc_pair_sizes_csv)['n_pairs']
             .sort_values(ascending=False).reset_index(drop=True))
    n_ccs = int(len(sizes))
    n_pairs = int(sizes.sum())
    largest_pct = float(sizes.iloc[0]) / n_pairs * 100.0 if n_pairs else 0.0
    top = sizes.head(top_n)
    title = (
        f'{pair_label} — {alphabet} — {threshold_id} (id={threshold_decimal(threshold_id):.2f})\n'
        f'top {len(top)} of {n_ccs:,} CCs  ·  {n_pairs:,} positive pairs  ·  '
        f'largest CC {largest_pct:.1f}%')
    size_barplot(
        sizes, top_n=top_n, out_png=out_png, title=title,
        xlabel='connected component (rank-ordered, largest first)',
        ylabel='positive pairs in CC',
        xticklabels=[f'CC{i + 1}' for i in range(len(top))])
    return {'n_ccs': n_ccs, 'n_pairs': n_pairs, 'largest_pct': largest_pct}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--run_dir', type=Path, required=True,
                   help='dataset run dir containing cc_pair_sizes.csv')
    p.add_argument('--pair_label', required=True, help='slot pair, e.g. HA-NA (title only)')
    p.add_argument('--alphabet', required=True, help='aa / nt_cds / nt_ctg (title only)')
    p.add_argument('--threshold_id', required=True, help='tXXX (title only)')
    p.add_argument('--top_n', type=int, default=20, help='largest CCs to draw (default 20)')
    p.add_argument('--out_png', type=Path, default=None,
                   help='default: <run_dir>/figures/cc_pair_sizes_barplot.png')
    args = p.parse_args()

    csv = args.run_dir / 'cc_pair_sizes.csv'
    if not csv.exists():
        raise SystemExit(f"missing {csv} (rebuild the dataset to emit it)")
    out_png = args.out_png or (args.run_dir / 'figures' / 'cc_pair_sizes_barplot.png')
    stats = plot_cc_size_barplot(
        csv, pair_label=args.pair_label, alphabet=args.alphabet,
        threshold_id=args.threshold_id, top_n=args.top_n, out_png=out_png)
    print(f"  {args.pair_label} {args.alphabet} {args.threshold_id}: "
          f"{stats['n_ccs']:,} CCs, {stats['n_pairs']:,} pairs, "
          f"largest {stats['largest_pct']:.1f}% -> {out_png}")


if __name__ == '__main__':
    main()
