"""Single (regime x threshold) metric heatmap from Stage-4 post-hoc level1 CSVs.

Pools a per-regime rate metric (default TNR = negative-rejection rate) across CV folds for one
run family -- runs named `{run_prefix}_{tXXX}_fold{f}` under `--runs_root`, each carrying
`post_hoc/level1_neg_regimes.csv` -- into a `(regime x threshold)` grid, then draws it with the
shared `plot_utils.annotated_heatmap`. The `--title` is caller-supplied so the figure states
exactly what it represents (which slot / clustering / recipe).

Pooling: a rate metric is aggregated over folds as `sum(metric_f * weight_f) / sum(weight_f)`
(default weight `n_neg`) -- the full-dataset rate, since GroupKFold test sets are disjoint. For
TNR this equals `sum(tn) / sum(tn + fp)`.

Generic beyond TNR: `--metric` / `--weight_col` / `--rows` / `--csv_rel` / `--row_col` let the
same tool draw, e.g., accuracy by regime, or a level2 metric by subtype (`--csv_rel
post_hoc/level2_by_subtype.csv --row_col stratum --rows H1N1 H3N2 ...`).

CLI:
    python -m src.analysis.regime_metric_heatmap \\
        --run_prefix lgbm_1dcd_cm0_slota \\
        --thresholds t099 t098 t097 t096 t095 \\
        --title "1D-CD slot a (HA held out): negative TNR by regime x t  (cm0/nt_cds, k=4)" \\
        --out_png tmp/regime_tnr_slota.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.plot_utils import annotated_heatmap  # noqa: E402

# The 8 negative regimes in easy -> hard order (by number of matching metadata axes); the
# host_subtype_year row (all three match) is the demographic-shortcut worst case.
_REGIMES = ['none_match', 'host_only', 'subtype_only', 'year_only',
            'host_subtype_only', 'host_year_only', 'subtype_year_only', 'host_subtype_year']


def pooled_matrix(runs_root: Path, run_prefix: str, thresholds, n_folds: int, csv_rel: str,
                  row_col: str, metric: str, weight_col: str, rows) -> np.ndarray:
    """`(len(rows) x len(thresholds))` fold-pooled `metric`, weighted by `weight_col`.

    Cells with no data (missing runs, or zero weight for that row/threshold) stay NaN.
    """
    ridx = {r: i for i, r in enumerate(rows)}
    num = np.zeros((len(rows), len(thresholds)))
    den = np.zeros((len(rows), len(thresholds)))
    for j, t in enumerate(thresholds):
        for f in range(n_folds):
            p = Path(runs_root) / f'{run_prefix}_{t}_fold{f}' / csv_rel
            if not p.exists():
                continue
            d = pd.read_csv(p, keep_default_na=False, na_values=[''])
            for rec in d.to_dict('records'):
                r = rec.get(row_col)
                if r not in ridx:
                    continue
                w = float(rec[weight_col]) if str(rec[weight_col]) != '' else 0.0
                mv = float(rec[metric]) if str(rec[metric]) != '' else np.nan
                if w <= 0 or np.isnan(mv):
                    continue
                num[ridx[r], j] += mv * w
                den[ridx[r], j] += w
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--runs_root', default=str(PROJ / 'models/flu/July_2025/runs'))
    ap.add_argument('--run_prefix', required=True,
                    help='Runs are {run_prefix}_{tXXX}_fold{f} (e.g. lgbm_1dcd_cm0_slota).')
    ap.add_argument('--thresholds', nargs='+', required=True, help='tXXX column order, strict-first.')
    ap.add_argument('--n_folds', type=int, default=4)
    ap.add_argument('--csv_rel', default='post_hoc/level1_neg_regimes.csv',
                    help='Per-run CSV to read (default the level1 neg-regime table).')
    ap.add_argument('--row_col', default='regime', help='CSV column that names the heatmap rows.')
    ap.add_argument('--rows', nargs='+', default=_REGIMES, help='Row values + order (default: 8 neg regimes).')
    ap.add_argument('--metric', default='tnr', help='CSV column to pool (default tnr).')
    ap.add_argument('--weight_col', default='n_neg', help='Fold-pooling weight column (default n_neg).')
    ap.add_argument('--title', required=True, help='Figure title -- state exactly what it represents.')
    ap.add_argument('--out_png', required=True, type=Path)
    ap.add_argument('--cbar_label', default='TNR (correctly-rejected negatives)')
    ap.add_argument('--xlabel', default='mmseqs identity threshold  t')
    ap.add_argument('--cmap', default='RdYlGn')
    ap.add_argument('--vmin', type=float, default=0.0)
    ap.add_argument('--vmax', type=float, default=1.0)
    args = ap.parse_args()

    mat = pooled_matrix(Path(args.runs_root), args.run_prefix, args.thresholds, args.n_folds,
                        args.csv_rel, args.row_col, args.metric, args.weight_col, args.rows)
    # tXXX -> 0.XX tick labels when they look like thresholds; else pass through.
    col_labels = [f'{int(t[1:]) / 100:.2f}' if t.startswith('t') and t[1:].isdigit() else t
                  for t in args.thresholds]
    annotated_heatmap(mat, row_labels=list(args.rows), col_labels=col_labels, out_png=args.out_png,
                      title=args.title, xlabel=args.xlabel, cbar_label=args.cbar_label,
                      cmap=args.cmap, vmin=args.vmin, vmax=args.vmax)
    n_ok = int(np.sum(~np.isnan(mat)))
    print(f'wrote {args.out_png}  ({mat.shape[0]}x{mat.shape[1]} grid, {n_ok} cells with data)')


if __name__ == '__main__':
    main()
