"""Plot the aa-vs-nt cluster_disjoint LGBM comparison (Experiment B-nt).

For each schema pair (HA/NA, PB2/PB1) and each routing configuration
{seq_disjoint, aa-cluster_id099, nt-cluster_id100, nt-cluster_id099},
this script reads `post_hoc/metrics.csv` from the corresponding LGBM
baseline run and emits a grouped-bar PNG + a flat CSV summary.

The reference deliverable promised by step 9 of
`docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` § B-nt
("threshold sweep plot the aa side couldn't reach") is implemented here
as a 4-routing comparison rather than a true threshold sweep — the
bipartite-component collapse on the full Flu A corpus rules out feasible
nt cluster_disjoint below id099 (see
`docs/results/2026-05-15_cluster_disjoint_feasibility_nt_*.csv`).

CLI:
    python -m src.analysis.plot_aa_vs_nt_cluster_disjoint \
        --runs_root models/flu/July_2025/runs \
        --out_dir   results/flu/July_2025/runs/cluster_aa_vs_nt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))


# Routing label → bundle name pattern (used to autodiscover the LGBM run).
# The pattern is matched against baseline_lgbm_<bundle>_<TS> directories;
# the latest matching timestamp wins.
_ROUTINGS = [
    ('seq_disjoint',      'flu_{pair}'),
    ('aa cluster_id099',  'flu_{pair}_cluster_id99'),
    ('nt cluster_id100',  'flu_{pair}_cluster_nt_id100'),
    ('nt cluster_id099',  'flu_{pair}_cluster_nt_id099'),
]
_PAIRS = [
    ('HA/NA',   'ha_na'),
    ('PB2/PB1', 'pb2_pb1'),
]
_METRIC_DISPLAY = [
    ('f1_score',      'F1'),
    ('avg_precision', 'AUC-PR'),
    ('auc_roc',       'AUC-ROC'),
    ('mcc',           'MCC'),
]


def _latest_lgbm_dir(runs_root: Path, bundle: str) -> Optional[Path]:
    """Return the latest `baseline_lgbm_<bundle>_<YYYYMMDD_HHMMSS>` dir.

    Bundle match is anchored — `bundle='flu_ha_na'` does NOT match
    `flu_ha_na_cluster_id99`, etc.
    """
    pattern = re.compile(
        rf'^baseline_lgbm_{re.escape(bundle)}_(\d{{8}}_\d{{6}})$'
    )
    candidates: list[tuple[str, Path]] = []
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if m:
            candidates.append((m.group(1), p))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def _load_metrics_row(model_dir: Path) -> Optional[dict]:
    csv = model_dir / 'post_hoc' / 'metrics.csv'
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    if df.empty:
        return None
    return {k: float(df[k].iloc[0]) for k in df.columns}


def build_summary(runs_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for pair_label, pair_short in _PAIRS:
        for routing_label, bundle_tmpl in _ROUTINGS:
            bundle = bundle_tmpl.format(pair=pair_short)
            model_dir = _latest_lgbm_dir(runs_root, bundle)
            row: dict = {
                'pair': pair_label,
                'routing': routing_label,
                'bundle': bundle,
                'model_dir': str(model_dir) if model_dir else '',
            }
            metrics = _load_metrics_row(model_dir) if model_dir else None
            if metrics is None:
                for col, _disp in _METRIC_DISPLAY:
                    row[col] = float('nan')
            else:
                for col, _disp in _METRIC_DISPLAY:
                    row[col] = metrics.get(col, float('nan'))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(df: pd.DataFrame, out_png: Path) -> None:
    n_metrics = len(_METRIC_DISPLAY)
    fig, axes = plt.subplots(
        nrows=len(_PAIRS), ncols=n_metrics,
        figsize=(3.4 * n_metrics, 3.4 * len(_PAIRS)),
        sharey='col',
    )
    if len(_PAIRS) == 1:
        axes = np.array([axes])
    if n_metrics == 1:
        axes = axes[:, None]

    routings = [r[0] for r in _ROUTINGS]
    # Color by alphabet/routing class so visual cues are consistent across panels.
    colors = {
        'seq_disjoint':     '#7a7a7a',     # neutral gray (baseline)
        'aa cluster_id099': '#1f78b4',     # blue (aa)
        'nt cluster_id100': '#33a02c',     # green dark (nt strict)
        'nt cluster_id099': '#b2df8a',     # green light (nt 99)
    }

    for row, (pair_label, _) in enumerate(_PAIRS):
        sub = df[df['pair'] == pair_label].set_index('routing').reindex(routings)
        for col, (metric_col, metric_disp) in enumerate(_METRIC_DISPLAY):
            ax = axes[row, col]
            values = sub[metric_col].values
            xs = np.arange(len(routings))
            ax.bar(xs, values, color=[colors[r] for r in routings],
                   edgecolor='black', linewidth=0.6)
            ax.set_xticks(xs)
            ax.set_xticklabels(routings, rotation=30, ha='right', fontsize=8)
            ax.set_ylim(0.0, 1.0)
            ax.set_axisbelow(True)
            ax.grid(axis='y', linestyle=':', alpha=0.5)
            if row == 0:
                ax.set_title(metric_disp, fontsize=11)
            if col == 0:
                ax.set_ylabel(f'{pair_label}\n{metric_disp}', fontsize=10)
            else:
                ax.set_ylabel(metric_disp, fontsize=10)
            for x, v in zip(xs, values):
                if np.isfinite(v):
                    ax.annotate(f'{v:.3f}', xy=(x, v),
                                xytext=(0, 2), textcoords='offset points',
                                ha='center', fontsize=7)

    fig.suptitle(
        'LGBM test metrics: seq_disjoint vs aa/nt cluster_disjoint '
        '(Experiment B-nt)\nFull Flu A corpus; k-mer k=6 nt features, '
        'unit_norm + unit_diff+prod, single seed.',
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--runs_root', default=str(PROJ / 'models/flu/July_2025/runs'),
                   help='Directory containing baseline_lgbm_<bundle>_<TS>/ runs.')
    p.add_argument('--out_dir',
                   default=str(PROJ / 'results/flu/July_2025/runs/cluster_aa_vs_nt'),
                   help='Output directory for the PNG + CSV.')
    args = p.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise SystemExit(f'runs_root not found: {runs_root}')
    out_dir = Path(args.out_dir)

    df = build_summary(runs_root)
    print('\nSummary frame:')
    print(df.to_string(index=False))
    if df[[m for m, _ in _METRIC_DISPLAY]].isna().any(axis=None):
        missing = df[df['model_dir'] == ''][['pair', 'routing', 'bundle']]
        if not missing.empty:
            print('\nWARNING: some bundles have no LGBM run yet:')
            print(missing.to_string(index=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'lgbm_cluster_aa_vs_nt.csv'
    out_png = out_dir / 'lgbm_cluster_aa_vs_nt.png'
    df.to_csv(out_csv, index=False)
    plot_summary(df, out_png)
    print(f'\nWrote: {out_csv}')
    print(f'Wrote: {out_png}')


if __name__ == '__main__':
    main()
