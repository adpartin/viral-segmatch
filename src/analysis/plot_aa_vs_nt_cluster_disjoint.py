"""Plot the aa-vs-nt cluster_disjoint comparison (Experiment B-nt).

For each schema pair (HA/NA, PB2/PB1), each routing configuration
{seq_disjoint, aa-cluster_id099, nt-cluster_id100, nt-cluster_id099},
and each model in {LGBM, 1-NN cosine margin}, this script reads
`post_hoc/metrics.csv` from the latest matching baseline run and emits
a grouped-bar PNG + a flat CSV summary.

1-NN cosine margin is the operational leakage diagnostic described in
`docs/methods/leakage_definitions.md` — its prediction is exactly the
label of the nearest train pair under cosine distance, so its accuracy
is the upper bound on what near-neighbor lookup can achieve on the given
dataset. The 1-NN-vs-LGBM gap at each routing therefore lower-bounds
what LGBM is doing *beyond* near-neighbor lookup. On Flu A (2026-05-15)
this gap turned out to be ≤0 at every cell — 1-NN matched LGBM at
id100 cells and slightly outperformed LGBM at id099 cells, indicating
that cluster_disjoint weakens the near-neighbor signal gradually
rather than removing it (see
`docs/results/2026-05-15_cluster_disjoint_nt_results.md` for the full
reading).

The reference deliverable promised by step 9 of
`docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` § B-nt
("threshold sweep plot the aa side couldn't reach") is implemented here
as a 4-routing comparison rather than a true threshold sweep — the
bipartite-component collapse on the full Flu A corpus rules out feasible
nt cluster_disjoint below id099 (see
`results/flu/July_2025/runs/cluster_disjoint_feasibility/feasibility_*_nt.csv`).

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


# Routing label → bundle name pattern (used to autodiscover the run).
# The pattern is matched against baseline_<model>_<bundle>_<TS>
# directories; the latest matching timestamp wins.
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
# (model_label, baseline_subdir_prefix). 1-NN cosine margin is the
# leakage diagnostic; LGBM is the production baseline.
_MODELS = [
    ('LGBM',           'lgbm',         '#1f78b4'),
    ('1-NN margin',    'knn1_margin',  '#e31a1c'),
]
_METRIC_DISPLAY = [
    ('f1_score', 'F1'),
    ('auc_pr',   'AUC-PR'),
    ('auc_roc',  'AUC-ROC'),
    ('mcc',      'MCC'),
]


def _latest_baseline_dir(runs_root: Path, model_prefix: str, bundle: str) -> Optional[Path]:
    """Return the latest `baseline_<model_prefix>_<bundle>_<YYYYMMDD_HHMMSS>` dir.

    Bundle match is anchored — `bundle='flu_ha_na'` does NOT match
    `flu_ha_na_cluster_id99`, etc.
    """
    pattern = re.compile(
        rf'^baseline_{re.escape(model_prefix)}_{re.escape(bundle)}_(\d{{8}}_\d{{6}})$'
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
            for model_label, model_prefix, _color in _MODELS:
                model_dir = _latest_baseline_dir(runs_root, model_prefix, bundle)
                row: dict = {
                    'pair': pair_label,
                    'routing': routing_label,
                    'model': model_label,
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
        figsize=(3.7 * n_metrics, 3.4 * len(_PAIRS)),
        sharey='col',
    )
    if len(_PAIRS) == 1:
        axes = np.array([axes])
    if n_metrics == 1:
        axes = axes[:, None]

    routings = [r[0] for r in _ROUTINGS]
    model_labels = [m[0] for m in _MODELS]
    model_colors = {m[0]: m[2] for m in _MODELS}

    n_models = len(_MODELS)
    width = 0.8 / n_models  # total group width = 0.8, evenly split across models

    for row, (pair_label, _) in enumerate(_PAIRS):
        sub = df[df['pair'] == pair_label]
        for col, (metric_col, metric_disp) in enumerate(_METRIC_DISPLAY):
            ax = axes[row, col]
            xs = np.arange(len(routings))
            for mi, model_label in enumerate(model_labels):
                vals = (
                    sub[sub['model'] == model_label]
                    .set_index('routing')
                    .reindex(routings)[metric_col].values
                )
                offset = (mi - (n_models - 1) / 2) * width
                bars = ax.bar(xs + offset, vals, width,
                              color=model_colors[model_label],
                              edgecolor='black', linewidth=0.5,
                              label=model_label if (row == 0 and col == 0) else None)
                for x, v in zip(xs + offset, vals):
                    if np.isfinite(v):
                        ax.annotate(f'{v:.3f}', xy=(x, v),
                                    xytext=(0, 2), textcoords='offset points',
                                    ha='center', fontsize=6.5)
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

    # Legend once, top-right of the figure.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right', ncol=len(model_labels),
                   bbox_to_anchor=(0.99, 0.99), frameon=False)

    fig.suptitle(
        'LGBM vs 1-NN cosine margin test metrics across routings '
        '(Experiment B-nt)\nFull Flu A corpus; k-mer k=6 nt features, '
        'unit_norm + unit_diff+prod, single seed. The 1-NN gap to LGBM '
        'at each routing is the residual-leakage signal.',
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
    out_csv = out_dir / 'cluster_aa_vs_nt.csv'
    out_png = out_dir / 'cluster_aa_vs_nt.png'
    df.to_csv(out_csv, index=False)
    plot_summary(df, out_png)
    print(f'\nWrote: {out_csv}')
    print(f'Wrote: {out_png}')


if __name__ == '__main__':
    main()
