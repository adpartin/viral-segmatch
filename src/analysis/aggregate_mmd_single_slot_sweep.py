"""Aggregate + plot the single-slot HA-only idXX-sweep MMD results.

Loads the 18 phase2 MMD CSVs produced by mmd_per_slot.py and mmd_per_pair.py
across {id100, id099, id098, id097, id096, id095} × {S1 HA, S1 NA, S2 pair}
for one feature space, and produces:

  - A combined CSV with one row per (idXX, slot/pair) and the MMD² + p-value.
  - A two-panel figure: MMD² vs idXX, with three lines per panel (HA, NA, pair).
    One panel per feature space available. Significant cells (p <= 0.05) drawn
    with filled markers, non-significant with open markers.

Optionally overlays the random / seq_disjoint / bilateral cluster_id099 baselines
as horizontal reference lines for context (one set per feature space).

Run:
    python -m src.analysis.aggregate_mmd_single_slot_sweep \\
        --results_dir results/flu/July_2025/runs/split_separation_mmd \\
        --pair_label HA-NA \\
        --feature_spaces esm2 kmer_aa \\
        --out_dir results/flu/July_2025/runs/split_separation_mmd/sweep_aggregate
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt


SWEEP_THRESHOLDS = [100, 99, 98, 97, 96, 95]   # high -> low (id↓ moves right on x-axis)

# Each model has a different output-dir prefix under models/.../runs/.
MODEL_DIR_PREFIX = {
    'mlp':         'training_flu_ha_na_kmer_aa_k3_HAonly_id',
    'lgbm':        'baseline_lgbm_flu_ha_na_kmer_aa_k3_HAonly_id',
    'knn1_margin': 'baseline_knn1_margin_flu_ha_na_kmer_aa_k3_HAonly_id',
}
MODEL_STYLE = {
    'mlp':         {'color': '#1f77b4', 'marker': 'o', 'label': 'MLP'},
    'lgbm':        {'color': '#2ca02c', 'marker': 's', 'label': 'LGBM'},
    'knn1_margin': {'color': '#d62728', 'marker': '^', 'label': '1-NN (cosine margin)'},
}


def _feature_suffix(feature_space: str) -> str:
    """File suffix convention used by the MMD scripts."""
    if feature_space == 'esm2':
        return 'esm2'
    if feature_space == 'kmer_aa':
        return 'kmer_aa_k3'
    if feature_space == 'kmer_nt':
        return 'kmer_nt_k6'
    raise ValueError(f"unknown feature_space: {feature_space}")


def _load_one(results_dir: Path, idxx: int, role: str, feature_space: str,
              label_filter: str = 'pos') -> Optional[dict]:
    """Load one phase2 MMD CSV. role in {'HA', 'NA', 'pair'}.
    `label_filter` in {'pos' (default), 'neg', 'both'} selects which
    label-filtered MMD file to load (file naming follows the suffix
    convention: pos → no suffix, neg → '_neg', both → '_both').
    """
    fs = _feature_suffix(feature_space)
    if label_filter == 'pos':
        lf_suffix = ''
    elif label_filter == 'neg':
        lf_suffix = '_neg'
    elif label_filter == 'both':
        lf_suffix = '_both'
    else:
        raise ValueError(f"label_filter must be 'pos'|'neg'|'both', got {label_filter}")
    if role == 'pair':
        # S2 (pair) file naming: phase2_perm_<label>_HA_NA_pair_<fs>_test3[_{neg,both}].csv
        fname = (f"phase2_perm_cluster_aa_id{idxx:03d}_HAonly_"
                 f"HA_NA_pair_{fs}_test3{lf_suffix}.csv")
    else:
        # S1 (slot) file naming: phase2_perm_<label>_<HA|NA>_<fs>[_{neg,both}].csv
        fname = (f"phase2_perm_cluster_aa_id{idxx:03d}_HAonly_"
                 f"{role}_{fs}{lf_suffix}.csv")
    path = results_dir / fname
    if not path.exists():
        return None
    df = pd.read_csv(path)
    assert len(df) == 1, f"expected 1 row in {path}, got {len(df)}"
    row = df.iloc[0]
    return {
        'idxx': idxx,
        'role': role,
        'feature_space': feature_space,
        'label_filter': label_filter,
        'mmd2': float(row['mmd2']),
        'p_value': float(row['p_value']),
        'n_train': int(row['n_A']),
        'n_test': int(row['n_B']),
        'sigma': float(row['sigma']),
        'n_extreme': int(row['n_extreme']) if not pd.isna(row['n_extreme']) else None,
        'source_file': str(path),
    }


def _load_reference_baselines(results_dir: Path, feature_space: str
                                ) -> pd.DataFrame:
    """Existing random / seq_disjoint / bilateral cluster_id099 baselines.

    Returns one row per (routing, role) with the same MMD² + p-value columns.
    Skips missing files silently — useful when a feature space wasn't run on
    a particular baseline.
    """
    fs = _feature_suffix(feature_space)
    rows = []
    # naming used by prior runs (per docs/results/2026-05-24_mmd_per_*_results.md):
    #   slot: phase2_perm_<routing>_<HA|NA>_<fs>.csv
    #   pair: phase2_perm_<routing>_HA_NA_pair_<fs>_test3.csv
    for routing in ('random', 'seq_disjoint', 'cluster_disjoint_id099'):
        for role in ('HA', 'NA', 'pair'):
            if role == 'pair':
                path = results_dir / f"phase2_perm_{routing}_HA_NA_pair_{fs}_test3.csv"
            else:
                path = results_dir / f"phase2_perm_{routing}_{role}_{fs}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if len(df) != 1:
                continue
            row = df.iloc[0]
            rows.append({
                'routing':       routing,
                'role':          role,
                'feature_space': feature_space,
                'mmd2':          float(row['mmd2']),
                'p_value':       float(row['p_value']),
                'sigma':         float(row['sigma']),
                'source_file':   str(path),
            })
    return pd.DataFrame(rows)


def _load_perf_for_idxx(models_dir: Path, idxx: int, model: str
                          ) -> Optional[dict]:
    """Locate the most recent (idxx, model) training/baseline dir and read its
    post_hoc/metrics.csv. Returns the metrics dict + dir path, or None if no
    matching dir found.
    """
    prefix = MODEL_DIR_PREFIX[model]
    candidates = sorted(
        models_dir.glob(f'{prefix}{idxx:03d}_*'),
        key=lambda p: p.name,
        reverse=True,
    )
    for cand in candidates:
        metrics_csv = cand / 'post_hoc' / 'metrics.csv'
        if metrics_csv.exists():
            row = pd.read_csv(metrics_csv).iloc[0]
            return {
                'idxx':      idxx,
                'model':     model,
                'f1_score':  float(row['f1_score']),
                'mcc':       float(row['mcc']),
                'auc_roc':   float(row['auc_roc']),
                'auc_pr':    float(row.get('avg_precision', float('nan'))),
                'accuracy':  float(row['accuracy']),
                'precision': float(row['precision']),
                'recall':    float(row['recall']),
                'brier':     float(row['brier_score']),
                'source_dir': str(cand),
            }
    return None


def aggregate_perf(models_dir: Path, models: list) -> pd.DataFrame:
    rows = []
    for idxx in SWEEP_THRESHOLDS:
        for model in models:
            r = _load_perf_for_idxx(models_dir, idxx, model)
            if r is not None:
                rows.append(r)
    return pd.DataFrame(rows)


def aggregate(results_dir: Path, feature_spaces: list,
              label_filters: tuple = ('pos', 'neg', 'both')) -> pd.DataFrame:
    """Aggregate MMD across (feature_space × idXX × role × label_filter).
    Missing files silently skipped — useful when only some label_filters
    were run for a given feature space."""
    rows = []
    for fs in feature_spaces:
        for idxx in SWEEP_THRESHOLDS:
            for role in ('HA', 'NA', 'pair'):
                for lf in label_filters:
                    r = _load_one(results_dir, idxx, role, fs, label_filter=lf)
                    if r is not None:
                        rows.append(r)
    return pd.DataFrame(rows)


# Visual conventions: HA=constrained=blue, NA=unconstrained=orange, pair=purple.
ROLE_STYLE = {
    'HA':   {'color': '#1f77b4', 'marker': 'o', 'label': 'S1 HA (constrained)'},
    'NA':   {'color': '#ff7f0e', 'marker': 's', 'label': 'S1 NA (unconstrained)'},
    'pair': {'color': '#9467bd', 'marker': '^', 'label': 'S2 pair (Test 3)'},
}


def plot_sweep(sweep_df: pd.DataFrame, refs_by_fs: dict, out_path: Path,
               label_filter: str = 'pos') -> None:
    """One subplot per feature space. x = idXX (high -> low), y = MMD².
    Plots only the rows matching `label_filter` (default 'pos' keeps backward
    compat with the original positives-only sweep)."""
    sweep_df = sweep_df[sweep_df['label_filter'] == label_filter]
    feature_spaces = sorted(sweep_df['feature_space'].unique())
    n = len(feature_spaces)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.0), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, fs in zip(axes, feature_spaces):
        sub = sweep_df[sweep_df['feature_space'] == fs].sort_values('idxx', ascending=False)

        # Map idXX to x-axis position: high idXX on the LEFT (lighter constraint),
        # low idXX on the RIGHT (heavier constraint).
        idxx_sorted = sorted(sub['idxx'].unique(), reverse=True)
        x_positions = {idxx: i for i, idxx in enumerate(idxx_sorted)}

        for role, style in ROLE_STYLE.items():
            role_df = sub[sub['role'] == role].sort_values('idxx', ascending=False)
            if role_df.empty:
                continue
            xs = [x_positions[v] for v in role_df['idxx']]
            ys = role_df['mmd2'].values
            ps = role_df['p_value'].values

            ax.plot(xs, ys, color=style['color'], linewidth=1.5, alpha=0.7,
                    label=style['label'])
            # Filled marker if p<=0.05, hollow otherwise — visual significance hint.
            for x, y, p in zip(xs, ys, ps):
                if p <= 0.05:
                    ax.scatter([x], [y], color=style['color'], marker=style['marker'],
                               s=80, zorder=5, edgecolor='black', linewidth=0.5)
                else:
                    ax.scatter([x], [y], facecolor='white', edgecolor=style['color'],
                               marker=style['marker'], s=80, zorder=5, linewidth=1.5)

        # Reference baselines as horizontal lines (random and cluster_id099 only,
        # keeps it readable; seq_disjoint suppressed by default).
        refs = refs_by_fs.get(fs)
        if refs is not None and not refs.empty:
            for routing, linestyle, alpha in (
                ('random', ':', 0.5),
                ('cluster_disjoint_id099', '--', 0.45),
            ):
                rsub = refs[refs['routing'] == routing]
                for role, style in ROLE_STYLE.items():
                    rval = rsub[rsub['role'] == role]
                    if rval.empty:
                        continue
                    ax.axhline(rval['mmd2'].values[0],
                               color=style['color'], linestyle=linestyle,
                               alpha=alpha, linewidth=1.0)

        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels([f"id{v:03d}" for v in x_positions.keys()])
        ax.set_xlabel('Cluster identity threshold (constraint strength: lighter → heavier)')
        ax.set_ylabel('MMD²  (RBF, fixed σ per feature space)')
        ax.set_title(f"Single-slot HA-only idXX sweep · {fs}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    fig.suptitle('S1+S2 MMD vs cluster-identity threshold under aa cluster_disjoint single-slot HA-only routing\n'
                 '(dotted = random baseline · dashed = bilateral cluster_id099 · filled marker = p ≤ 0.05)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote plot: {out_path}")


def plot_perf(perf_df: pd.DataFrame, out_path: Path,
              metrics: list = ('f1_score', 'auc_roc', 'mcc')) -> None:
    """Three subplots, one per metric. x = idXX (high -> low), y = metric.
    One line per model (MLP, LGBM, 1-NN)."""
    if perf_df.empty:
        print('plot_perf: no rows to plot, skipping.')
        return
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharex=True)
    if n == 1:
        axes = [axes]
    idxx_sorted = sorted(perf_df['idxx'].unique(), reverse=True)
    x_positions = {v: i for i, v in enumerate(idxx_sorted)}

    for ax, metric in zip(axes, metrics):
        for model, style in MODEL_STYLE.items():
            sub = perf_df[perf_df['model'] == model].sort_values('idxx', ascending=False)
            if sub.empty:
                continue
            xs = [x_positions[v] for v in sub['idxx']]
            ys = sub[metric].values
            ax.plot(xs, ys, color=style['color'], linewidth=1.5, alpha=0.85,
                    marker=style['marker'], markersize=8,
                    markeredgecolor='black', markeredgewidth=0.5,
                    label=style['label'])
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels([f'id{v:03d}' for v in x_positions.keys()])
        ax.set_xlabel('Cluster identity threshold (lighter constraint → heavier)')
        ax.set_ylabel(metric)
        ax.set_title(f'Test {metric} vs idXX')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)

    fig.suptitle('Held-out test performance vs cluster-identity threshold\n'
                 '(single-slot HA-only routing, aa k=3 features, Test 3 interaction)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote perf plot: {out_path}')


def plot_mmd_vs_perf(sweep_df: pd.DataFrame, perf_df: pd.DataFrame,
                      out_path: Path, mmd_role: str = 'pair',
                      mmd_feature: str = 'kmer_aa',
                      mmd_label_filter: str = 'both',
                      metric: str = 'f1_score') -> None:
    """Single panel: x = MMD² of one (role, feature, label_filter),
    y = perf metric. One marker per (model × idXX); idXX labeled by
    annotation. Default `mmd_label_filter='both'` matches what the
    model actually trains on (pos + neg)."""
    if perf_df.empty or sweep_df.empty:
        print('plot_mmd_vs_perf: missing data, skipping.')
        return
    mmd_sub = sweep_df[(sweep_df['feature_space'] == mmd_feature)
                       & (sweep_df['role'] == mmd_role)
                       & (sweep_df['label_filter'] == mmd_label_filter)
                       ][['idxx', 'mmd2']]
    if mmd_sub.empty:
        print(f'plot_mmd_vs_perf: no MMD rows for ({mmd_feature}, {mmd_role}, {mmd_label_filter}).')
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    merged = perf_df.merge(mmd_sub, on='idxx')
    for model, style in MODEL_STYLE.items():
        sub = merged[merged['model'] == model].sort_values('mmd2')
        if sub.empty:
            continue
        ax.plot(sub['mmd2'], sub[metric], color=style['color'], linewidth=1.5,
                alpha=0.85, marker=style['marker'], markersize=10,
                markeredgecolor='black', markeredgewidth=0.5,
                label=style['label'])
        for _, r in sub.iterrows():
            ax.annotate(f'id{int(r["idxx"]):03d}',
                        (r['mmd2'], r[metric]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    ax.set_xlabel(f'MMD²  ({mmd_feature} {mmd_role}, label={mmd_label_filter}; '
                  f'smaller → more train/test overlap)')
    ax.set_ylabel(f'Test {metric}')
    ax.set_title(f'Test {metric} vs MMD² — does distribution shift predict perf drop?')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote MMD-vs-perf plot: {out_path}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--results_dir', type=Path,
                    default=Path('results/flu/July_2025/runs/split_separation_mmd'))
    ap.add_argument('--feature_spaces', nargs='+',
                    default=['esm2', 'kmer_aa'],
                    choices=['esm2', 'kmer_aa', 'kmer_nt'])
    ap.add_argument('--out_dir', type=Path,
                    default=Path('results/flu/July_2025/runs/split_separation_mmd/sweep_aggregate'))
    ap.add_argument('--models_dir', type=Path,
                    default=Path('models/flu/July_2025/runs'),
                    help='Where MLP + baseline training output dirs live.')
    ap.add_argument('--models', nargs='+',
                    default=['mlp', 'lgbm', 'knn1_margin'],
                    help='Which trained models to aggregate perf for. '
                         'Skips any whose dirs do not exist yet.')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sweep_df = aggregate(args.results_dir, args.feature_spaces)
    print(f"Loaded {len(sweep_df)} sweep cells across {sweep_df['feature_space'].nunique()} "
          f"feature spaces and {sweep_df['idxx'].nunique()} thresholds.")

    sweep_csv = args.out_dir / 'sweep_combined.csv'
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"Wrote combined sweep CSV: {sweep_csv}")

    refs_by_fs = {fs: _load_reference_baselines(args.results_dir, fs)
                  for fs in args.feature_spaces}
    refs_combined = pd.concat([df for df in refs_by_fs.values() if not df.empty],
                              ignore_index=True)
    if not refs_combined.empty:
        refs_csv = args.out_dir / 'reference_baselines.csv'
        refs_combined.to_csv(refs_csv, index=False)
        print(f"Wrote reference baselines CSV: {refs_csv}  ({len(refs_combined)} rows)")

    # Backward-compat: the original "main" trajectory plot is positives-only.
    plot_path = args.out_dir / 'sweep_mmd_vs_idxx.png'
    plot_sweep(sweep_df, refs_by_fs, plot_path, label_filter='pos')

    # Perf rollup — silently skips any model whose dirs don't exist yet.
    perf_df = aggregate_perf(args.models_dir, args.models)
    if not perf_df.empty:
        perf_csv = args.out_dir / 'sweep_perf.csv'
        perf_df.to_csv(perf_csv, index=False)
        print(f"Wrote perf CSV: {perf_csv}  ({len(perf_df)} rows)")
        plot_perf(perf_df, args.out_dir / 'sweep_perf_vs_idxx.png')
        # Pair MMD on kmer_aa is the natural x-axis (matches Test 3 + the
        # model's feature space). Emit one F1-vs-MMD plot per label_filter
        # that has rows; 'both' is the most defensible single summary.
        for lf in ('pos', 'neg', 'both'):
            n_rows = ((sweep_df['feature_space'] == 'kmer_aa')
                      & (sweep_df['role'] == 'pair')
                      & (sweep_df['label_filter'] == lf)).sum()
            if n_rows == 0:
                continue
            plot_mmd_vs_perf(sweep_df, perf_df,
                              args.out_dir / f'sweep_perf_vs_mmd_pair_kmer_aa_{lf}.png',
                              mmd_role='pair', mmd_feature='kmer_aa',
                              mmd_label_filter=lf, metric='f1_score')
    else:
        print('No perf rows aggregated (training likely still in progress).')

    # Compact per-feature summary table to stdout. One block per (feature_space,
    # label_filter) since (idxx × role) can now be non-unique across label_filters.
    print()
    for fs in args.feature_spaces:
        for lf in sorted(sweep_df['label_filter'].unique()):
            sub = sweep_df[(sweep_df['feature_space'] == fs) & (sweep_df['label_filter'] == lf)]
            if sub.empty:
                continue
            pivot = sub.pivot(index='idxx', columns='role', values='mmd2').sort_index(ascending=False)
            pivot_p = sub.pivot(index='idxx', columns='role', values='p_value').sort_index(ascending=False)
            print(f"[{fs} | label={lf}] MMD² table:")
            print(pivot.round(5).to_string())
            print(f"[{fs} | label={lf}] p-value table:")
            print(pivot_p.round(4).to_string())
            print()


if __name__ == '__main__':
    main()
