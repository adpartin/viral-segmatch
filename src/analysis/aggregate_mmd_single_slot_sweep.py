"""Aggregate + plot a single-slot idXX-sweep MMD results.

Loads the phase2 MMD CSVs produced by mmd_per_slot.py and mmd_per_pair.py
across {id100, id099, id098, id097, id096, id095} × {S1 slot-a, S1 slot-b,
S2 pair} for one or more feature spaces, and produces:

  - A combined CSV with one row per (idXX, slot/pair) and the MMD² + p-value.
  - A two-panel figure: MMD² vs idXX, with three lines per panel
    (slot-a, slot-b, pair). One panel per feature space available.
    Significant cells (p <= 0.05) drawn with filled markers,
    non-significant with open markers.

Optionally overlays the random / seq_disjoint / bilateral cluster_id099
baselines as horizontal reference lines for context. Reference-baseline
files are HA-NA-specific (only built for the original sweep) and silently
skipped when missing.

Parametrized for any single-slot sweep produced by mmd_sweep.sh +
stage4_sweep.sh. The MMD output files always use literal `_HA_` / `_NA_`
for slot-a / slot-b regardless of actual protein names (a contract with
mmd_sweep.sh); the aggregator preserves that lookup convention and lets
the user override the *display* names via --slot_a_display /
--slot_b_display.

Defaults reproduce the original HA-NA HA-only sweep behavior.

Run (HA-NA HA-only, the default):
    python -m src.analysis.aggregate_mmd_single_slot_sweep \\
        --feature_spaces esm2 kmer_aa

Run (PB2-PB1 PB2-only):
    python -m src.analysis.aggregate_mmd_single_slot_sweep \\
        --routing_direction PB2only \\
        --training_bundle flu_pb2_pb1_kmer_aa_k3 \\
        --slot_a_display PB2 --slot_b_display PB1 \\
        --pair_display PB2-PB1 \\
        --out_subdir pb2_pb1_PB2only \\
        --feature_spaces kmer_aa
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


def make_model_dir_prefix(training_bundle: str, routing_direction: str) -> dict:
    """Per-model output-dir prefix under models/.../runs/. Pattern matches
    what stage4_sweep.sh writes: `training_<bundle>_<direction>_id<idxx>_...`
    and `baseline_<name>_<bundle>_<direction>_id<idxx>_...`."""
    base = f"{training_bundle}_{routing_direction}_id"
    return {
        'mlp':         f"training_{base}",
        'lgbm':        f"baseline_lgbm_{base}",
        'knn1_margin': f"baseline_knn1_margin_{base}",
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
              routing_direction: str = 'HAonly',
              label_filter: str = 'pos') -> Optional[dict]:
    """Load one phase2 MMD CSV. role in {'HA', 'NA', 'pair'} (literal —
    mmd_sweep.sh always writes slot-a as `_HA_` and slot-b as `_NA_`
    regardless of actual protein).
    `routing_direction` is the suffix on the routing label (e.g.,
    'HAonly', 'PB2only'); it identifies which single-slot sweep this is.
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
        fname = (f"phase2_perm_cluster_aa_id{idxx:03d}_{routing_direction}_"
                 f"HA_NA_pair_{fs}_test3{lf_suffix}.csv")
    else:
        # S1 (slot) file naming: phase2_perm_<label>_<HA|NA>_<fs>[_{neg,both}].csv
        fname = (f"phase2_perm_cluster_aa_id{idxx:03d}_{routing_direction}_"
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


def _seed_from_dirname(name: str, default: int = 42) -> int:
    """Extract seed from dir name pattern `..._seed{N}_{timestamp}`. Falls
    back to `default` if no `_seedN_` segment is present (the original
    single-seed runs were named without it; master_seed=42 was the
    default)."""
    m = re.search(r'_seed(\d+)_', name)
    return int(m.group(1)) if m else default


def _load_perf_for_idxx(models_dir: Path, idxx: int, model: str,
                          model_dir_prefix: dict) -> list:
    """Return one dict per training run found for this (idxx, model) —
    one row per seed. A run is "found" iff its `post_hoc/metrics.csv`
    exists. Seed is parsed from the dir name (`_seedN_` segment, else 42).
    """
    prefix = model_dir_prefix[model]
    candidates = sorted(models_dir.glob(f'{prefix}{idxx:03d}_*'),
                        key=lambda p: p.name)
    seen_seeds = {}
    for cand in candidates:
        metrics_csv = cand / 'post_hoc' / 'metrics.csv'
        if not metrics_csv.exists():
            continue
        seed = _seed_from_dirname(cand.name)
        # If we have multiple runs for the same seed, keep the most recent
        # (sorted ascending → later entries are newer).
        row = pd.read_csv(metrics_csv).iloc[0]
        seen_seeds[seed] = {
            'idxx':      idxx,
            'model':     model,
            'seed':      seed,
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
    return list(seen_seeds.values())


def aggregate_perf(models_dir: Path, models: list,
                    model_dir_prefix: dict) -> pd.DataFrame:
    """Long-format perf table: one row per (idxx, model, seed)."""
    rows = []
    for idxx in SWEEP_THRESHOLDS:
        for model in models:
            rows.extend(_load_perf_for_idxx(models_dir, idxx, model, model_dir_prefix))
    return pd.DataFrame(rows)


def aggregate_perf_summary(perf_df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/min/max/n per (idxx, model) across seeds for each metric."""
    if perf_df.empty:
        return perf_df
    metrics = ['f1_score', 'mcc', 'auc_roc', 'auc_pr', 'accuracy',
               'precision', 'recall', 'brier']
    agg = perf_df.groupby(['idxx', 'model'])[metrics].agg(['mean', 'std', 'min', 'max', 'count'])
    agg.columns = [f'{m}_{stat}' for m, stat in agg.columns]
    return agg.reset_index()


def aggregate(results_dir: Path, feature_spaces: list,
              routing_direction: str = 'HAonly',
              label_filters: tuple = ('pos', 'neg', 'both')) -> pd.DataFrame:
    """Aggregate MMD across (feature_space × idXX × role × label_filter).
    Missing files silently skipped — useful when only some label_filters
    were run for a given feature space."""
    rows = []
    for fs in feature_spaces:
        for idxx in SWEEP_THRESHOLDS:
            for role in ('HA', 'NA', 'pair'):
                for lf in label_filters:
                    r = _load_one(results_dir, idxx, role, fs,
                                  routing_direction=routing_direction,
                                  label_filter=lf)
                    if r is not None:
                        rows.append(r)
    return pd.DataFrame(rows)


# Visual conventions: slot-a=constrained=blue, slot-b=unconstrained=orange, pair=purple.
# The dict keys ('HA', 'NA', 'pair') match the literal role tokens in MMD
# filenames; the label text is constructed from the user-supplied
# slot_a_display / slot_b_display so plots show the correct protein names.
def make_role_style(slot_a_display: str, slot_b_display: str) -> dict:
    return {
        'HA':   {'color': '#1f77b4', 'marker': 'o',
                 'label': f'S1 {slot_a_display} (constrained)'},
        'NA':   {'color': '#ff7f0e', 'marker': 's',
                 'label': f'S1 {slot_b_display} (unconstrained)'},
        'pair': {'color': '#9467bd', 'marker': '^',
                 'label': 'S2 pair (Test 3)'},
    }


def plot_sweep(sweep_df: pd.DataFrame, refs_by_fs: dict, out_path: Path,
               role_style: dict,
               routing_direction: str = 'HAonly',
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

        for role, style in role_style.items():
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
                for role, style in role_style.items():
                    rval = rsub[rsub['role'] == role]
                    if rval.empty:
                        continue
                    ax.axhline(rval['mmd2'].values[0],
                               color=style['color'], linestyle=linestyle,
                               alpha=alpha, linewidth=1.0)

        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels([f"t{v:03d}" for v in x_positions.keys()])
        ax.set_xlabel('Cluster identity threshold (constraint strength: lighter → heavier)')
        ax.set_ylabel('MMD²  (RBF, fixed σ per feature space)')
        ax.set_title(f"Single-slot {routing_direction} idXX sweep · {fs}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    fig.suptitle(f'S1+S2 MMD vs cluster-identity threshold under aa cluster_disjoint single-slot {routing_direction} routing\n'
                 '(dotted = random baseline · dashed = bilateral cluster_id099 · filled marker = p ≤ 0.05)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote plot: {out_path}")


def plot_perf(perf_df: pd.DataFrame, out_path: Path,
              metrics: list = ('f1_score', 'auc_roc', 'mcc'),
              routing_direction: str = 'HAonly') -> None:
    """Three subplots, one per metric. x = idXX (high -> low), y = metric.
    Mean across seeds plotted as the line; ± std as a shaded band; individual
    seed values as small open markers. Single-seed cells show as a point
    only (no band)."""
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
            sub = perf_df[perf_df['model'] == model]
            if sub.empty:
                continue
            agg = sub.groupby('idxx')[metric].agg(['mean', 'std', 'count']).sort_index(ascending=False)
            xs = [x_positions[v] for v in agg.index]
            means = agg['mean'].values
            stds = agg['std'].fillna(0.0).values
            # Shaded ±1 std band (only when >1 seed).
            ax.fill_between(xs, means - stds, means + stds, color=style['color'], alpha=0.18)
            ax.plot(xs, means, color=style['color'], linewidth=1.5, alpha=0.9,
                    marker=style['marker'], markersize=8,
                    markeredgecolor='black', markeredgewidth=0.5,
                    label=f"{style['label']}  (n_seed≤{int(agg['count'].max())})")
            # Individual seed points as small open markers.
            for v in agg.index:
                ys_seed = sub[sub['idxx'] == v][metric].values
                if len(ys_seed) > 1:
                    ax.scatter([x_positions[v]] * len(ys_seed), ys_seed,
                               facecolor='none', edgecolor=style['color'],
                               marker=style['marker'], s=30, alpha=0.5, linewidth=0.8)
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels([f't{v:03d}' for v in x_positions.keys()])
        ax.set_xlabel('Cluster identity threshold (lighter constraint → heavier)')
        ax.set_ylabel(metric)
        ax.set_title(f'Test {metric} vs idXX')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)

    fig.suptitle(f'Held-out test performance vs cluster-identity threshold\n'
                 f'(single-slot {routing_direction} routing, aa k=3 features, Test 3 interaction; '
                 'line=mean, band=±1 std, dots=per-seed)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
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
        sub = merged[merged['model'] == model]
        if sub.empty:
            continue
        # Aggregate mean ± std per idxx (and hence per mmd2).
        agg = (sub.groupby(['idxx', 'mmd2'])[metric]
               .agg(['mean', 'std', 'count']).reset_index()
               .sort_values('mmd2'))
        ax.errorbar(agg['mmd2'], agg['mean'],
                    yerr=agg['std'].fillna(0.0),
                    color=style['color'], linewidth=1.5, alpha=0.9,
                    marker=style['marker'], markersize=10,
                    markeredgecolor='black', markeredgewidth=0.5,
                    capsize=3, capthick=1.0,
                    label=f"{style['label']}  (n_seed≤{int(agg['count'].max())})")
        # Per-seed scatter overlay (small open markers).
        if (agg['count'] > 1).any():
            ax.scatter(sub['mmd2'], sub[metric],
                       facecolor='none', edgecolor=style['color'],
                       marker=style['marker'], s=20, alpha=0.4, linewidth=0.7)
        for _, r in agg.iterrows():
            ax.annotate(f't{int(r["idxx"]):03d}',
                        (r['mmd2'], r['mean']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    ax.set_xlabel(f'MMD²  ({mmd_feature} {mmd_role}, label={mmd_label_filter}; '
                  f'smaller → more train/test overlap)')
    ax.set_ylabel(f'Test {metric}  (mean ± std across seeds)')
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
                    default=Path('results/flu/July_2025/runs/split_separation_mmd/sweep_aggregate'),
                    help='Base output directory. If --out_subdir is set, '
                         'plots/CSVs land under <out_dir>/<out_subdir>/.')
    ap.add_argument('--out_subdir', type=str, default=None,
                    help='Sub-namespace under --out_dir to keep per-sweep '
                         'outputs separated. Convention since 2026-05-26: '
                         'always pass this (e.g. "ha_na_HAonly", '
                         '"pb2_pb1_PB2only"). If omitted, outputs land at '
                         '--out_dir root, which is fine for one-off runs '
                         'but will collide with the per-sweep dirs from '
                         'past runs.')
    ap.add_argument('--models_dir', type=Path,
                    default=Path('models/flu/July_2025/runs'),
                    help='Where MLP + baseline training output dirs live.')
    ap.add_argument('--models', nargs='+',
                    default=['mlp', 'lgbm', 'knn1_margin'],
                    help='Which trained models to aggregate perf for. '
                         'Skips any whose dirs do not exist yet.')
    # ----- experiment-identity flags (defaults reproduce HA-NA HA-only) -----
    ap.add_argument('--routing_direction', type=str, default='HAonly',
                    help='Suffix on the routing label (e.g., HAonly, '
                         'PB2only). Must match the --routing_label_pattern '
                         'used by mmd_sweep.sh.')
    ap.add_argument('--training_bundle', type=str,
                    default='flu_ha_na_kmer_aa_k3',
                    help='Training bundle name. Used to construct model + '
                         'baseline dir prefixes: '
                         '`training_<bundle>_<direction>_id<idxx>_*` etc.')
    ap.add_argument('--slot_a_display', type=str, default='HA',
                    help='Display name for slot a (the constrained slot) '
                         'in plot legends. MMD files always use literal '
                         '"HA" regardless — this is plot-text only.')
    ap.add_argument('--slot_b_display', type=str, default='NA',
                    help='Display name for slot b (the unconstrained slot).')
    ap.add_argument('--skip_reference_baselines', action='store_true',
                    help='Skip loading + plotting random / seq_disjoint / '
                         'bilateral cluster_id099 reference baselines. These '
                         'were only built for HA-NA; for any other pair they '
                         'live at the same results_dir but reflect HA-NA '
                         'data, so loading them would put misleading context '
                         'on a PB2/PB1 (etc.) plot. Set this flag for any '
                         'non-HA-NA sweep until that pair has its own '
                         'random/seq_disjoint baselines computed.')
    args = ap.parse_args()

    if args.out_subdir:
        args.out_dir = args.out_dir / args.out_subdir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model_dir_prefix = make_model_dir_prefix(args.training_bundle,
                                              args.routing_direction)
    role_style = make_role_style(args.slot_a_display, args.slot_b_display)

    sweep_df = aggregate(args.results_dir, args.feature_spaces,
                         routing_direction=args.routing_direction)
    print(f"Loaded {len(sweep_df)} sweep cells across {sweep_df['feature_space'].nunique()} "
          f"feature spaces and {sweep_df['idxx'].nunique()} thresholds.")

    sweep_csv = args.out_dir / 'sweep_combined.csv'
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"Wrote combined sweep CSV: {sweep_csv}")

    if args.skip_reference_baselines:
        refs_by_fs = {fs: pd.DataFrame() for fs in args.feature_spaces}
        print("Reference baselines: SKIPPED (--skip_reference_baselines)")
    else:
        refs_by_fs = {fs: _load_reference_baselines(args.results_dir, fs)
                      for fs in args.feature_spaces}
        refs_combined = pd.concat([df for df in refs_by_fs.values() if not df.empty],
                                  ignore_index=True)
        if not refs_combined.empty:
            refs_csv = args.out_dir / 'reference_baselines.csv'
            refs_combined.to_csv(refs_csv, index=False)
            print(f"Wrote reference baselines CSV: {refs_csv}  ({len(refs_combined)} rows)")
            if args.routing_direction != 'HAonly':
                print(f"  WARNING: --routing_direction={args.routing_direction} but reference "
                      "baselines were loaded. These were built for HA-NA — they may not be "
                      "the right context for this sweep. Use --skip_reference_baselines "
                      "to suppress.")

    # Backward-compat: the original "main" trajectory plot is positives-only.
    plot_path = args.out_dir / 'sweep_mmd_vs_idxx.png'
    plot_sweep(sweep_df, refs_by_fs, plot_path, role_style=role_style,
               routing_direction=args.routing_direction, label_filter='pos')

    # Perf rollup — long format (one row per idxx × model × seed) plus
    # a mean ± std summary across seeds.
    perf_df = aggregate_perf(args.models_dir, args.models, model_dir_prefix)
    if not perf_df.empty:
        perf_csv = args.out_dir / 'sweep_perf.csv'
        perf_df.to_csv(perf_csv, index=False)
        n_seed_max = int(perf_df.groupby(['idxx', 'model']).size().max())
        print(f"Wrote perf CSV: {perf_csv}  ({len(perf_df)} rows, "
              f"up to {n_seed_max} seeds per (idxx, model))")
        summary_df = aggregate_perf_summary(perf_df)
        summary_csv = args.out_dir / 'sweep_perf_summary.csv'
        summary_df.to_csv(summary_csv, index=False)
        print(f"Wrote perf summary CSV: {summary_csv}  ({len(summary_df)} rows)")
        plot_perf(perf_df, args.out_dir / 'sweep_perf_vs_idxx.png',
                  routing_direction=args.routing_direction)
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
