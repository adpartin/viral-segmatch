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


def _feature_suffix(feature_space: str) -> str:
    """File suffix convention used by the MMD scripts."""
    if feature_space == 'esm2':
        return 'esm2'
    if feature_space == 'kmer_aa':
        return 'kmer_aa_k3'
    if feature_space == 'kmer_nt':
        return 'kmer_nt_k6'
    raise ValueError(f"unknown feature_space: {feature_space}")


def _load_one(results_dir: Path, idxx: int, role: str, feature_space: str
              ) -> Optional[dict]:
    """Load one phase2 MMD CSV. role in {'HA', 'NA', 'pair'}."""
    fs = _feature_suffix(feature_space)
    if role == 'pair':
        # S2 (pair) file naming: phase2_perm_<label>_HA_NA_pair_<fs>_test3.csv
        fname = (f"phase2_perm_cluster_aa_id{idxx:03d}_HAonly_"
                 f"HA_NA_pair_{fs}_test3.csv")
    else:
        # S1 (slot) file naming: phase2_perm_<label>_<HA|NA>_<fs>.csv
        fname = (f"phase2_perm_cluster_aa_id{idxx:03d}_HAonly_"
                 f"{role}_{fs}.csv")
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


def aggregate(results_dir: Path, feature_spaces: list) -> pd.DataFrame:
    rows = []
    for fs in feature_spaces:
        for idxx in SWEEP_THRESHOLDS:
            for role in ('HA', 'NA', 'pair'):
                r = _load_one(results_dir, idxx, role, fs)
                if r is not None:
                    rows.append(r)
    return pd.DataFrame(rows)


# Visual conventions: HA=constrained=blue, NA=unconstrained=orange, pair=purple.
ROLE_STYLE = {
    'HA':   {'color': '#1f77b4', 'marker': 'o', 'label': 'S1 HA (constrained)'},
    'NA':   {'color': '#ff7f0e', 'marker': 's', 'label': 'S1 NA (unconstrained)'},
    'pair': {'color': '#9467bd', 'marker': '^', 'label': 'S2 pair (Test 3)'},
}


def plot_sweep(sweep_df: pd.DataFrame, refs_by_fs: dict, out_path: Path) -> None:
    """One subplot per feature space. x = idXX (high -> low), y = MMD²."""
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--results_dir', type=Path,
                    default=Path('results/flu/July_2025/runs/split_separation_mmd'))
    ap.add_argument('--feature_spaces', nargs='+',
                    default=['esm2', 'kmer_aa'],
                    choices=['esm2', 'kmer_aa', 'kmer_nt'])
    ap.add_argument('--out_dir', type=Path,
                    default=Path('results/flu/July_2025/runs/split_separation_mmd/sweep_aggregate'))
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

    plot_path = args.out_dir / 'sweep_mmd_vs_idxx.png'
    plot_sweep(sweep_df, refs_by_fs, plot_path)

    # Compact per-feature summary table to stdout.
    print()
    for fs in args.feature_spaces:
        sub = sweep_df[sweep_df['feature_space'] == fs]
        if sub.empty:
            print(f"[{fs}] no rows.")
            continue
        pivot = sub.pivot(index='idxx', columns='role', values='mmd2').sort_index(ascending=False)
        pivot_p = sub.pivot(index='idxx', columns='role', values='p_value').sort_index(ascending=False)
        print(f"[{fs}] MMD² table:")
        print(pivot.round(5).to_string())
        print(f"[{fs}] p-value table:")
        print(pivot_p.round(4).to_string())
        print()


if __name__ == '__main__':
    main()
