"""Consolidated clustering structural analysis (aa + nt, 8 major proteins).

Reads the per-function (major protein) redundancy sweep outputs (from
`seq_redundancy_per_function.py`) and bipartite-component
feasibility tables (from `cluster_disjoint_feasibility.py`) and emits
a single set of plots + tables that articulate, for each of the 8 major
Flu A protein functions at each mmseqs2 identity threshold:

  - How much sequence redundancy is in the underlying corpus (per
    function, per alphabet).
  - How quickly clusters collapse as the identity threshold drops.
  - Whether the resulting bipartite-component structure can support
    an 80/10/10 routing, per schema pair × alphabet.
  - How many residue mismatches each threshold *concretely* admits
    inside a cluster, per function — making "id095" mean something
    biological (`max_mutations = L - ceil(L * t)`).

This script is the structural counterpart to
`plot_aa_vs_nt_cluster_disjoint.py`, which reads model-results from
training run dirs. The two are kept separate because they have
different inputs and consumers:
- This script: reads cluster artifacts. Pure clustering analysis. No
  model outputs needed.
- plot_aa_vs_nt_cluster_disjoint.py: reads training metrics. Needs
  LGBM and 1-NN baselines to have run.

CLI:
    python -m src.analysis.cluster_analysis_summary \\
        [--clusters_aa  data/processed/flu/July_2025/clusters_aa] \\
        [--clusters_nt  data/processed/flu/July_2025/clusters_nt] \\
        [--protein_final data/processed/flu/July_2025/protein_final.csv] \\
        [--cds_final     data/processed/flu/July_2025/cds_final.parquet] \\
        [--feasibility_dir results/flu/July_2025/runs/cluster_disjoint_feasibility] \\
        [--out_dir         results/flu/July_2025/runs/cluster_analysis]

Outputs (under --out_dir):
    cluster_summary.csv               — per (function, alphabet, threshold) row
    sequence_length_summary.csv       — per (function, alphabet) length stats
    mutations_tolerated_table.csv     — per (function, alphabet, threshold) max mismatches
    unique_sequence_retention.png     — Plot A (2 grouped-bar panels)
    cluster_counts_vs_threshold.png   — Plot B (log-Y, 8 lines × 2 alphabets)
    bipartite_largest_pct_vs_threshold.png — Plot C (2 pairs × 2 alphabets, 80% line)
"""
from __future__ import annotations

import argparse
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


# Function short → full name (matches conf/virus/flu.yaml::function_short_names).
_FUNCTION_TO_SHORT = {
    'RNA-dependent RNA polymerase PB2 subunit': 'PB2',
    'RNA-dependent RNA polymerase catalytic core PB1 subunit': 'PB1',
    'RNA-dependent RNA polymerase PA subunit': 'PA',
    'Hemagglutinin precursor': 'HA',
    'Nucleocapsid protein': 'NP',
    'Neuraminidase protein': 'NA',
    'Matrix protein 1': 'M1',
    'Non-structural protein 1, interferon antagonist and host mRNA processing inhibitor': 'NS1',
}
_SHORT_ORDER = ['PB2', 'PB1', 'PA', 'HA', 'NP', 'NA', 'M1', 'NS1']
_FUNCTION_COLORS = {
    'PB2': '#1f77b4', 'PB1': '#ff7f0e', 'PA':  '#2ca02c', 'HA':  '#d62728',
    'NP':  '#9467bd', 'NA':  '#8c564b', 'M1':  '#e377c2', 'NS1': '#7f7f7f',
}

# Schema pairs we have feasibility data for.
_SCHEMA_PAIRS = [
    ('HA/NA',   'ha_na',   'HA',  'NA'),
    ('PB2/PB1', 'pb2_pb1', 'PB2', 'PB1'),
]


# ----------------------------------------------------------------------
# Input readers
# ----------------------------------------------------------------------

def load_redundancy_stats(clusters_root: Path, alphabet: str) -> pd.DataFrame:
    """Load redundancy_stats.csv emitted by seq_redundancy_per_function.py.

    `keep_default_na=False` is critical: the `function_short` column
    contains the literal string `'NA'` (Neuraminidase), which pandas
    would otherwise parse as NaN and silently drop downstream filters.
    """
    p = clusters_root / 'redundancy_stats.csv'
    if not p.exists():
        raise FileNotFoundError(f'{p} not found — run seq_redundancy_per_function.py first.')
    df = pd.read_csv(p, keep_default_na=False, na_values=[''])
    df = df[df['function_short'].isin(_SHORT_ORDER)].copy()
    df['alphabet'] = alphabet
    return df


def load_feasibility(feasibility_dir: Path, pair_short: str, alphabet: str) -> Optional[pd.DataFrame]:
    """Load the per-pair bipartite feasibility CSV.

    Default path (set by `cluster_disjoint_feasibility.py` since the
    2026-05-20 docs/results migration):
        <feasibility_dir>/feasibility_<pair_short>_<alphabet>.csv
    e.g. `results/.../cluster_disjoint_feasibility/feasibility_ha_na_aa.csv`.

    `pair_short` is lowercase ("ha_na", "pb2_pb1"). Returns None if the
    file is missing (graceful: the plot just drops that line).
    """
    p = feasibility_dir / f'feasibility_{pair_short.lower()}_{alphabet}.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p, keep_default_na=False, na_values=[''])
    df['alphabet'] = alphabet
    df['pair_short'] = pair_short
    return df


def compute_length_stats(protein_final: Path, cds_final: Path) -> pd.DataFrame:
    """Return per-(function, alphabet) sequence-length stats.

    Reads protein_final.csv for aa lengths (`length` column = AA count)
    and cds_final.parquet for nt lengths (`cds_length` column = nt count
    after splicing). Spliced rows that failed parse-time validation
    (BV-BRC sentinel `length=-1`) are absent from cds_final, which is
    fine — we want the lengths that the extractor actually produces.

    Returns a DataFrame indexed by `function_short` × `alphabet` with
    columns {n_rows, min, p25, median, p75, max, mean}.
    """
    rows: list[dict] = []
    if protein_final.exists():
        if protein_final.suffix == '.csv':
            # keep_default_na=False guards the project-wide 'NA'-string trap
            # (CLAUDE.md Conventions): the current `function` column uses
            # full names so no row is affected today, but adding the kwarg
            # defensively prevents a silent data loss if any future column
            # in protein_final.csv ever contains the literal string 'NA'.
            prot = pd.read_csv(
                protein_final, usecols=['function', 'length'],
                keep_default_na=False, na_values=[''],
            )
        else:
            prot = pd.read_parquet(protein_final, columns=['function', 'length'])
        prot = prot[prot['function'].isin(_FUNCTION_TO_SHORT)].copy()
        prot['function_short'] = prot['function'].map(_FUNCTION_TO_SHORT)
        for s, sub in prot.groupby('function_short'):
            rows.append({
                'function_short': s, 'alphabet': 'aa',
                'n_rows': int(len(sub)),
                'min': int(sub['length'].min()),
                'p25': int(sub['length'].quantile(0.25)),
                'median': int(sub['length'].median()),
                'p75': int(sub['length'].quantile(0.75)),
                'max': int(sub['length'].max()),
                'mean': float(sub['length'].mean()),
            })
    if cds_final.exists():
        cds = pd.read_parquet(cds_final, columns=['function', 'cds_length'])
        cds = cds[cds['function'].isin(_FUNCTION_TO_SHORT)].copy()
        cds['function_short'] = cds['function'].map(_FUNCTION_TO_SHORT)
        for s, sub in cds.groupby('function_short'):
            rows.append({
                'function_short': s, 'alphabet': 'nt',
                'n_rows': int(len(sub)),
                'min': int(sub['cds_length'].min()),
                'p25': int(sub['cds_length'].quantile(0.25)),
                'median': int(sub['cds_length'].median()),
                'p75': int(sub['cds_length'].quantile(0.75)),
                'max': int(sub['cds_length'].max()),
                'mean': float(sub['cds_length'].mean()),
            })
    return pd.DataFrame(rows)


def build_mutations_tolerated(length_stats: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """At each (function, alphabet, threshold), max admitted mismatches.

    max_mutations = L - ceil(L * t)

    Concrete biological meaning of "id <t>": inside a cluster, two
    sequences differ in at most this many residue positions
    (aa or nt depending on alphabet). The same threshold is
    biologically stricter on shorter proteins / CDS.

    The formula is written as `L - ceil(L * t)` rather than the
    mathematically equivalent `floor(L * (1 - t))` because the latter
    suffers float-precision error when `1 - t` produces a non-exact
    binary fraction (e.g., `1 - 0.9 = 0.0999...` in float, so
    `int(760 * (1 - 0.9)) = 75` rather than the correct 76).
    """
    import math
    out_rows: list[dict] = []
    for _, row in length_stats.iterrows():
        L = row['median']
        for t in thresholds:
            mut = int(L - math.ceil(L * t))
            out_rows.append({
                'function_short': row['function_short'],
                'alphabet': row['alphabet'],
                'threshold': t,
                'median_length': L,
                'max_mutations_tolerated': mut,
            })
    return pd.DataFrame(out_rows)


# ----------------------------------------------------------------------
# Plotters
# ----------------------------------------------------------------------

def plot_unique_sequence_retention(length_stats: pd.DataFrame, red_aa: pd.DataFrame,
                                    red_nt: pd.DataFrame, out_png: Path) -> None:
    """Plot A: per-function n_unique vs n_input, one panel per alphabet."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)

    for ax, (alpha, red, label) in zip(
        axes, [('aa', red_aa, 'aa (prot_seq)'), ('nt', red_nt, 'nt (cds_dna)')]
    ):
        rows: dict = {}
        for s in _SHORT_ORDER:
            r_len = length_stats[
                (length_stats['function_short'] == s) & (length_stats['alphabet'] == alpha)
            ]
            r_red = red[(red['function_short'] == s) & (red['threshold'] == 1.0)]
            if len(r_len) == 0 or len(r_red) == 0:
                rows[s] = (np.nan, np.nan)
                continue
            n_input = int(r_len['n_rows'].iloc[0])
            n_unique = int(r_red['n_sequences'].iloc[0])
            rows[s] = (n_input, n_unique)

        xs = np.arange(len(_SHORT_ORDER))
        width = 0.38
        offset = 0.21
        in_vals = np.array([rows[s][0] for s in _SHORT_ORDER], dtype=float)
        un_vals = np.array([rows[s][1] for s in _SHORT_ORDER], dtype=float)
        ax.bar(xs - offset, in_vals, width, color='#bdbdbd',
               edgecolor='black', linewidth=0.5, label='total seqs')
        ax.bar(xs + offset, un_vals, width,
               color=[_FUNCTION_COLORS[s] for s in _SHORT_ORDER],
               edgecolor='black', linewidth=0.5, label='uniq seqs')
        ax.set_xticks(xs)
        ax.set_xticklabels(_SHORT_ORDER, fontsize=9)
        ax.set_title(f'Unique sequence retention — {label}', fontsize=11)
        ax.set_ylabel('count')
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.set_axisbelow(True)
        for x, ui, un in zip(xs, in_vals, un_vals):
            if np.isfinite(ui):
                ax.annotate(f'{int(ui):,}', xy=(x - offset, ui),
                            xytext=(0, 2), textcoords='offset points',
                            ha='center', fontsize=7, color='#444')
            if np.isfinite(un):
                ax.annotate(f'{int(un):,}', xy=(x + offset, un),
                            xytext=(0, 2), textcoords='offset points',
                            ha='center', fontsize=7, color='#222')
        ax.legend(loc='center right', fontsize=8, frameon=False)

    fig.suptitle('Per-protein corpus redundancy (Flu A July 2025) — '
                 'grey = total seqs (1 per (isolate, protein)); '
                 'colored = uniq seqs after dedup.',
                 fontsize=10, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_cluster_counts_vs_threshold(red_aa: pd.DataFrame, red_nt: pd.DataFrame,
                                      out_png: Path) -> None:
    """Plot B: n_clusters vs threshold, log-Y, 8 functions × 2 alphabets."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for alpha, df, marker, ls in [('aa', red_aa, 'o', '-'),
                                  ('nt', red_nt, 's', '--')]:
        for s in _SHORT_ORDER:
            sub = df[df['function_short'] == s].sort_values('threshold')
            if len(sub) == 0:
                continue
            ax.plot(sub['threshold'], sub['n_clusters'],
                    marker=marker, linestyle=ls,
                    color=_FUNCTION_COLORS[s],
                    label=f'{s} {alpha}' if alpha == 'aa' else None,
                    markersize=4, linewidth=1.3, alpha=0.85)

    ax.set_xlabel('mmseqs2 identity threshold (`--min-seq-id`)')
    ax.set_ylabel('n_clusters (log scale)')
    ax.set_yscale('log')
    ax.invert_xaxis()  # higher threshold (1.0) on the left
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)

    # Two legends, both anchored OUTSIDE the axes on the right so they
    # never overlap the curves (which fan from top-left at id100 down to
    # bottom-right at id080).
    handles = [plt.Line2D([0], [0], color=_FUNCTION_COLORS[s], marker='o',
                          linestyle='-', label=s, markersize=5)
               for s in _SHORT_ORDER]
    leg1 = ax.legend(handles=handles, loc='upper left',
                     bbox_to_anchor=(1.02, 1.0), fontsize=8, ncol=1,
                     title='function', frameon=False)
    ax.add_artist(leg1)
    style_handles = [
        plt.Line2D([0], [0], color='#444', marker='o', linestyle='-',
                   label='aa (prot_seq)', markersize=5),
        plt.Line2D([0], [0], color='#444', marker='s', linestyle='--',
                   label='nt (cds_dna)', markersize=5),
    ]
    ax.legend(handles=style_handles, loc='upper left',
              bbox_to_anchor=(1.02, 0.35), fontsize=8,
              title='alphabet', frameon=False)

    ax.set_title('Cluster collapse vs identity threshold (per function, aa vs nt)',
                 fontsize=11)
    # Reserve right-side space for the external legends; bbox_inches='tight'
    # on savefig then includes them in the PNG.
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_bipartite_largest_pct(feasibilities: list[pd.DataFrame], out_png: Path) -> None:
    """Plot C: largest bipartite component % vs threshold, per (pair, alphabet)."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    pair_colors = {'ha_na': '#1f77b4', 'pb2_pb1': '#d62728'}
    alpha_style = {'aa': '-', 'nt': '--'}
    alpha_marker = {'aa': 'o', 'nt': 's'}

    for df in feasibilities:
        if df is None or len(df) == 0:
            continue
        pair = df['pair_short'].iloc[0]
        alpha = df['alphabet'].iloc[0]
        sub = df.sort_values('threshold')
        ax.plot(sub['threshold'], sub['largest_pct'],
                marker=alpha_marker[alpha], linestyle=alpha_style[alpha],
                color=pair_colors[pair],
                label=f"{'HA/NA' if pair == 'ha_na' else 'PB2/PB1'} {alpha}",
                markersize=6, linewidth=1.5)

    ax.axhline(80.0, color='#666', linestyle=':', linewidth=1,
               label='80% (single-bin ceiling)')
    ax.set_xlabel('mmseqs2 identity threshold (`--min-seq-id`)')
    ax.set_ylabel('Largest bipartite component (% of deduped pairs)')
    ax.set_ylim(0, 102)
    ax.invert_xaxis()
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='center right', fontsize=9, frameon=False)
    ax.set_title('Bipartite-component feasibility for 80/10/10 routing\n'
                 'A pair × alphabet × threshold is FEASIBLE when the largest '
                 'component is below 80% (and the second below 20%).',
                 fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--clusters_aa',
                   default=str(PROJ / 'data/processed/flu/July_2025/clusters_aa'),
                   help='Directory containing aa redundancy_stats.csv + per-(function, threshold) cluster parquets.')
    p.add_argument('--clusters_nt',
                   default=str(PROJ / 'data/processed/flu/July_2025/clusters_nt'),
                   help='Directory containing nt redundancy_stats.csv + per-(function, threshold) cluster parquets.')
    p.add_argument('--protein_final',
                   default=str(PROJ / 'data/processed/flu/July_2025/protein_final.csv'),
                   help='Stage 1 protein_final.csv (or .parquet) — used for aa length stats.')
    p.add_argument('--cds_final',
                   default=str(PROJ / 'data/processed/flu/July_2025/cds_final.parquet'),
                   help='Stage 1.5 cds_final.parquet — used for nt length stats.')
    p.add_argument('--feasibility_dir',
                   default=str(PROJ / 'results/flu/July_2025/runs/cluster_disjoint_feasibility'),
                   help='Directory containing feasibility_<pair>_<alphabet>.csv files '
                        '(emitted by cluster_disjoint_feasibility.py since the '
                        '2026-05-20 docs/results migration).')
    p.add_argument('--out_dir', default=str(PROJ / 'results/flu/July_2025/runs/cluster_analysis'),
                   help='Output directory for tables + plots.')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load inputs --------------------------------------------------
    red_aa = load_redundancy_stats(Path(args.clusters_aa), 'aa')
    red_nt = load_redundancy_stats(Path(args.clusters_nt), 'nt')
    length_stats = compute_length_stats(Path(args.protein_final), Path(args.cds_final))

    feasibilities: list[pd.DataFrame] = []
    for pair_label, pair_short, _, _ in _SCHEMA_PAIRS:
        for alpha in ('aa', 'nt'):
            f = load_feasibility(Path(args.feasibility_dir), pair_short, alpha)
            if f is None:
                print(f'WARNING: missing feasibility CSV for {pair_short} {alpha}; '
                      f'Plot C will skip this line.')
                continue
            feasibilities.append(f)

    # ---- summary table -----------------------------------------------
    keep_cols = ['function_short', 'alphabet', 'threshold', 'n_sequences',
                 'n_clusters', 'largest_cluster', 'p90_cluster_size',
                 'p99_cluster_size', 'median_cluster_size',
                 'fraction_singletons']
    summary = pd.concat([red_aa, red_nt], ignore_index=True)[keep_cols]
    summary = summary.sort_values(['function_short', 'alphabet', 'threshold'],
                                   ascending=[True, True, False]).reset_index(drop=True)
    summary_csv = out_dir / 'cluster_summary.csv'
    summary.to_csv(summary_csv, index=False)
    print(f'wrote {summary_csv} ({len(summary):,} rows)')

    # ---- length + mutations-tolerated tables -------------------------
    length_csv = out_dir / 'sequence_length_summary.csv'
    length_stats.to_csv(length_csv, index=False)
    print(f'wrote {length_csv} ({len(length_stats):,} rows)')

    thresholds_for_mutations = sorted(set(summary['threshold'].tolist()), reverse=True)
    mut_df = build_mutations_tolerated(length_stats, thresholds_for_mutations)
    mut_csv = out_dir / 'mutations_tolerated_table.csv'
    mut_df.to_csv(mut_csv, index=False)
    print(f'wrote {mut_csv} ({len(mut_df):,} rows)')

    # ---- plots --------------------------------------------------------
    plot_unique_sequence_retention(
        length_stats, red_aa, red_nt,
        out_dir / 'unique_sequence_retention.png',
    )
    print(f'wrote {out_dir / "unique_sequence_retention.png"}')

    plot_cluster_counts_vs_threshold(
        red_aa, red_nt,
        out_dir / 'cluster_counts_vs_threshold.png',
    )
    print(f'wrote {out_dir / "cluster_counts_vs_threshold.png"}')

    plot_bipartite_largest_pct(
        feasibilities,
        out_dir / 'bipartite_largest_pct_vs_threshold.png',
    )
    print(f'wrote {out_dir / "bipartite_largest_pct_vs_threshold.png"}')

    print('\nDone.')


if __name__ == '__main__':
    main()
