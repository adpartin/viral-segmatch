"""Per-protein sequence-abundance plots: rank-abundance, cumulative coverage, Lorenz, duplicate-count bins.

Four lenses on the same skewed-abundance question: how concentrated is the
corpus on a few common sequences?

  - plot_rank_abundance:      Whittaker curve — shape of the decline (log y).
  - plot_cumulative_coverage: actionable "top K% covers X% of records".
  - plot_lorenz:              standardized inequality (curve + per-line Gini).
  - plot_duplicate_bins:      mass distribution across the multiplicity axis
                              (inventory and records views, side-by-side).

The plot functions take a `{protein_short: counts_array}` dict and a
free-form `title`, so they are source-agnostic. The same functions render
raw per-unique-sequence corpus copy counts AND per-cluster member counts
at any threshold; the caller assembles the dict and owns the title text.

Two thin drivers are expected: this file ships the raw driver (reads
cds_final.parquet); a cluster driver (reads cluster parquets at one
threshold) is planned but not in this commit.

CLI (raw driver):
    python -m src.analysis.plot_seq_abundance \\
        [--cds_final data/processed/flu/July_2025/cds_final.parquet] \\
        [--out_dir   results/flu/July_2025/runs/seq_abundance/raw]

Outputs (under --out_dir):
    rank_abundance_linear_{aa,nt_cds}.png
    rank_abundance_log_{aa,nt_cds}.png
    cumulative_coverage_{aa,nt_cds}.png
    lorenz_{aa,nt_cds}.png
    duplicate_bins_{aa,nt_cds}.png
    duplicate_bins.csv                (combined aa + nt_cds)
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

from src.utils.plot_config import get_protein_color


# Function full → short (matches conf/virus/flu.yaml::function_short_names).
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

# Duplicate-count bin edges, half-open [e_i, e_{i+1}) for np.histogram /
# pd.cut(right=False). Labels follow the spec verbatim. The 2001 → inf
# edge encodes the open-ended "2001+" bin.
_DEFAULT_BIN_EDGES = [1, 2, 3, 6, 11, 51, 101, 501, 1001, 2001, np.inf]
_DEFAULT_BIN_LABELS = ['1', '2', '3-5', '6-10', '11-50', '51-100',
                       '101-500', '501-1000', '1001-2000', '2001+']


def _gini(values: np.ndarray) -> float:
    """Gini coefficient on a non-negative array. 0 = perfect evenness, 1 = max inequality."""
    if len(values) == 0:
        return 0.0
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    s = v.sum()
    if s == 0:
        return 0.0
    return (2.0 * np.sum(np.arange(1, n + 1) * v)) / (n * s) - (n + 1) / n


# ----------------------------------------------------------------------
# Plotters — source-agnostic. counts_by_protein values are arrays of
# per-entity record counts (raw: per-unique-seq corpus frequency;
# cluster: per-cluster member count, in either inventory or records
# units depending on what the caller passed). The caller owns title text.
# ----------------------------------------------------------------------

def plot_rank_abundance(
    counts_by_protein: dict[str, np.ndarray],
    title: str,
    out_png: Path,
    x_scale: str = 'linear',
) -> None:
    """Whittaker rank-abundance: x = rank, y = count (log), one line per protein.

    Each line spans rank 1..n_uniq for that protein. Lines of different
    lengths simply stop at their own n_uniq.

    Args:
        x_scale: 'linear' (default; matches the classic Whittaker layout)
                 or 'log' (compresses the long tail and exposes the
                 high-rank structure that linear-x squashes).
    """
    if x_scale not in ('linear', 'log'):
        raise ValueError(f"x_scale must be 'linear' or 'log', got {x_scale!r}")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for s in _SHORT_ORDER:
        if s not in counts_by_protein:
            continue
        counts = np.sort(counts_by_protein[s])[::-1]
        if len(counts) == 0:
            continue
        ranks = np.arange(1, len(counts) + 1)
        ax.plot(ranks, counts, color=get_protein_color(s), label=s,
                linewidth=1.4, alpha=0.85)
    ax.set_xlabel('rank of unique entity (most common first)')
    ax.set_ylabel('count (records per entity, log)')
    ax.set_yscale('log')
    if x_scale == 'log':
        ax.set_xscale('log')
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=9, frameon=False, ncol=2)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_cumulative_coverage(
    counts_by_protein: dict[str, np.ndarray],
    title: str,
    out_png: Path,
    x_mode: str = 'top_pct',
) -> None:
    """Cumulative coverage: x = top K (or top %) of entities, y = cum % of records.

    Args:
        x_mode: 'top_pct' (x = 0..100 % of entities; all curves share x-axis)
                or 'top_k' (x = 1..n_uniq, log scale; per-protein x ranges).
    """
    if x_mode not in ('top_pct', 'top_k'):
        raise ValueError(f"x_mode must be 'top_pct' or 'top_k', got {x_mode!r}")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    xlabel = 'top % of unique entities' if x_mode == 'top_pct' else 'top K unique entities (rank)'
    for s in _SHORT_ORDER:
        if s not in counts_by_protein:
            continue
        counts = np.sort(counts_by_protein[s])[::-1]
        n = len(counts)
        total = counts.sum()
        if n == 0 or total == 0:
            continue
        cum = np.cumsum(counts) / total * 100.0
        if x_mode == 'top_pct':
            xs = np.arange(1, n + 1) / n * 100.0
        else:
            xs = np.arange(1, n + 1)
        ax.plot(xs, cum, color=get_protein_color(s), label=s,
                linewidth=1.4, alpha=0.85)
    if x_mode == 'top_k':
        ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('cumulative % of records')
    ax.set_ylim(0, 102)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', fontsize=9, frameon=False, ncol=2)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_lorenz(
    counts_by_protein: dict[str, np.ndarray],
    title: str,
    out_png: Path,
) -> None:
    """Lorenz curve: x = cum % of entities (sorted ascending), y = cum % of records.

    The diagonal marks perfect evenness; curves bow toward the bottom-right
    in proportion to inequality. Each protein's Gini appears in its legend
    label so the curve and the scalar can be read together.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot([0, 100], [0, 100], color='#888', linestyle='--',
            linewidth=1, label='Perfect evenness')
    for s in _SHORT_ORDER:
        if s not in counts_by_protein:
            continue
        counts = np.sort(counts_by_protein[s])
        n = len(counts)
        total = counts.sum()
        if n == 0 or total == 0:
            continue
        # Prepend 0 so the Lorenz curve passes through the origin.
        cum_records = np.concatenate(([0.0], np.cumsum(counts) / total)) * 100.0
        xs = np.arange(0, n + 1) / n * 100.0
        gini = _gini(counts)
        ax.plot(xs, cum_records, color=get_protein_color(s),
                label=f'{s}  (Gini={gini:.3f})', linewidth=1.4, alpha=0.85)
    ax.set_xlabel('cumulative % of unique entities (sorted ascending)')
    ax.set_ylabel('cumulative % of records')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=1)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_duplicate_bins(
    counts_by_protein: dict[str, np.ndarray],
    title: str,
    out_png: Path,
    out_csv: Optional[Path] = None,
    csv_extra: Optional[dict] = None,
    bin_edges: list = _DEFAULT_BIN_EDGES,
    bin_labels: list = _DEFAULT_BIN_LABELS,
) -> pd.DataFrame:
    """2x4 grid (one panel per protein), grouped bars: inventory + records.

    Inventory: % of unique entities falling in each copy-count bin (sums to
               100 per protein).
    Records:   % of records contributed by entities in each bin (sums to
               100 per protein).

    Args:
        out_csv:   If provided, write the per-(protein, bin) counts and
                   percentages as a CSV. Returned DataFrame is the same
                   content regardless.
        csv_extra: Extra columns merged into every CSV row (e.g.,
                   `{'alphabet': 'aa'}`). Lets the caller stamp source
                   context without altering the plot fn's data shape.
    """
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
    xs = np.arange(len(bin_labels))
    width = 0.40
    csv_rows = []
    for ax, s in zip(axes.flat, _SHORT_ORDER):
        if s not in counts_by_protein:
            ax.set_visible(False)
            continue
        counts = counts_by_protein[s]
        n_uniq = len(counts)
        total_records = counts.sum() if n_uniq > 0 else 0
        if n_uniq == 0 or total_records == 0:
            ax.set_visible(False)
            continue
        inv_counts, _ = np.histogram(counts, bins=bin_edges)
        rec_counts = np.zeros(len(bin_labels), dtype=int)
        for i in range(len(bin_labels)):
            in_bin = (counts >= bin_edges[i]) & (counts < bin_edges[i + 1])
            rec_counts[i] = int(counts[in_bin].sum())
        inv_pct = inv_counts / n_uniq * 100.0
        rec_pct = rec_counts / total_records * 100.0
        color = get_protein_color(s)
        ax.bar(xs - width / 2, inv_pct, width, color=color, alpha=0.85,
               edgecolor='black', linewidth=0.4, label='inventory')
        ax.bar(xs + width / 2, rec_pct, width, color=color, alpha=0.35,
               hatch='///', edgecolor='black', linewidth=0.4, label='records')
        ax.set_xticks(xs)
        ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=7)
        ax.set_title(f'{s}  (n_uniq={n_uniq:,}, n_rec={int(total_records):,})',
                     fontsize=9)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.set_axisbelow(True)
        for i in range(len(bin_labels)):
            row = {
                'protein': s,
                'bin': bin_labels[i],
                'inventory_count': int(inv_counts[i]),
                'inventory_pct': round(float(inv_pct[i]), 4),
                'records_count': int(rec_counts[i]),
                'records_pct': round(float(rec_pct[i]), 4),
            }
            if csv_extra:
                row = {**csv_extra, **row}
            csv_rows.append(row)
    for ax in axes[1, :]:
        ax.set_xlabel('copy-count bin', fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel('% per protein', fontsize=8)
    # Legend on the first visible panel only.
    for ax in axes.flat:
        if ax.get_visible():
            ax.legend(loc='upper right', fontsize=7, frameon=False)
            break
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)

    df = pd.DataFrame(csv_rows)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


# ----------------------------------------------------------------------
# Raw driver: corpus copy-counts from cds_final.parquet
# ----------------------------------------------------------------------

def raw_counts_by_protein(cds_final: Path, alphabet: str) -> dict[str, np.ndarray]:
    """Returns {protein_short: per-unique-seq corpus copy-count array}.

    Each array entry = number of isolate records carrying one unique
    sequence (i.e., the level-0 multiplicity). Source: cds_final.parquet
    (seq_hash for aa, cds_dna_hash for nt_cds).
    """
    if alphabet == 'aa':
        df = pd.read_parquet(cds_final, columns=['function', 'seq_hash'])
        hash_col = 'seq_hash'
    elif alphabet == 'nt_cds':
        df = pd.read_parquet(cds_final, columns=['function', 'cds_dna_hash'])
        hash_col = 'cds_dna_hash'
    else:
        raise ValueError(f"alphabet must be 'aa' or 'nt_cds', got {alphabet!r}")
    df['function_short'] = df['function'].map(_FUNCTION_TO_SHORT)
    df = df.dropna(subset=['function_short'])
    return {
        s: df.loc[df['function_short'] == s].groupby(hash_col).size().values
        for s in _SHORT_ORDER
        if (df['function_short'] == s).any()
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cds_final',
                   default=str(PROJ / 'data/processed/flu/July_2025/cds_final.parquet'),
                   help='Stage 1.5 cds_final.parquet — provides seq_hash and cds_dna_hash.')
    p.add_argument('--out_dir',
                   default=str(PROJ / 'results/flu/July_2025/runs/seq_abundance/raw'),
                   help='Output directory for plots and CSV.')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_csv_frames: list[pd.DataFrame] = []
    for alphabet in ('aa', 'nt_cds'):
        suffix = 'prot_seq' if alphabet == 'aa' else 'cds_dna'
        counts = raw_counts_by_protein(Path(args.cds_final), alphabet)

        for x_scale in ('linear', 'log'):
            rank_png = out_dir / f'rank_abundance_{x_scale}_{alphabet}.png'
            plot_rank_abundance(
                counts,
                title=f'Rank-abundance (raw, unique sequences) — {alphabet} ({suffix}), x={x_scale}',
                out_png=rank_png,
                x_scale=x_scale,
            )
            print(f'wrote {rank_png}')

        cum_png = out_dir / f'cumulative_coverage_{alphabet}.png'
        plot_cumulative_coverage(
            counts,
            title=f'Cumulative coverage (raw, unique sequences) — {alphabet} ({suffix})',
            out_png=cum_png,
        )
        print(f'wrote {cum_png}')

        lor_png = out_dir / f'lorenz_{alphabet}.png'
        plot_lorenz(
            counts,
            title=f'Lorenz curve (raw, unique sequences) — {alphabet} ({suffix})',
            out_png=lor_png,
        )
        print(f'wrote {lor_png}')

        dup_png = out_dir / f'duplicate_bins_{alphabet}.png'
        dup_df = plot_duplicate_bins(
            counts,
            title=f'Duplicate-count bins (raw, unique sequences) — {alphabet} ({suffix})',
            out_png=dup_png,
            csv_extra={'alphabet': alphabet},
        )
        print(f'wrote {dup_png}')
        bin_csv_frames.append(dup_df)

    bin_csv = out_dir / 'duplicate_bins.csv'
    pd.concat(bin_csv_frames, ignore_index=True).to_csv(bin_csv, index=False)
    print(f'wrote {bin_csv}')

    print('\nDone.')


if __name__ == '__main__':
    main()
