"""Cross-tab aa vs nt cluster memberships at a fixed identity threshold.

Joins aa-clustered isolates (`clusters_aa/tNN/<fn>_cluster.parquet`) with
nt_cds-clustered isolates (`clusters_nt_cds/tNN/<fn>_cluster.parquet`) via
`cds_final.parquet` and reports, per function:

- n_aa_clusters / n_nt_clusters / n_isolates
- distribution of distinct aa clusters per nt cluster (mean, median, max)
- distribution of distinct nt clusters per aa cluster (mean, median, max)
- count of "merger" nt clusters that contain >1 aa cluster
- count of "splitter" aa clusters that span >1 nt cluster

Use this to investigate the open methodology question: under symmetric
linclust at id099, why does nt have FEWER clusters than aa on most
functions (opposite of the conventional "synonymous codons → more
nt diversity" expectation)?

Linkage:
    aa_cluster.seq_hash  = md5(prot_seq)  = cds_final.seq_hash
    nt_cluster.seq_hash  = md5(cds_dna)   = cds_final.cds_dna_hash
    cds_final.assembly_id is the per-isolate row.

Usage:
    python src/analysis/aa_nt_cluster_crosstab.py \
        --threshold 0.99 \
        --functions HA NA PB2 PB1 PA NP M1 NS1 \
        --data_root data/processed/flu/July_2025 \
        --out_dir results/flu/July_2025/runs/cluster_aa_nt_crosstab
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FUNCTION_TO_SHORT = {
    'RNA-dependent RNA polymerase PB2 subunit': 'PB2',
    'RNA-dependent RNA polymerase catalytic core PB1 subunit': 'PB1',
    'RNA-dependent RNA polymerase PA subunit': 'PA',
    'Hemagglutinin precursor': 'HA',
    'Nucleocapsid protein': 'NP',
    'Neuraminidase protein': 'NA',
    'Matrix protein 1': 'M1',
    'Non-structural protein 1, interferon antagonist and host mRNA processing inhibitor': 'NS1',
}
SHORT_TO_FUNCTION = {v: k for k, v in FUNCTION_TO_SHORT.items()}
SHORT_TO_SEGMENT = {'PB2': 1, 'PB1': 2, 'PA': 3, 'HA': 4, 'NP': 5, 'NA': 6, 'M1': 7, 'NS1': 8}
SHORT_ORDER = ['PB2', 'PB1', 'PA', 'HA', 'NP', 'NA', 'M1', 'NS1']


def _threshold_label(threshold: float) -> str:
    return f't{int(round(threshold * 100)):03d}'


def load_joined(data_root: Path, function_short: str, threshold: float) -> pd.DataFrame:
    """Return per-isolate (assembly_id, aa_cluster_id, nt_cluster_id) table."""
    tag = _threshold_label(threshold)
    aa_path = data_root / 'clusters_aa' / tag / f'{function_short}_cluster.parquet'
    nt_path = data_root / 'clusters_nt' / tag / f'{function_short}_cluster.parquet'
    cds_path = data_root / 'cds_final.parquet'

    aa = pd.read_parquet(aa_path)[['seq_hash', 'cluster_id']].rename(
        columns={'seq_hash': 'prot_hash', 'cluster_id': 'aa_cluster_id'}
    )
    nt = pd.read_parquet(nt_path)[['seq_hash', 'cluster_id']].rename(
        columns={'seq_hash': 'cds_dna_hash', 'cluster_id': 'nt_cluster_id'}
    )
    fn_full = SHORT_TO_FUNCTION[function_short]
    cds = pd.read_parquet(cds_path, columns=['assembly_id', 'function', 'seq_hash', 'cds_dna_hash'])
    cds = cds.loc[cds['function'] == fn_full, ['assembly_id', 'seq_hash', 'cds_dna_hash']]
    cds = cds.rename(columns={'seq_hash': 'prot_hash'})

    merged = cds.merge(aa, on='prot_hash', how='inner').merge(
        nt, on='cds_dna_hash', how='inner'
    )
    return merged


def compute_function_summary(joined: pd.DataFrame) -> dict:
    n_isolates = len(joined)
    n_aa = joined['aa_cluster_id'].nunique()
    n_nt = joined['nt_cluster_id'].nunique()

    # distinct aa clusters per nt cluster
    aa_per_nt = joined.groupby('nt_cluster_id')['aa_cluster_id'].nunique()
    # distinct nt clusters per aa cluster
    nt_per_aa = joined.groupby('aa_cluster_id')['nt_cluster_id'].nunique()

    return {
        'n_isolates': n_isolates,
        'n_aa_clusters': n_aa,
        'n_nt_clusters': n_nt,
        'nt_per_aa_ratio': n_nt / n_aa if n_aa else float('nan'),
        # nt mergers: nt clusters absorbing multiple distinct aa clusters
        'nt_clusters_with_multi_aa': int((aa_per_nt > 1).sum()),
        'nt_clusters_with_multi_aa_pct': float((aa_per_nt > 1).mean() * 100),
        'max_aa_per_nt': int(aa_per_nt.max()),
        'mean_aa_per_nt': float(aa_per_nt.mean()),
        'median_aa_per_nt': float(aa_per_nt.median()),
        # aa splitters: aa clusters spread across multiple nt clusters
        'aa_clusters_with_multi_nt': int((nt_per_aa > 1).sum()),
        'aa_clusters_with_multi_nt_pct': float((nt_per_aa > 1).mean() * 100),
        'max_nt_per_aa': int(nt_per_aa.max()),
        'mean_nt_per_aa': float(nt_per_aa.mean()),
        'median_nt_per_aa': float(nt_per_aa.median()),
        '_aa_per_nt_series': aa_per_nt,
        '_nt_per_aa_series': nt_per_aa,
    }


def plot_histograms(summaries: dict, threshold: float, out_dir: Path) -> None:
    """One PNG per function showing aa-per-nt + nt-per-aa side-by-side."""
    for short, summary in summaries.items():
        aa_per_nt = summary['_aa_per_nt_series']
        nt_per_aa = summary['_nt_per_aa_series']

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        # aa-per-nt (nt as merger)
        max_aa = int(aa_per_nt.max())
        bins_aa = np.arange(0.5, min(max_aa, 30) + 1.5)
        axes[0].hist(aa_per_nt.clip(upper=30), bins=bins_aa, color='#4477AA', edgecolor='black')
        axes[0].set_yscale('log')
        axes[0].set_xlabel('distinct aa clusters per nt cluster')
        axes[0].set_ylabel('# nt clusters (log)')
        axes[0].set_title(
            f'{short} {_threshold_label(threshold)}: '
            f'{summary["nt_clusters_with_multi_aa"]:,} / '
            f'{summary["n_nt_clusters"]:,} nt clusters absorb >1 aa cluster '
            f'(max={max_aa})'
        )

        # nt-per-aa (aa as splitter)
        max_nt = int(nt_per_aa.max())
        bins_nt = np.arange(0.5, min(max_nt, 30) + 1.5)
        axes[1].hist(nt_per_aa.clip(upper=30), bins=bins_nt, color='#EE6677', edgecolor='black')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('distinct nt clusters per aa cluster')
        axes[1].set_ylabel('# aa clusters (log)')
        axes[1].set_title(
            f'{short} {_threshold_label(threshold)}: '
            f'{summary["aa_clusters_with_multi_nt"]:,} / '
            f'{summary["n_aa_clusters"]:,} aa clusters span >1 nt cluster '
            f'(max={max_nt})'
        )

        fig.suptitle(
            f'{short} — aa vs nt cluster membership cross-tab '
            f'(linclust {_threshold_label(threshold)})', y=1.02
        )
        fig.tight_layout()
        out_path = out_dir / f'{short}_{_threshold_label(threshold)}_aa_nt_histograms.png'
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  wrote {out_path}')


def write_top_mergers(joined: pd.DataFrame, short: str, threshold: float, out_dir: Path, k: int = 20) -> None:
    """Top-k nt clusters by distinct aa clusters absorbed."""
    grouped = joined.groupby('nt_cluster_id').agg(
        n_isolates=('assembly_id', 'count'),
        n_aa_clusters=('aa_cluster_id', 'nunique'),
        n_unique_prot=('prot_hash', 'nunique'),
        n_unique_cds=('cds_dna_hash', 'nunique'),
    ).reset_index()
    grouped = grouped.sort_values(['n_aa_clusters', 'n_isolates'], ascending=False).head(k)
    out_path = out_dir / f'{short}_{_threshold_label(threshold)}_top_nt_mergers.csv'
    grouped.to_csv(out_path, index=False)
    print(f'  wrote {out_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--threshold', type=float, default=0.99)
    ap.add_argument('--functions', nargs='+', default=SHORT_ORDER,
                    help='function_short list; default: 8 core proteins with CDS.')
    ap.add_argument('--data_root', type=Path, default=Path('data/processed/flu/July_2025'))
    ap.add_argument('--out_dir', type=Path, default=Path('results/flu/July_2025/runs/cluster_aa_nt_crosstab'))
    ap.add_argument('--top_k_mergers', type=int, default=20)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = _threshold_label(args.threshold)

    rows = []
    summaries = {}
    for short in args.functions:
        if short not in SHORT_TO_FUNCTION:
            print(f'WARNING: unknown function_short={short}; skipping')
            continue
        print(f'[{short}] loading + joining at {tag} ...')
        joined = load_joined(args.data_root, short, args.threshold)
        if joined.empty:
            print(f'WARNING: empty join for {short}; skipping')
            continue
        summary = compute_function_summary(joined)
        summaries[short] = summary

        rows.append({
            'segment': SHORT_TO_SEGMENT[short],
            'function_short': short,
            'n_isolates': summary['n_isolates'],
            'n_aa_clusters': summary['n_aa_clusters'],
            'n_nt_clusters': summary['n_nt_clusters'],
            'nt/aa_ratio': summary['nt_per_aa_ratio'],
            'nt_with_multi_aa': summary['nt_clusters_with_multi_aa'],
            'nt_with_multi_aa_pct': summary['nt_clusters_with_multi_aa_pct'],
            'max_aa_per_nt': summary['max_aa_per_nt'],
            'mean_aa_per_nt': summary['mean_aa_per_nt'],
            'median_aa_per_nt': summary['median_aa_per_nt'],
            'aa_with_multi_nt': summary['aa_clusters_with_multi_nt'],
            'aa_with_multi_nt_pct': summary['aa_clusters_with_multi_nt_pct'],
            'max_nt_per_aa': summary['max_nt_per_aa'],
            'mean_nt_per_aa': summary['mean_nt_per_aa'],
            'median_nt_per_aa': summary['median_nt_per_aa'],
        })
        write_top_mergers(joined, short, args.threshold, args.out_dir, k=args.top_k_mergers)

    if not rows:
        print('ERROR: no functions processed; nothing to summarize')
        sys.exit(1)

    summary_df = pd.DataFrame(rows).sort_values('segment').reset_index(drop=True)
    summary_path = args.out_dir / f'crosstab_summary_{tag}.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f'\nwrote {summary_path}')
    print(summary_df.to_string(index=False))

    plot_histograms(summaries, args.threshold, args.out_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
