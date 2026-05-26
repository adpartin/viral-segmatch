"""Six-panel 2-D embedding plots for an idXX sweep: see train ↔ test
separation grow as the cluster-identity threshold tightens.

Produces two figures, both 6-panel grids (one panel per idXX):

  - `idxx_sweep_pair_svd.png`  — pair k-mer features projected to 2-D
    via TruncatedSVD-2 (sparse-friendly; the canonical reduction for
    non-negative count features per `dim_reduction_utils.py`).
  - `idxx_sweep_pair_umap.png` — UMAP-2 on top of a TruncatedSVD-50
    pre-step (matches `plot_kmer_routing_geometry.py`'s pair geometry).

In each figure, **one global projection** is fit on the union of all
sampled pairs across the sweep, then applied to all 6 panels. This way
the same point lands in the same XY across panels — what changes
between panels is which color (split) it gets, because the routing
re-assigns it.

Visual reading: at id100 the partition is essentially random; train and
test points should overlap heavily. As idXX drops, the cluster_disjoint
constraint pulls them apart, and the train/test scatters separate.

Designed for the PB2-PB1 PB2-only sweep on aa k=3 features but accepts
any idXX sweep produced by `mmd_sweep.sh` + Stage 3.

CLI (PB2-PB1 PB2-only example):
    python -m src.analysis.plot_idxx_sweep_geometry \\
        --dataset_pattern 'dataset_flu_pb2_pb1_cluster_aa_id{thr}_PB2only_*' \\
        --thresholds 100 099 098 097 096 095 \\
        --kmer_dir data/embeddings/flu/July_2025 \\
        --alphabet aa --kmer_k 3 \\
        --slot_a_display PB2 --slot_b_display PB1 \\
        --routing_label 'PB2-PB1 PB2-only (aa)' \\
        --out_dir results/flu/July_2025/runs/split_separation_mmd/sweep_aggregate/pb2_pb1_PB2only

Defaults reproduce the HA-NA HA-only sweep when run with no overrides.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.dim_reduction_utils import (
    UMAP_AVAILABLE,
    compute_truncated_svd_reduction,
    compute_umap_reduction,
)
from src.utils.kmer_utils import load_kmer_index, load_kmer_matrix

# Train / test display style. Val is omitted from these plots for visual
# clarity (the train-vs-test contrast is what carries the leakage story).
SPLIT_COLORS = {'train': '#1f77b4', 'test': '#d62728'}
SPLIT_MARKERS = {'train': 'o', 'test': 's'}
SPLITS_DRAWN = ('train', 'test')


def _occ_col(alphabet: str, side: str) -> str:
    suf = side.lower()
    if alphabet == 'nt':
        return f'ctg_{suf}'
    if alphabet == 'aa':
        return f'brc_{suf}'
    raise ValueError(f"alphabet must be 'nt' or 'aa'; got {alphabet!r}")


def _normalize_occurrence(series: pd.Series, alphabet: str) -> pd.Series:
    s = series.astype(str)
    if alphabet != 'nt':
        return s
    return s.apply(lambda x: str(float(x)) if x.replace('.', '', 1).isdigit() else x)


def _resolve_dataset_dir(dataset_root: Path, pattern: str, thr: str) -> Path:
    p = dataset_root / pattern.replace('{thr}', thr)
    matches = sorted(dataset_root.glob(pattern.replace('{thr}', thr)))
    if not matches:
        raise FileNotFoundError(f"No dataset matching {p}")
    return matches[0]


def _load_dataset_pairs(run_dir: Path, idxx: int, splits=SPLITS_DRAWN
                         ) -> pd.DataFrame:
    frames = []
    for split in splits:
        path = run_dir / f"{split}_pairs.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False,
                         keep_default_na=False, na_values=[''])
        df['split'] = split
        df['idxx'] = idxx
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No {'/'.join(splits)} pair CSVs under {run_dir}")
    return pd.concat(frames, ignore_index=True)


def _stratified_sample(pairs: pd.DataFrame, max_per_cell: int,
                        random_state: int) -> pd.DataFrame:
    """Cap rows per (idxx, split, label) cell. Preserves label balance."""
    out = []
    seed = random_state
    for _key, grp in pairs.groupby(['idxx', 'split', 'label'], sort=False):
        if len(grp) <= max_per_cell:
            out.append(grp)
        else:
            out.append(grp.sample(n=max_per_cell, random_state=seed))
            seed += 1
    return pd.concat(out, ignore_index=True).reset_index(drop=True)


def _lookup_kmer_rows(pairs: pd.DataFrame, side: str, key_to_row: dict,
                       alphabet: str) -> tuple:
    asm = pairs[f'assembly_id_{side}'].astype(str)
    occ = _normalize_occurrence(pairs[_occ_col(alphabet, side)], alphabet)
    keys = list(zip(asm, occ))
    rows = np.array([key_to_row.get(k, -1) for k in keys], dtype=np.int64)
    valid = rows >= 0
    return rows, valid


def _build_pair_features(pairs: pd.DataFrame, kmer_matrix: sp.csr_matrix,
                          key_to_row: dict, alphabet: str) -> tuple:
    """Returns (features as csr_matrix, kept_pairs_df). Drops pairs whose
    k-mer rows are missing in the cache (rare; usually 0)."""
    rows_a, valid_a = _lookup_kmer_rows(pairs, 'a', key_to_row, alphabet)
    rows_b, valid_b = _lookup_kmer_rows(pairs, 'b', key_to_row, alphabet)
    valid = valid_a & valid_b
    n_in = len(pairs)
    pairs = pairs.loc[valid].reset_index(drop=True)
    rows_a = rows_a[valid]
    rows_b = rows_b[valid]
    dropped = n_in - len(pairs)
    if dropped:
        print(f"  dropped {dropped:,}/{n_in:,} pairs (missing k-mer rows)")
    # Concatenate sparse k-mer rows: feature dim = 2 * kmer_matrix.shape[1].
    feats = sp.hstack([kmer_matrix[rows_a], kmer_matrix[rows_b]], format='csr')
    return feats, pairs


def _draw_panel(ax, coords: np.ndarray, pairs: pd.DataFrame, idxx: int,
                 xlab: str, ylab: str,
                 slot_a_display: str, slot_b_display: str,
                 show_legend: bool = False) -> None:
    """Single-panel scatter for one idXX. Train below test in z-order so
    test points are visible against the (denser) train cloud."""
    sub = pairs.copy()
    sub['x'] = coords[:, 0]
    sub['y'] = coords[:, 1]
    # Draw train first so test sits on top.
    for split in SPLITS_DRAWN:
        cell = sub[sub['split'] == split]
        if cell.empty:
            continue
        ax.scatter(cell['x'], cell['y'],
                   c=SPLIT_COLORS[split], marker=SPLIT_MARKERS[split],
                   s=8, alpha=0.45, linewidth=0.0,
                   label=f"{split} (n={len(cell):,})" if show_legend else None)
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(f"id{idxx:03d}", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)


def _draw_6panel(coords_all: np.ndarray, pairs_all: pd.DataFrame,
                  xlab: str, ylab: str, suptitle: str,
                  slot_a_display: str, slot_b_display: str,
                  out_path: Path) -> None:
    """6-panel figure (2 rows × 3 cols), one panel per idXX. Shared
    XY range across panels to make the visual comparison fair."""
    idxx_sorted = sorted(pairs_all['idxx'].unique(), reverse=True)
    if len(idxx_sorted) != 6:
        print(f"  WARNING: expected 6 thresholds, got {len(idxx_sorted)}")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    pairs_all = pairs_all.reset_index(drop=True)
    pairs_all['_x'] = coords_all[:, 0]
    pairs_all['_y'] = coords_all[:, 1]
    xmin, xmax = pairs_all['_x'].min(), pairs_all['_x'].max()
    ymin, ymax = pairs_all['_y'].min(), pairs_all['_y'].max()
    xpad = 0.05 * (xmax - xmin)
    ypad = 0.05 * (ymax - ymin)
    for i, idxx in enumerate(idxx_sorted):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        sub = pairs_all[pairs_all['idxx'] == idxx]
        coords_sub = sub[['_x', '_y']].to_numpy()
        # show_legend only on first panel.
        _draw_panel(ax, coords_sub, sub, idxx, xlab, ylab,
                    slot_a_display, slot_b_display,
                    show_legend=(i == 0))
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)
    # If fewer than 6 panels, blank the rest.
    for j in range(len(idxx_sorted), len(axes_flat)):
        axes_flat[j].axis('off')

    # Single legend on first panel.
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        axes_flat[0].legend(handles=handles, labels=labels,
                            loc='best', fontsize=9, framealpha=0.9)

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dataset_root', type=Path,
                    default=Path('data/datasets/flu/July_2025/runs'))
    ap.add_argument('--dataset_pattern', type=str,
                    default='dataset_flu_ha_na_cluster_aa_id{thr}_HAonly_*')
    ap.add_argument('--thresholds', type=str, nargs='+',
                    default=['100', '099', '098', '097', '096', '095'])
    ap.add_argument('--kmer_dir', type=Path,
                    default=Path('data/embeddings/flu/July_2025'))
    ap.add_argument('--alphabet', choices=['aa', 'nt'], default='aa')
    ap.add_argument('--kmer_k', type=int, default=3)
    ap.add_argument('--max_per_cell', type=int, default=400,
                    help='Cap per (idxx, split, label). 6 idxx × 2 splits × '
                         '2 labels × 400 = up to 9,600 points total.')
    ap.add_argument('--umap_pre_pca_dim', type=int, default=50)
    ap.add_argument('--slot_a_display', type=str, default='HA')
    ap.add_argument('--slot_b_display', type=str, default='NA')
    ap.add_argument('--routing_label', type=str,
                    default='HA-NA HA-only (aa)',
                    help='Used in plot suptitles only.')
    ap.add_argument('--out_dir', type=Path,
                    default=Path('results/flu/July_2025/runs/split_separation_mmd'
                                 '/sweep_aggregate'))
    ap.add_argument('--random_state', type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading k-mer cache ({args.alphabet} k={args.kmer_k}) ...")
    kmer_matrix = load_kmer_matrix(args.kmer_dir, k=args.kmer_k,
                                    alphabet=args.alphabet)
    key_to_row = load_kmer_index(args.kmer_dir, k=args.kmer_k,
                                  alphabet=args.alphabet)
    print(f"  matrix shape: {kmer_matrix.shape}  |  index entries: {len(key_to_row):,}")

    print("Loading pair CSVs and sampling per (idxx, split, label) ...")
    frames = []
    for thr in args.thresholds:
        run_dir = _resolve_dataset_dir(args.dataset_root,
                                        args.dataset_pattern, thr)
        idxx = int(thr)
        pairs = _load_dataset_pairs(run_dir, idxx)
        print(f"  id{idxx:03d}: {len(pairs):,} pairs from {run_dir.name}")
        frames.append(pairs)
    pairs_all = pd.concat(frames, ignore_index=True)
    pairs_all['label'] = pd.to_numeric(pairs_all['label'], errors='coerce')
    pairs_all = pairs_all[pairs_all['label'].isin([0, 1])].reset_index(drop=True)

    pairs_sampled = _stratified_sample(pairs_all, args.max_per_cell,
                                        args.random_state)
    print(f"  sampled {len(pairs_sampled):,} pairs total "
          f"({args.max_per_cell} per (idxx, split, label) cell)")

    print("Building pair k-mer features (concat slot a + slot b) ...")
    feats, pairs_kept = _build_pair_features(pairs_sampled, kmer_matrix,
                                              key_to_row, args.alphabet)
    print(f"  features shape: {feats.shape}  ({feats.nnz:,} nnz)")

    # ----- TruncatedSVD-2 (single global projection) -----
    print("Fitting TruncatedSVD-2 on union of all panels ...")
    svd_2d, _ = compute_truncated_svd_reduction(
        feats, n_components=2, algorithm='randomized',
        random_state=args.random_state,
    )
    svd_suptitle = (f'Pair TruncatedSVD-2 (concat) — {args.routing_label}\n'
                    f'one global projection across the sweep; '
                    f'color = train/test under each idXX')
    _draw_6panel(svd_2d, pairs_kept,
                 xlab='SVD 1', ylab='SVD 2',
                 suptitle=svd_suptitle,
                 slot_a_display=args.slot_a_display,
                 slot_b_display=args.slot_b_display,
                 out_path=args.out_dir / 'idxx_sweep_pair_svd.png')

    # ----- UMAP (TruncatedSVD-K pre-step → UMAP-2) -----
    if UMAP_AVAILABLE:
        pre_dim = min(args.umap_pre_pca_dim,
                       feats.shape[0] - 1, feats.shape[1])
        print(f"Fitting TruncatedSVD-{pre_dim} pre-step + UMAP-2 ...")
        pre_svd, _ = compute_truncated_svd_reduction(
            feats, n_components=pre_dim, algorithm='randomized',
            random_state=args.random_state,
        )
        umap_2d, _ = compute_umap_reduction(
            pre_svd, n_components=2, random_state=args.random_state,
        )
        umap_suptitle = (f'Pair UMAP (TruncatedSVD-{pre_dim} pre-step) '
                         f'— {args.routing_label}\n'
                         f'one global projection across the sweep; '
                         f'color = train/test under each idXX')
        _draw_6panel(umap_2d, pairs_kept,
                     xlab='UMAP 1', ylab='UMAP 2',
                     suptitle=umap_suptitle,
                     slot_a_display=args.slot_a_display,
                     slot_b_display=args.slot_b_display,
                     out_path=args.out_dir / 'idxx_sweep_pair_umap.png')
    else:
        print("WARNING: umap-learn not installed; skipping UMAP panel")


if __name__ == '__main__':
    main()
