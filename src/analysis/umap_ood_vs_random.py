"""
k-mer UMAPs of the OOD-vs-random split geometry (one fold): per slot (HA, NA) x arm (ood, random),
4 separate figures. Points = unique POSITIVE sequences of the slot, colored test vs train for the chosen
fold; each slot is embedded ONCE and reused for both arms, so ood/random sit on identical coordinates
and differ only in coloring.

The two arms partition the SAME pool over the SAME feature space (same sequences -> same embedding); the
figures show how each split cuts it. OOD test = the held-out CC (a contiguous region -> the model must
extrapolate); random test = spread across the hubs (-> interpolation, the in-distribution control).

Representation is the exact k-mer matrix the model consumes -- reuses `umap_cc.cluster_vectors`
(cds_dna_hash -> k-mer row, TruncatedSVD) and `dim_reduction_utils.compute_umap_reduction`. A thin
fixed-color scatter (not `plot_utils.umap_scatter`) is used so test/train keep the SAME colors across the
4 figs (umap_scatter colors by size rank, which would swap test/train between the ood and random panels).

CLI:
    python -m src.analysis.umap_ood_vs_random \\
        --run_dir data/datasets/flu/July_2025/runs/dataset_cc_nt_cds_ood_ood_vs_random_t095 \\
        --fold 0 --pair HA-NA --alphabet nt_cds
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.umap_cc import cluster_vectors  # noqa: E402
from src.utils.dim_reduction_utils import compute_umap_reduction  # noqa: E402
from src.utils.plot_utils import savefig  # noqa: E402

_TEST_COLOR, _TRAIN_COLOR = '#d43d51', '#d0d0d0'   # crimson test (on top) over gray train background


def _fragmented_ccmap(run_dir: Path, pair: str):
    """(pair_key -> fragmented cc_id, cc_id -> size rank) for cross-referencing the plot to the CC-size
    barplot, or (None, None) if the fragmented artifact can't be located. Path derived from the run's
    resolved_config cluster_id_path (build_cc_structure layout: cc_{source}/{pair}/{t}/fragmented). NB
    fragmented cc_id is a per-file union-find label (0..n-1, size-independent), unrelated to the natural
    cc_id -- so we also return the size rank (CC1 = largest) that the barplot labels by."""
    try:
        from omegaconf import OmegaConf
        cip = Path(str(OmegaConf.load(run_dir / 'resolved_config.yaml').dataset.split_strategy.cluster_id_path))
        if not cip.is_absolute():
            cip = PROJ / cip
        source = cip.parents[1].name.removeprefix('clusters_')          # e.g. nt_cds_ood
        cc_pq = cip.parents[2] / f'cc_{source}' / pair / cip.parent.name / 'fragmented' / 'pairs_with_cc.parquet'
        df = pd.read_parquet(cc_pq, columns=['pair_key', 'cc_id'])
    except Exception:
        return None, None
    key2cc = dict(zip(df['pair_key'].astype(str), df['cc_id']))
    ranks = {int(cc): r for r, cc in                                    # CC1 = largest by pair count
             enumerate(df.groupby('cc_id').size().sort_values(ascending=False).index, 1)}
    return key2cc, ranks


def _scatter(emb, is_test, out_png: Path, title: str) -> None:
    """Shared-embedding scatter: gray train+val background, crimson test on top. Fixed colors so the 4
    figs are directly comparable. Legend counts are UNIQUE sequences (the points); pair counts are in
    the title. 'train+val' = every positive not held out this fold (val is seen during training)."""
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(emb[~is_test, 0], emb[~is_test, 1], s=6, c=_TRAIN_COLOR, linewidths=0,
               rasterized=True, label=f'train+val · {int((~is_test).sum()):,} seqs')
    # test drawn on top at alpha 0.5: dense regions (OOD held-out CC) still build up to near-solid
    # crimson, while where test interleaves train (random arm) the gray shows through -> intermixing reads.
    ax.scatter(emb[is_test, 0], emb[is_test, 1], s=13, c=_TEST_COLOR, alpha=0.5, linewidths=0,
               rasterized=True, label=f'test · {int(is_test.sum()):,} seqs')
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.set_title(title, fontsize=10)
    savefig(out_png, dpi=200)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--run_dir', type=Path, required=True, help='OOD-vs-random run dir (holds ood/ + random/)')
    p.add_argument('--fold', type=int, default=0, help='fold to visualize (default 0)')
    p.add_argument('--pair', default='HA-NA', help='slot pair, e.g. HA-NA (a=left/HA, b=right/NA; title only)')
    p.add_argument('--alphabet', default='nt_cds', help='k-mer alphabet (default nt_cds)')
    p.add_argument('--out_dir', type=Path, default=None, help='default: <run_dir>/figures')
    args = p.parse_args()

    out_dir = args.out_dir or (args.run_dir / 'figures')
    out_dir.mkdir(parents=True, exist_ok=True)
    sa, sb = args.pair.split('-')

    def load(arm, split):
        return pd.read_csv(args.run_dir / arm / f'fold_{args.fold}' / f'{split}_pairs.csv',
                           usecols=['pair_key', 'label', 'cds_dna_hash_a', 'cds_dna_hash_b'],
                           keep_default_na=False, na_values=[''])

    arms = {a: {s: load(a, s) for s in ('train', 'val', 'test')} for a in ('ood', 'random')}

    # per-arm (slot-independent) title metadata: positive pair counts + the test-CC cross-reference.
    key2cc, cc_rank = _fragmented_ccmap(args.run_dir, args.pair)
    arm_meta = {}
    for arm in ('ood', 'random'):
        pos = {s: int((arms[arm][s]['label'] == 1).sum()) for s in ('train', 'val', 'test')}
        cc_label = 'test = held-out CC' if arm == 'ood' else 'test = size-matched random'
        if key2cc is not None:                                          # OOD test = exactly one fragmented CC
            tks = arms[arm]['test'].loc[arms[arm]['test']['label'] == 1, 'pair_key'].astype(str)
            ccs = sorted({key2cc[k] for k in tks if k in key2cc})
            if arm == 'ood' and len(ccs) == 1:
                cc = int(ccs[0])
                cc_label = f'test = cc_id {cc} (CC{cc_rank.get(cc, "?")})'
        arm_meta[arm] = {'pos': pos, 'cc': cc_label}
        print(f'  {arm:7}: {arm_meta[arm]["cc"]} | positive pairs train {pos["train"]:,} / '
              f'val {pos["val"]:,} / test {pos["test"]:,}')

    for slot_name, col in [(sa, 'cds_dna_hash_a'), (sb, 'cds_dna_hash_b')]:
        # pool positives for this slot (identical across arms) -> embed ONCE, reuse for both arms
        pool = pd.concat([arms['ood'][s] for s in ('train', 'val', 'test')])
        hashes = sorted(set(pool.loc[pool['label'] == 1, col].astype(str)))
        X, keep = cluster_vectors(hashes, args.alphabet)
        hk = np.asarray(hashes)[keep]
        emb = compute_umap_reduction(
            X, n_components=2, n_neighbors=15, min_dist=0.1,
            metric='cosine', random_state=42)[0]
        print(f'{slot_name}: {len(hashes):,} unique pos seqs ({keep.sum():,} with a k-mer row) embedded once')

        for arm in ('ood', 'random'):
            test_h = set(arms[arm]['test'].loc[arms[arm]['test']['label'] == 1, col].astype(str))
            is_test = np.fromiter((h in test_h for h in hk), dtype=bool, count=len(hk))
            out = out_dir / f'umap_kmer_{slot_name}_{arm}_fold{args.fold}.png'
            pos = arm_meta[arm]['pos']
            title = (f'{args.pair} {args.alphabet} {slot_name} - {arm} arm - fold {args.fold}\n'
                     f'{arm_meta[arm]["cc"]}  ·  positive pairs: train {pos["train"]:,} / '
                     f'val {pos["val"]:,} / test {pos["test"]:,}')
            _scatter(emb, is_test, out, title)
            print(f'  {arm:7} {slot_name}: test={int(is_test.sum()):,} unq / '
                  f'train+val={int((~is_test).sum()):,} unq -> {out.name}')


if __name__ == '__main__':
    main()
