"""Similarity-leakage diagnostic: nearest train-neighbor identity, aa vs nt.

Purpose
-------
For a Stage 3 pair-split directory, compute the distribution of
nearest-train-neighbor percent identity for each unique TEST sequence,
separately at:

  - aa level on slot_a (HA in HA/NA bundles)
  - aa level on slot_b (NA)
  - dna level on slot_a contig
  - dna level on slot_b contig

The metric is `matches / max(L_train, L_test)` after right-padding each
sequence to a common length with distinct pad bytes (train pad = 0,
test pad = 255) so that pad-pad positions cannot spuriously match.
For each test sequence, the script reports the MAX identity over all
train sequences as the "nearest-neighbor" score.

Identity-disjointness (no sequence shared between train and test) is
enforced upstream by Stage 3's `split_strategy.mode=seq_disjoint`. This
script quantifies the residual **similarity** leakage that
identity-disjointness does not prevent: a test protein 1 aa away from
some train protein, or a test contig 1 nt away from some train contig.

Reads (per dataset_dir):
  - {train,val,test}_pairs.csv with columns
    seq_a, seq_b, dna_seq_a, dna_seq_b

Reports nearest-neighbor identity distribution per side x level. Useful
to flag whether the aa representation enjoys more similarity-leakage
than the dna representation (or vice versa), which is one of the five
named leakage modes in docs/methods/leakage.md.

Cross-references
----------------
  - docs/results/2026-05-13_aa_vs_nt_similarity_leakage.md
      The note that calls this script and records the headline result.
  - docs/methods/leakage.md
      The 5-mode leakage taxonomy. This script targets mode #4
      (cluster leakage), measured via nearest-neighbor identity rather
      than explicit clustering.
  - docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md
      The planned mitigation (cluster-disjoint splits with mmseqs2).

Example
-------
    python src/analysis/similarity_leakage_aa_vs_nt.py \\
        --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_regimes_20260512_114205

The script is read-only on the dataset; it does not modify any file.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def encode_uint8(seqs: list[str], max_len: int, pad_byte: int) -> np.ndarray:
    """Encode a list of sequences into a (N, max_len) uint8 array.

    Right-pads each row with `pad_byte` if the sequence is shorter than
    `max_len`. Choosing distinct pad bytes for train and test (e.g. 0
    vs 255) prevents pad-pad positions from spuriously matching when
    counting per-position equality.
    """
    arr = np.full((len(seqs), max_len), pad_byte, dtype=np.uint8)
    for i, s in enumerate(seqs):
        if not isinstance(s, str):
            continue
        L = min(len(s), max_len)
        arr[i, :L] = np.frombuffer(s[:L].encode('ascii', errors='replace'), dtype=np.uint8)
    return arr


def nearest_neighbor_identity(
    train_seqs: list[str],
    test_seqs: list[str],
    label: str,
    print_buckets: bool = True,
) -> np.ndarray:
    """For each unique test sequence, find the max % identity over all
    unique train sequences. Returns a (N_test_unique,) array of percent
    identities in [0, 100].

    Identity for a (train_i, test_j) pair is
        matches / max(L_train_i, L_test_j) * 100
    where `matches` is the number of positions where the two byte-encoded
    sequences agree, computed after independent right-padding with distinct
    pad bytes so pad-pad positions cannot inflate the match count.
    """
    train_seqs = [s for s in train_seqs if isinstance(s, str)]
    test_seqs  = [s for s in test_seqs  if isinstance(s, str)]
    if not train_seqs or not test_seqs:
        print(f'  WARNING: empty input for {label} — skipping')
        return np.array([])

    train_uniq = list(dict.fromkeys(train_seqs))
    test_uniq  = list(dict.fromkeys(test_seqs))
    train_lens = np.array([len(s) for s in train_uniq])
    test_lens  = np.array([len(s) for s in test_uniq])
    max_len = int(max(train_lens.max(), test_lens.max()))

    train_arr = encode_uint8(train_uniq, max_len, pad_byte=0)
    test_arr  = encode_uint8(test_uniq,  max_len, pad_byte=255)

    print(f'\n=== {label} ===')
    print(f'  train unique={len(train_uniq):,}, test unique={len(test_uniq):,}, '
          f'max_len={max_len}')

    nn_id = np.zeros(len(test_uniq), dtype=np.float64)
    for i in range(len(test_uniq)):
        matches = (train_arr == test_arr[i]).sum(axis=1)
        denom = np.maximum(train_lens, test_lens[i]).astype(np.float64)
        nn_id[i] = (matches / denom).max()
    pct = nn_id * 100

    print(
        f'  nearest-train identity (%):'
        f' min={pct.min():.2f}, p10={np.percentile(pct, 10):.2f},'
        f' p50={np.percentile(pct, 50):.2f}, p90={np.percentile(pct, 90):.2f},'
        f' p99={np.percentile(pct, 99):.2f}, max={pct.max():.2f},'
        f' mean={pct.mean():.2f}'
    )
    if print_buckets:
        buckets = [(99.5, 100.01), (99, 99.5), (98, 99),
                   (95, 98), (90, 95), (80, 90), (0, 80)]
        print(f'  bucket counts (n={len(pct):,}):')
        for lo, hi in buckets:
            n = ((pct >= lo) & (pct < hi)).sum()
            print(f'    [{lo:>5}, {hi:>6})  {n:>6,}  ({100 * n / len(pct):.2f}%)')
    return pct


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--dataset_dir', type=Path, required=True,
        help='Path to a Stage 3 run directory containing '
             '{train,test}_pairs.csv.',
    )
    args = parser.parse_args()

    if not args.dataset_dir.exists():
        print(f'ERROR: dataset_dir not found: {args.dataset_dir}', file=sys.stderr)
        sys.exit(1)

    train_path = args.dataset_dir / 'train_pairs.csv'
    test_path  = args.dataset_dir / 'test_pairs.csv'
    if not train_path.exists() or not test_path.exists():
        print(
            f'ERROR: missing train_pairs.csv or test_pairs.csv in '
            f'{args.dataset_dir}',
            file=sys.stderr,
        )
        sys.exit(2)

    train = pd.read_csv(train_path, engine='python')
    test  = pd.read_csv(test_path,  engine='python')
    print(f'train pairs: {len(train):,}, test pairs: {len(test):,}')

    aa_a  = nearest_neighbor_identity(
        train['seq_a'].tolist(), test['seq_a'].tolist(),
        'slot_a aa (e.g. HA in HA/NA bundle)',
    )
    aa_b  = nearest_neighbor_identity(
        train['seq_b'].tolist(), test['seq_b'].tolist(),
        'slot_b aa (e.g. NA in HA/NA bundle)',
    )
    dna_a = nearest_neighbor_identity(
        train['dna_seq_a'].tolist(), test['dna_seq_a'].tolist(),
        'slot_a dna contig',
    )
    dna_b = nearest_neighbor_identity(
        train['dna_seq_b'].tolist(), test['dna_seq_b'].tolist(),
        'slot_b dna contig',
    )

    print('\n=== Headline summary ===')
    for tag, pct in [
        ('slot_a aa',  aa_a),
        ('slot_a dna', dna_a),
        ('slot_b aa',  aa_b),
        ('slot_b dna', dna_b),
    ]:
        if len(pct) == 0:
            continue
        print(
            f'{tag}: mean={pct.mean():.3f}%, p50={np.percentile(pct, 50):.3f}%,'
            f' p90={np.percentile(pct, 90):.3f}%,'
            f' p99={np.percentile(pct, 99):.3f}%,'
            f' max={pct.max():.3f}%,'
            f' frac>=99.5%={(pct >= 99.5).mean() * 100:.2f}%'
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
