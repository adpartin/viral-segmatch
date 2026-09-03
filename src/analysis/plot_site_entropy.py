"""Per-site Shannon entropy along each CDS, for a dataset built with equal-length CDS.

Serves two purposes for the per-site nucleotide features in
`docs/plans/2026-08-28_per_site_nt_features_plan.md` (step 2):

1. A conservation map. Step 6 reads a per-position importance map against this, to say whether
   the model leans on positions that vary or on positions that do not.
2. A check that the positions are comparable at all. Per-site features compare position 200
   across sequences, which only means something if position 200 is the same place in every one.
   If the sequences were not lined up, entropy would be high and flat along the whole length.
   This catches wholesale misalignment, not one or two shifted sequences -- the
   `require_complete_cds_at_pinned_length` filter is what rules those out.

The reading-frame check is the sharper of the two. Most changes at the third base of a codon do
not change the amino acid, so third positions should be more variable than first and second. If
that ordering is absent, the frame is wrong, and a flat-entropy plot would not say so nearly as
clearly.

Entropy is computed over UNIQUE CDS, one row per `cds_dna_hash`, not per pair row: a heavily
sampled strain would otherwise decide the answer. It uses every split. Nothing is fitted here, so
that leaks nothing -- but if entropy is ever used to SELECT positions, it must be recomputed on
the training split alone.

The per-protein CSVs are named `site_entropy_HA.csv` / `site_entropy_NA.csv` rather than carrying
a protein column, so they do not hit the `NA`-parsed-as-NaN trap that `pd.read_csv` sets for any
column holding the literal string `NA` (Neuraminidase).

Outputs (to `--out_dir`, by default derived from the dataset dir):
    site_entropy.png                per-position trace and codon-position summary, one row per protein
    site_entropy_{SHORT}.csv        position, codon_position, entropy_bits, n_symbols

CLI:
    python -m src.analysis.plot_site_entropy \\
        --dataset_dir data/datasets/flu/July_2025/runs/dataset_ha_na_h3n2_2024_random_cv4_pinned_length
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

from src.utils.config_hydra import get_function_short_name_map, get_virus_config_hydra  # noqa: E402
from src.utils.plot_utils import savefig, setup_plot_style  # noqa: E402
from src.utils.site_utils import column_entropy, sequences_to_byte_matrix  # noqa: E402

# Sampled from tmp/score/h3n2_f1_macro_within_fold.png so the per-site figures read as one series
# with the score plots.
TRACE_COLOR = '#4C7CAB'
ACCENT_COLOR = '#CF8793'
MARKER_EDGE = '#222222'

# Codon position labels, indexed by (0-based position along the CDS) % 3.
_CODON_POSITION_LABELS = ['1st', '2nd', '3rd']


def collect_slot_hashes(dataset_dir: Path, n_folds: int) -> dict:
    """Collect the unique CDS hash per protein from every fold and split of a dataset run.

    Reads `func_a`/`func_b` and `cds_dna_hash_a`/`cds_dna_hash_b` from each fold's pair tables and
    unions them, so the result is the CDS population the dataset was built on rather than one
    split's view of it. Handles a single-split run (no `fold_*` dirs) as well as a CV run.

    Args:
      dataset_dir: run directory holding `fold_{k}/` subdirs, or the split files directly.
      n_folds: number of folds to look for; ignored when the run is single-split.

    Returns:
      `{function: set of cds_dna_hash}`, keyed by the full function string.

    Raises:
      FileNotFoundError: no pair table was found under `dataset_dir`.
    """
    fold_dirs = [dataset_dir / f'fold_{k}' for k in range(n_folds)]
    fold_dirs = [d for d in fold_dirs if d.exists()] or [dataset_dir]

    per_function: dict[str, set] = {}
    n_tables = 0
    for d in fold_dirs:
        for split in ('train', 'val', 'test'):
            path = d / f'{split}_pairs.parquet'
            if not path.exists():
                continue
            n_tables += 1
            pairs = pd.read_parquet(
                path, columns=['func_a', 'func_b', 'cds_dna_hash_a', 'cds_dna_hash_b'])
            for slot in ('a', 'b'):
                for func, group in pairs.groupby(f'func_{slot}')[f'cds_dna_hash_{slot}']:
                    per_function.setdefault(str(func), set()).update(group.dropna())
    if n_tables == 0:
        raise FileNotFoundError(
            f"no *_pairs.parquet found under {dataset_dir} (looked in fold_0..{n_folds - 1} "
            f"and in the directory itself)")
    print(f"Read {n_tables} pair tables from {len(fold_dirs)} directories.")
    return per_function


def site_entropy(sequences: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Shannon entropy in bits at each position of a set of equal-length sequences.

    Thin wrapper over `site_utils.column_entropy`, which the importance map also uses, so the
    entropy of a position means the same thing in both places.

    Args:
      sequences: equal-length sequences, one per unique CDS. Case is normalised, so a lowercase
          store and an uppercase one give the same answer.

    Returns:
      `(entropy_bits, n_symbols)`, both length L. `n_symbols` is how many distinct characters
      appear at that position, which separates "one rare variant" from "genuinely mixed".

    Raises:
      ValueError: the sequences are not all the same length, or the list is empty.
    """
    matrix = sequences_to_byte_matrix(sequences)
    return column_entropy(matrix)


def summarize(short: str, entropy: np.ndarray, n_seqs: int) -> pd.DataFrame:
    """Print the per-protein entropy summary and return the codon-position table.

    Args:
      short: short protein name, used in the printed lines.
      entropy: per-position entropy in bits.
      n_seqs: number of unique CDS the entropy was computed over.

    Returns:
      One row per codon position with `codon_position`, `mean_bits` and `n_sites`.
    """
    codon_position = np.arange(len(entropy)) % 3
    invariant = int((entropy == 0).sum())
    print(f"\n{short}: {n_seqs:,} unique CDS, {len(entropy):,} positions")
    print(f"  mean entropy {entropy.mean():.4f} bits, max {entropy.max():.4f} "
          f"at position {int(entropy.argmax()) + 1}")
    print(f"  invariant positions (0 bits): {invariant:,} ({100 * invariant / len(entropy):.1f}%)")

    rows = []
    for i, label in enumerate(_CODON_POSITION_LABELS):
        at_position = entropy[codon_position == i]
        rows.append({'codon_position': label, 'mean_bits': float(at_position.mean()),
                     'n_sites': int(at_position.size)})
        print(f"  {label} codon position: mean {at_position.mean():.4f} bits "
              f"over {at_position.size:,} sites")
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset_dir', type=Path, required=True,
                   help='dataset run dir holding fold_*/ (or the split files directly)')
    p.add_argument('--config_bundle', default='flu_ha_na_h3n2_2024_random_cv4_pinned_length',
                   help='bundle to read function short names from')
    p.add_argument('--cds_final_path', type=Path,
                   default=PROJ / 'data/processed/flu/July_2025/cds_dna_final.parquet',
                   help='source of cds_dna_seq, keyed by cds_dna_hash')
    p.add_argument('--n_folds', type=int, default=4)
    p.add_argument('--out_dir', type=Path, default=None,
                   help='default: results/<virus>/<version>/<run name>/site_entropy')
    p.add_argument('--dpi', type=int, default=200)
    args = p.parse_args()

    if args.out_dir is None:
        # data/datasets/<virus>/<version>/runs/<run> -> results/<virus>/<version>/<run>/site_entropy
        parts = args.dataset_dir.resolve().parts
        i = parts.index('datasets')
        args.out_dir = PROJ / 'results' / parts[i + 1] / parts[i + 2] / parts[-1] / 'site_entropy'

    config = get_virus_config_hydra(args.config_bundle, config_path=str(PROJ / 'conf'))
    function_to_short = get_function_short_name_map(config)
    protein_order = list(config.virus.protein_order)

    hashes_by_function = collect_slot_hashes(args.dataset_dir, args.n_folds)
    functions = sorted(hashes_by_function, key=protein_order.index)

    cds = pd.read_parquet(args.cds_final_path, columns=['cds_dna_hash', 'cds_dna_seq'])
    hash_to_seq = dict(zip(cds['cds_dna_hash'], cds['cds_dna_seq']))

    results = []
    for func in functions:
        short = function_to_short.get(func, func)
        hashes = sorted(hashes_by_function[func])
        missing = [h for h in hashes if h not in hash_to_seq]
        if missing:
            raise ValueError(
                f"{short}: {len(missing):,} of {len(hashes):,} cds_dna_hash values are not in "
                f"{args.cds_final_path}. The dataset and the CDS table are out of step.")
        sequences = [hash_to_seq[h] for h in hashes]
        entropy, n_symbols = site_entropy(sequences)
        codon_summary = summarize(short, entropy, len(sequences))
        results.append({'short': short, 'entropy': entropy, 'n_symbols': n_symbols,
                        'n_seqs': len(sequences), 'codon_summary': codon_summary})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for rs in results:
        table = pd.DataFrame({
            'position': np.arange(1, len(rs['entropy']) + 1),
            'codon_position': [_CODON_POSITION_LABELS[i % 3] for i in range(len(rs['entropy']))],
            'entropy_bits': rs['entropy'],
            'n_symbols': rs['n_symbols'],
        })
        out_csv = args.out_dir / f"site_entropy_{rs['short']}.csv"
        table.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")

    setup_plot_style()
    fig, axes = plt.subplots(len(results), 2, figsize=(15, 3.4 * len(results)),
                             gridspec_kw={'width_ratios': [3.2, 1]}, squeeze=False)

    for row, rs in enumerate(results):
        entropy = rs['entropy']
        ax_trace, ax_codon = axes[row][0], axes[row][1]

        positions = np.arange(1, len(entropy) + 1)
        ax_trace.plot(positions, entropy, color=TRACE_COLOR, linewidth=0.7)
        ax_trace.set_xlim(1, len(entropy))
        ax_trace.set_ylim(0, max(0.5, float(entropy.max()) * 1.08))
        ax_trace.set_xlabel(f"position along the {rs['short']} CDS (nt)")
        ax_trace.set_ylabel('entropy (bits)')
        ax_trace.set_title(f"{rs['short']}: {len(entropy):,} positions, "
                           f"{rs['n_seqs']:,} unique CDS")
        ax_trace.grid(axis='y', alpha=0.3)

        codon_position = np.arange(len(entropy)) % 3
        by_position = [entropy[codon_position == i] for i in range(3)]
        bars = ax_codon.bar(_CODON_POSITION_LABELS, [v.mean() for v in by_position],
                            color=[TRACE_COLOR, TRACE_COLOR, ACCENT_COLOR],
                            edgecolor=MARKER_EDGE, linewidth=0.7)
        for bar, values in zip(bars, by_position):
            ax_codon.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                          f'{values.mean():.3f}', ha='center', va='bottom', fontsize=9)
        ax_codon.set_xlabel('codon position')
        ax_codon.set_ylabel('mean entropy (bits)')
        ax_codon.set_title('3rd should be highest\n(silent changes)', fontsize=10)
        ax_codon.grid(axis='y', alpha=0.3)

    fig.suptitle(f'{args.dataset_dir.name}  |  {args.n_folds} folds, all splits',
                 fontsize=10, y=1.005)
    fig.tight_layout()
    fig.text(0.995, 0.002, f'src/analysis/{Path(__file__).name}', ha='right', va='bottom',
             fontsize=7, color='0.45')
    out_png = savefig(args.out_dir / 'site_entropy.png', dpi=args.dpi)
    print(f"Done. Wrote {out_png}")


if __name__ == '__main__':
    main()
