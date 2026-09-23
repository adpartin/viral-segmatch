#!/usr/bin/env python3
"""Export paired CDS sequences for a combined sequence similarity diagnostic.

For each requested schema pair and fold, this script reads the train and test pair tables. It
reconstructs each row's slot-A and slot-B CDS from their hashes, concatenates them without a
separator, and writes the retained positives and generated negatives to one FASTA file. A sibling
TSV maps each FASTA label to its class, split, assembly IDs, CDS hashes, and available metadata.

Each tree leaf therefore represents a pair row, not an isolate or an individual CDS. The resulting
tree can show whether generated negatives lie near retained positives under combined sequence
similarity. It is not a phylogeny of either segment and does not establish shared ancestry or
biological compatibility.

How?

- The pair tables carry `cds_dna_hash_a` / `cds_dna_hash_b` but not the sequences, so each hash is
  looked up in `cds_dna_final.parquet` for its `cds_dna_seq`.
- Records are labelled `{pair}_{split}_{pos|neg}_{index}`, short enough to stay readable as a tree
  leaf. The metadata goes in a sibling TSV keyed on that label. ggtree joins a frame in that shape
  directly; iTOL needs it converted to one of its own `DATASET_*` formats.
- Sequence lines wrap at 60 characters.
- Train and test rows are exported. A fold's validation rows are not.
- The concatenated sequence is required to equal the sum of the pair's two configured pinned
  lengths. That is a check on the total, not on each slot separately.

For datasets using within-fold negative sampling, negative rows recombine CDS drawn from the same
split's positives; they do not introduce new component sequences. This script does not align
sequences or build a tree.

CLI:
    python -m src.analysis.export_pair_fasta --pairs pb2_ha pb2_np --fold 0

Outputs:
    {pair}_fold{k}_{population}_all.fasta          one record per pair row
    {pair}_fold{k}_{population}_annotations.tsv    one row per record, keyed on its label
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.config_hydra import get_function_short_name_map, get_virus_config_hydra  # noqa: E402

LINE_WIDTH = 60
SPLITS = ('train', 'test')

# Pair-table column -> TSV column. The metadata is per slot, so each side keeps its own value.
ANNOTATION_COLUMNS = {
    'assembly_id_a': 'assembly_a', 'assembly_id_b': 'assembly_b',
    'host_a': 'host_a', 'host_b': 'host_b',
    'hn_subtype_a': 'subtype_a', 'hn_subtype_b': 'subtype_b',
    'year_a': 'year_a', 'year_b': 'year_b',
    'cds_dna_hash_a': 'cds_hash_a', 'cds_dna_hash_b': 'cds_hash_b',
}


def load_cds_sequences(config) -> dict:
    """CDS DNA sequence for every hash in the corpus.

    Args:
      config: a resolved bundle config, naming the virus and data version.

    Returns:
      `cds_dna_hash` -> `cds_dna_seq`.
    """
    cds_path = (PROJ / 'data/processed' / config.virus.virus_name / config.virus.data_version
                / 'cds_dna_final.parquet')
    cds = pd.read_parquet(cds_path, columns=['cds_dna_hash', 'cds_dna_seq'])
    # One sequence per hash: the same CDS recurs across isolates, and they are identical by
    # definition of the hash.
    unique = cds.drop_duplicates('cds_dna_hash')
    return dict(zip(unique['cds_dna_hash'], unique['cds_dna_seq']))


def build_records(pairs: pd.DataFrame, pair_label: str, split: str, sequences: dict) -> pd.DataFrame:
    """One record per pair row, with its label and its concatenated sequence.

    Args:
      pairs: one split's pair table.
      pair_label: short pair name, e.g. `pb2_ha`.
      split: `train` or `test`.
      sequences: `cds_dna_hash` -> `cds_dna_seq`.

    Returns:
      The annotation columns plus `label`, `split`, `class` and `sequence`, positives first.

    Raises:
      ValueError: a hash has no sequence, so the record would be written truncated.
    """
    frames = []
    for class_name, label_value in (('pos', 1), ('neg', 0)):
        rows = pairs[pairs['label'] == label_value].reset_index(drop=True)
        seq_a = rows['cds_dna_hash_a'].map(sequences)
        seq_b = rows['cds_dna_hash_b'].map(sequences)
        missing = int(seq_a.isna().sum() + seq_b.isna().sum())
        if missing:
            raise ValueError(
                f"build_records: {pair_label} {split} {class_name} has {missing} hashes with no "
                f"sequence in cds_dna_final.parquet.")

        record = rows[list(ANNOTATION_COLUMNS)].rename(columns=ANNOTATION_COLUMNS)
        record.insert(0, 'label', [f'{pair_label}_{split}_{class_name}_{i:04d}'
                                   for i in range(len(rows))])
        record.insert(1, 'split', split)
        record.insert(2, 'class', class_name)
        record['sequence'] = seq_a + seq_b
        frames.append(record)
    return pd.concat(frames, ignore_index=True)


def check_lengths(records: pd.DataFrame, expected_nt: int, pair_label: str) -> None:
    """Fail when a concatenated record is not the sum of the pair's two pinned lengths.

    Args:
      records: the records for one pair.
      expected_nt: slot-A pin plus slot-B pin.
      pair_label: short pair name, used in the error text.

    Returns:
      None.

    Raises:
      ValueError: at least one record has another length.
    """
    lengths = records['sequence'].str.len()
    wrong = lengths[lengths != expected_nt]
    if not wrong.empty:
        raise ValueError(
            f"check_lengths: {pair_label} expects {expected_nt:,} nt per record; "
            f"{len(wrong)} records differ, e.g. {sorted(set(wrong))[:5]}.")


def write_fasta(records: pd.DataFrame, out_path: Path) -> None:
    """Write the records as FASTA, wrapping sequence lines.

    Args:
      records: needs `label` and `sequence`.
      out_path: file to write.

    Returns:
      None. Writes `out_path`.
    """
    with open(out_path, 'w') as handle:
        for label, sequence in zip(records['label'], records['sequence']):
            handle.write(f'>{label}\n')
            for start in range(0, len(sequence), LINE_WIDTH):
                handle.write(sequence[start:start + LINE_WIDTH] + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--pairs', nargs='+', default=['pb2_ha', 'pb2_np'],
                        help='pair tokens, matching the dataset directory suffixes')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--bundle_prefix', default='flu_28p_codon_')
    parser.add_argument('--dataset_prefix', default='exp3_28p_codon_')
    parser.add_argument('--population', default='human_h3n2_2024',
                        help='population token for the output file names')
    parser.add_argument('--out_dir', type=Path,
                        default=PROJ / 'results/flu/July_2025/pair_fasta_human_h3n2_2024')
    args = parser.parse_args()

    config = get_virus_config_hydra(f'{args.bundle_prefix}{args.pairs[0]}',
                                    config_path=str(PROJ / 'conf'))
    runs_dir = (PROJ / 'data/datasets' / config.virus.virus_name / config.virus.data_version
                / 'runs')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading CDS sequences from {config.virus.data_version}...')
    sequences = load_cds_sequences(config)
    print(f'  {len(sequences):,} unique CDS sequences\n')

    for pair_label in args.pairs:
        pair_config = get_virus_config_hydra(f'{args.bundle_prefix}{pair_label}',
                                             config_path=str(PROJ / 'conf'))
        short_of = get_function_short_name_map(pair_config)
        pins = {str(k): int(v['nt']) for k, v in dict(pair_config.virus.cds_length).items()}
        proteins = [short_of[f] for f in pair_config.dataset.schema_pair]
        expected_nt = sum(pins[p] for p in proteins)

        dataset_dir = runs_dir / f'{args.dataset_prefix}{pair_label}' / f'fold_{args.fold}'
        frames = []
        for split in SPLITS:
            pairs = pd.read_parquet(dataset_dir / f'{split}_pairs.parquet')
            frames.append(build_records(pairs, pair_label, split, sequences))
        records = pd.concat(frames, ignore_index=True)
        check_lengths(records, expected_nt, pair_label)

        stem = f'{pair_label}_fold{args.fold}_{args.population}'
        fasta_path = args.out_dir / f'{stem}_all.fasta'
        tsv_path = args.out_dir / f'{stem}_annotations.tsv'
        write_fasta(records, fasta_path)
        records.drop(columns=['sequence']).to_csv(tsv_path, sep='\t', index=False)

        counts = records.groupby(['split', 'class']).size().to_dict()
        print(f'{"-".join(proteins)}  {expected_nt:,} nt per record '
              f'({" + ".join(f"{p} {pins[p]:,}" for p in proteins)})')
        for split in SPLITS:
            print(f'  {split:5s} pos {counts[(split, "pos")]:5,}  neg {counts[(split, "neg")]:5,}')
        print(f'  {fasta_path}')
        print(f'  {tsv_path}\n')
    print('Done.')


if __name__ == '__main__':
    main()
