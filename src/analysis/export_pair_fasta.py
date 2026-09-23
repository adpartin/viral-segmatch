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
from omegaconf import OmegaConf

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.config_hydra import get_function_short_name_map  # noqa: E402

LINE_WIDTH = 60
SPLITS = ('train', 'test')

# Pair-table column -> TSV column. The metadata is per slot, so each side keeps its own value.
# `pair_key` is the dataset's own key for the row, so a leaf traces straight back to its source.
ANNOTATION_COLUMNS = {
    'pair_key': 'pair_key',
    'assembly_id_a': 'assembly_a', 'assembly_id_b': 'assembly_b',
    'host_a': 'host_a', 'host_b': 'host_b',
    'hn_subtype_a': 'subtype_a', 'hn_subtype_b': 'subtype_b',
    'year_a': 'year_a', 'year_b': 'year_b',
    'cds_dna_hash_a': 'cds_hash_a', 'cds_dna_hash_b': 'cds_hash_b',
}


def load_cds_sequences(config) -> dict:
    """CDS DNA sequence for every hash in the corpus.

    Args:
      config: a run's resolved config, naming the virus and data version.

    Returns:
      `cds_dna_hash` -> `cds_dna_seq`.
    """
    cds_path = (PROJ / 'data/processed' / config.virus.virus_name / config.virus.data_version
                / 'cds_dna_final.parquet')
    cds = pd.read_parquet(cds_path, columns=['cds_dna_hash', 'cds_dna_seq'])
    # One row per hash. The same CDS recurs across isolates, and `cds_dna_hash` is the md5 of the
    # sequence, so the duplicates hold identical sequences.
    unique = cds.drop_duplicates('cds_dna_hash')
    return dict(zip(unique['cds_dna_hash'], unique['cds_dna_seq']))


def slot_lengths(config, short_of: dict) -> tuple:
    """The pinned CDS length each slot of the run's schema pair must have.

    Args:
      config: a run's resolved config.
      short_of: full function name -> short protein name.

    Returns:
      `(protein_a, pin_a, protein_b, pin_b)`, the proteins in slot order.
    """
    pins = {str(k): int(v['nt']) for k, v in dict(config.virus.cds_length).items()}
    protein_a, protein_b = (short_of[f] for f in config.dataset.schema_pair)
    return protein_a, pins[protein_a], protein_b, pins[protein_b]


def build_records(pairs: pd.DataFrame, pair_label: str, split: str, sequences: dict,
                  schema_pair: tuple, pin_a: int, pin_b: int) -> pd.DataFrame:
    """One record per pair row, with its label and its concatenated sequence.

    Args:
      pairs: one split's pair table.
      pair_label: short pair name, e.g. `pb2_ha`.
      split: `train` or `test`.
      sequences: `cds_dna_hash` -> `cds_dna_seq`.
      schema_pair: the run's (slot-A function, slot-B function), full names.
      pin_a: pinned CDS length for slot A.
      pin_b: pinned CDS length for slot B.

    Returns:
      The annotation columns plus `label`, `split`, `class` and `sequence`, positives first.

    Raises:
      ValueError: the table holds a label other than 0 or 1, its slot functions are not the run's
          schema pair in that order, a hash has no sequence, or a slot is off its pin.
    """
    labels = set(pairs['label'].dropna().unique())
    if labels - {0, 1} or len(pairs['label'].dropna()) != len(pairs):
        raise ValueError(
            f"build_records: {pair_label} {split} must hold only labels 0 and 1; found "
            f"{sorted(labels)} over {len(pairs):,} rows, {int(pairs['label'].isna().sum())} null.")

    # Slot order decides which sequence is concatenated first, so it is read off the rows rather
    # than assumed from the config, whose schema_pair order the builder may have canonicalized.
    observed = (set(pairs['func_a'].unique()), set(pairs['func_b'].unique()))
    if observed != ({schema_pair[0]}, {schema_pair[1]}):
        raise ValueError(
            f"build_records: {pair_label} {split} has func_a={sorted(observed[0])}, "
            f"func_b={sorted(observed[1])}; the run's schema_pair is {list(schema_pair)}.")

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
        # Per slot, not on the total: lengths that are wrong in opposite directions would pass a
        # check on the concatenation.
        for slot, seq, pin in (('A', seq_a, pin_a), ('B', seq_b, pin_b)):
            off_pin = seq.str.len()[seq.str.len() != pin]
            if not off_pin.empty:
                raise ValueError(
                    f"build_records: {pair_label} {split} {class_name} slot {slot} expects "
                    f"{pin:,} nt; {len(off_pin)} rows differ, e.g. {sorted(set(off_pin))[:5]}.")

        record = rows[list(ANNOTATION_COLUMNS)].rename(columns=ANNOTATION_COLUMNS)
        record.insert(0, 'label', [f'{pair_label}_{split}_{class_name}_{i:04d}'
                                   for i in range(len(rows))])
        record.insert(1, 'split', split)
        record.insert(2, 'class', class_name)
        record['sequence'] = seq_a + seq_b
        frames.append(record)

    records = pd.concat(frames, ignore_index=True)
    if len(records) != len(pairs):
        raise ValueError(
            f"build_records: {pair_label} {split} read {len(pairs):,} rows but built "
            f"{len(records):,} records.")
    return records


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
    parser.add_argument('--dataset_prefix', default='exp3_28p_codon_')
    parser.add_argument('--runs_dir', type=Path,
                        default=PROJ / 'data/datasets/flu/July_2025/runs')
    parser.add_argument('--population', default='human_h3n2_2024',
                        help='population token for the output file names')
    parser.add_argument('--out_dir', type=Path,
                        default=PROJ / 'results/flu/July_2025/pair_fasta_human_h3n2_2024')
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sequences = None

    for pair_label in args.pairs:
        run_dir = args.runs_dir / f'{args.dataset_prefix}{pair_label}'
        config = OmegaConf.load(run_dir / 'resolved_config.yaml')
        short_of = get_function_short_name_map(config)
        protein_a, pin_a, protein_b, pin_b = slot_lengths(config, short_of)
        schema_pair = tuple(str(f) for f in config.dataset.schema_pair)

        if sequences is None:
            print(f'Loading CDS sequences from {config.virus.data_version}...')
            sequences = load_cds_sequences(config)
            print(f'  {len(sequences):,} unique CDS sequences\n')

        fold_dir = run_dir / f'fold_{args.fold}'
        frames = [build_records(pd.read_parquet(fold_dir / f'{split}_pairs.parquet'),
                                pair_label, split, sequences, schema_pair, pin_a, pin_b)
                  for split in SPLITS]
        records = pd.concat(frames, ignore_index=True)

        stem = f'{pair_label}_fold{args.fold}_{args.population}'
        fasta_path = args.out_dir / f'{stem}_all.fasta'
        tsv_path = args.out_dir / f'{stem}_annotations.tsv'
        write_fasta(records, fasta_path)
        records.drop(columns=['sequence']).to_csv(tsv_path, sep='\t', index=False)

        counts = records.groupby(['split', 'class']).size().to_dict()
        # The negative sampler decides whether negatives reuse the positives' own sequences, which
        # is what the tree is read against, so it is named here rather than left to the config.
        print(f'{protein_a}-{protein_b}  {pin_a + pin_b:,} nt per record '
              f'({protein_a} {pin_a:,} + {protein_b} {pin_b:,})  '
              f'negative_scope: {config.dataset.split_strategy.negative_scope}')
        for split in SPLITS:
            print(f'  {split:5s} pos {counts[(split, "pos")]:5,}  neg {counts[(split, "neg")]:5,}')
        print(f'  {fasta_path}')
        print(f'  {tsv_path}\n')
    print('Done.')


if __name__ == '__main__':
    main()
