"""
Compute per-site codon features placed in protein-alignment columns.

Sibling of `compute_site_features.py`, which needs every CDS at one pinned length. Here the
protein alignment decides which codons share a column, so CDS of different lengths, and CDS whose
proteins have internal gaps, get one shared set of sites. Planned in
docs/plans/2026-11-17_aligned_codon_features_plan.md.

Each protein's feature width is W + 1: the W columns of its final alignment plus one column after
the last. For one CDS whose aligned protein has n protein symbols (amino acids or X):

  codon i, i <= n    -> the column of the protein's i-th symbol
  the stop codon     -> the column immediately after the last symbol (the stop codon placement
                        rule, defined under "Method" in docs/results/2026-11-09_protein_alignment.md)
  any other column   -> GAP before the stop codon, AFTER_STOP after it

  aligned protein    M    K    -    V    -    (W = 5)
  CDS                ATG  AAA       GTT       TAA
  features           ATG  AAA  GAP  GTT  TAA  AFTER_STOP

The bundle given with `--config_bundle` must select aligned mode, codon units and ordinal encoding.
`site.alignment_dirs` names each protein's final alignment directory. The script reads
`selected_cds.parquet` and `aligned_proteins.fasta` there, after
`align_proteins.check_inspected_alignment` confirms that the alignment passed validation and that
every file matches the hash its manifest recorded.

Outputs (per protein, to its alignment directory):
  site_features_codon_{SHORT}_aligned.npz            uint8 codes, one row per unique CDS
  site_features_codon_{SHORT}_aligned_index.parquet  cds_dna_hash -> row
  site_features_codon_{SHORT}_aligned_metadata.json  code map, width, source hashes, row checks
  site_features_codon_{SHORT}_aligned_table.parquet  the same cells as text: uppercase triplet,
                                                     GAP or AFTER_STOP; columns site_1..site_{W+1}

Codon codes are GenSLM's ids, as in `compute_site_features.py`, with `<unk>` for a codon holding
an ambiguous base. GAP is 97 and AFTER_STOP is 98, above every GenSLM id (0-96).

Before anything is written, every row is checked against its source:
  row check       the codon cells, in column order, are the codes `encode_codon` gives for the
                  CDS, and the table's codon cells join to `cds_dna_seq.upper()`
  position check  in both outputs, codon cells are exactly the protein's symbol columns plus the
                  stop codon's column; other cells are GAP before the stop codon and AFTER_STOP
                  after it

Stops without writing when any output already exists, unless `--force_recompute` is given.

Usage:
    python src/embeddings/compute_protein_aligned_codon_features.py \
        --config_bundle flu_28p_aligned_codon_pb1_ha
    python src/embeddings/compute_protein_aligned_codon_features.py \
        --config_bundle flu_28p_aligned_codon_pb1_ha --proteins PB1 HA
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analysis.align_proteins import check_inspected_alignment  # noqa: E402
from src.embeddings.compute_site_features import (  # noqa: E402
    CODON_UNK_CODE,
    GENSLM_TOKENIZER,
    encode_codon,
    load_genslm_codon_codes,
)
from src.utils.config_hydra import (  # noqa: E402
    get_function_short_name_map,
    get_virus_config_hydra,
)
from src.utils.experiment_utils import get_git_info  # noqa: E402
from src.utils.fasta_utils import read_fasta  # noqa: E402
from src.utils.site_utils import (  # noqa: E402
    AFTER_STOP_CODE,
    AFTER_STOP_TEXT,
    GAP_CODE,
    GAP_TEXT,
    aligned_site_cache_stem,
    sequences_to_byte_matrix,
)
from src.utils.timer_utils import Timer  # noqa: E402

UNIT = 'codon'

# The same two states in the int32 triplet matrix, where a codon cell holds its three ASCII bytes
# packed as b0 << 16 | b1 << 8 | b2, which is never negative.
GAP_TRIPLET = -1
AFTER_STOP_TRIPLET = -2

STOP_CODON_PLACEMENT_RULE = (
    "Each CDS's stop codon goes in the column immediately after its last protein symbol. When "
    "that symbol is in the last alignment column, the stop codon goes in one extra column added "
    "after the alignment.")

OUTPUT_SUFFIXES = ('.npz', '_index.parquet', '_metadata.json', '_table.parquet')
SELECTED_COLUMNS = ['function', 'cds_dna_hash', 'cds_dna_seq', 'cds_length', 'prot_hash',
                    'prot_seq']

# Cell classes for the position check.
CODON_CELL, GAP_CELL, AFTER_STOP_CELL = 0, 1, 2


def output_paths(alignment_dir: Path, protein: str) -> dict:
    """The four output files of one protein.

    Args:
      alignment_dir: the protein's final alignment directory.
      protein: short protein name, e.g. `HA`.

    Returns:
      Suffix (`.npz`, `_index.parquet`, `_metadata.json`, `_table.parquet`) -> path.
    """
    stem = aligned_site_cache_stem(UNIT, protein)
    paths = {suffix: alignment_dir / f'{stem}{suffix}' for suffix in OUTPUT_SUFFIXES}
    return paths


def symbol_columns(aligned_row: str) -> np.ndarray:
    """0-based columns of an aligned protein row that hold a protein symbol, not a gap.

    Args:
      aligned_row: one row of `aligned_proteins.fasta`, gaps written as `-`.

    Returns:
      int64 array of column indices, ascending.
    """
    row = np.frombuffer(aligned_row.encode('ascii'), dtype=np.uint8)
    columns = np.flatnonzero(row != ord('-'))
    return columns


def load_selection(alignment_dir: Path, function: str) -> tuple[pd.DataFrame, dict, dict]:
    """Read one protein's selected CDS and aligned proteins, after checking they belong together.

    Args:
      alignment_dir: the protein's final alignment directory.
      function: the protein's full function string.

    Returns:
      `(unique_cds, aligned, hashes)`. `unique_cds` has one row per `cds_dna_hash`, sorted by it,
      with the columns in `SELECTED_COLUMNS`. `aligned` maps `prot_hash` to its aligned row.
      `hashes` is file name -> sha256, from `check_inspected_alignment`.

    Raises:
      ValueError: the selection holds another function, a CDS hash names two proteins, the
          aligned IDs differ from the selected proteins, the rows differ in width from
          `aligned_width`, or a row without its gaps is not its protein without the terminal `*`.
    """
    hashes = check_inspected_alignment(alignment_dir)
    selected = pd.read_parquet(alignment_dir / 'selected_cds.parquet', columns=SELECTED_COLUMNS)
    other_functions = sorted(set(selected['function']) - {function})
    if other_functions:
        raise ValueError(f'{alignment_dir}: selected_cds.parquet holds {other_functions}, not '
                         f'only {function!r}.')
    proteins_per_cds = selected.groupby('cds_dna_hash')['prot_hash'].nunique()
    if proteins_per_cds.gt(1).any():
        raise ValueError(f'{alignment_dir}: {int(proteins_per_cds.gt(1).sum())} CDS hashes name '
                         f'more than one protein.')
    unique_cds = (selected.drop_duplicates('cds_dna_hash')
                  .sort_values('cds_dna_hash').reset_index(drop=True))

    # The codon count is the CDS length over 3, and the protein, including its terminal *, has
    # one symbol per codon.
    bad_length = (unique_cds['cds_dna_seq'].str.len().ne(unique_cds['cds_length'])
                  | unique_cds['cds_length'].ne(3 * unique_cds['prot_seq'].str.len()))
    if bad_length.any():
        raise ValueError(f'{alignment_dir}: {int(bad_length.sum())} CDS disagree with their '
                         f'recorded length or their protein, e.g. '
                         f'{unique_cds.loc[bad_length, "cds_dna_hash"].iloc[0]}.')

    aligned = read_fasta(alignment_dir / 'aligned_proteins.fasta')
    if set(aligned) != set(unique_cds['prot_hash']):
        raise ValueError(f'{alignment_dir}: the aligned protein IDs differ from the selected '
                         f'proteins.')
    validation = json.loads((alignment_dir / 'alignment_validation.json').read_text())
    widths = {len(row) for row in aligned.values()}
    if widths != {int(validation['aligned_width'])}:
        raise ValueError(f'{alignment_dir}: aligned rows have widths {sorted(widths)}, but '
                         f'alignment_validation.json records {validation["aligned_width"]}.')

    # A row without its gaps must be its protein without the terminal *, so the i-th protein
    # symbol and the i-th codon describe the same residue.
    proteins = unique_cds.drop_duplicates('prot_hash')
    mismatched = [prot_hash for prot_hash, prot_seq in zip(proteins['prot_hash'],
                                                           proteins['prot_seq'])
                  if not prot_seq.endswith('*')
                  or aligned[prot_hash].replace('-', '') != prot_seq[:-1]]
    if mismatched:
        raise ValueError(f'{alignment_dir}: {len(mismatched)} aligned rows are not their protein '
                         f'without the terminal *, e.g. {mismatched[0]}.')
    return unique_cds, aligned, hashes


def place_codons(unique_cds: pd.DataFrame, aligned: dict,
                 genslm_codes: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Place every CDS's codons in its protein's alignment columns.

    CDS are encoded one length at a time, because `sequences_to_byte_matrix` needs equal lengths.

    Args:
      unique_cds: one row per CDS, from `load_selection`.
      aligned: `prot_hash` -> aligned row.
      genslm_codes: `{codon: id}` from `load_genslm_codon_codes`.

    Returns:
      `(codes, triplets, code_map)`. `codes` is (n CDS, W + 1) uint8 with GenSLM ids, `GAP_CODE`
      and `AFTER_STOP_CODE`. `triplets` is the same layout as int32, holding each codon's
      upper-cased bytes packed into one int, or `GAP_TRIPLET` / `AFTER_STOP_TRIPLET`. `code_map`
      is `encode_codon`'s map.

    Raises:
      ValueError: a protein's symbol count is not its CDS's codon count minus the stop codon.
    """
    width = len(next(iter(aligned.values()))) + 1
    n_cds = len(unique_cds)
    codes = np.empty((n_cds, width), dtype=np.uint8)
    triplets = np.empty((n_cds, width), dtype=np.int32)
    code_map = None

    for cds_length, group in unique_cds.groupby('cds_length'):
        rows = group.index.to_numpy()
        n_codons = int(cds_length) // 3
        byte_matrix = sequences_to_byte_matrix(group['cds_dna_seq'].tolist())
        group_codes, code_map = encode_codon(byte_matrix, genslm_codes)
        group_bytes = byte_matrix.reshape(len(group), n_codons, 3).astype(np.int32)
        group_triplets = (group_bytes[:, :, 0] << 16) | (group_bytes[:, :, 1] << 8) \
            | group_bytes[:, :, 2]

        # Target column of every codon: the protein's symbol columns for all codons but the
        # last, then the column after the last symbol for the stop codon.
        group_proteins = pd.Index(group['prot_hash'].unique())
        positions = [symbol_columns(aligned[prot_hash]) for prot_hash in group_proteins]
        wrong_count = [prot_hash for prot_hash, columns in zip(group_proteins, positions)
                       if len(columns) != n_codons - 1]
        if wrong_count:
            raise ValueError(f'{len(wrong_count)} proteins of {cds_length}-nt CDS do not have '
                             f'{n_codons - 1} protein symbols, e.g. {wrong_count[0]}.')
        protein_rows = group_proteins.get_indexer(group['prot_hash'])
        symbol_targets = np.stack(positions)[protein_rows]
        stop_columns = symbol_targets[:, -1] + 1
        targets = np.column_stack([symbol_targets, stop_columns])

        # Start from GAP before the stop codon's column and AFTER_STOP after it, then write the
        # codons over their target columns.
        after_stop = np.arange(width)[None, :] > stop_columns[:, None]
        block_codes = np.where(after_stop, AFTER_STOP_CODE, GAP_CODE).astype(np.uint8)
        block_triplets = np.where(after_stop, AFTER_STOP_TRIPLET, GAP_TRIPLET).astype(np.int32)
        block_rows = np.arange(len(group))[:, None]
        block_codes[block_rows, targets] = group_codes
        block_triplets[block_rows, targets] = group_triplets
        codes[rows] = block_codes
        triplets[rows] = block_triplets
        print(f'  {cds_length} nt: {len(group):,} CDS, {len(group_proteins):,} proteins')
    return codes, triplets, code_map


def build_table_vocabulary(triplets: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Turn the triplet matrix into indices into a list of cell texts.

    Args:
      triplets: (n CDS, W + 1) int32 from `place_codons`.

    Returns:
      `(text_ids, vocabulary)`. `text_ids` is int16 with the shape of `triplets`; `vocabulary`
      holds the uppercase triplets, `GAP_TEXT` and `AFTER_STOP_TEXT`.

    Raises:
      ValueError: more distinct cell texts than int16 can index.
    """
    values, inverse = np.unique(triplets, return_inverse=True)
    if len(values) >= np.iinfo(np.int16).max:
        raise ValueError(f'{len(values):,} distinct cell texts do not fit int16 indices.')
    special = {GAP_TRIPLET: GAP_TEXT, AFTER_STOP_TRIPLET: AFTER_STOP_TEXT}
    vocabulary = [special[int(value)] if int(value) in special
                  else bytes([(int(value) >> 16) & 255, (int(value) >> 8) & 255,
                              int(value) & 255]).decode('ascii')
                  for value in values]
    text_ids = inverse.reshape(triplets.shape).astype(np.int16)
    return text_ids, vocabulary


def check_rows(codes: np.ndarray, text_ids: np.ndarray, vocabulary: list[str],
               unique_cds: pd.DataFrame, aligned: dict, genslm_codes: dict) -> int:
    """Run the row and position checks on every row of both outputs.

    The expected values are rebuilt from the CDS and the aligned rows, not from `place_codons`'s
    target columns, so a placement error shows up as a mismatch.

    Args:
      codes: (n CDS, W + 1) uint8 from `place_codons`.
      text_ids: (n CDS, W + 1) int16 from `build_table_vocabulary`.
      vocabulary: cell texts that `text_ids` index.
      unique_cds: one row per CDS, in the row order of `codes`.
      aligned: `prot_hash` -> aligned row.
      genslm_codes: `{codon: id}` from `load_genslm_codon_codes`.

    Returns:
      The number of rows that passed both checks, which is every row.

    Raises:
      ValueError: a row fails either check; the message names the first failing CDS.
    """
    n_cds, width = codes.shape
    hashes = unique_cds['cds_dna_hash'].to_numpy()

    def fail(check: str, failing: np.ndarray) -> None:
        raise ValueError(f'{check} failed on {int(failing.sum()):,} of {n_cds:,} rows, e.g. '
                         f'cds_dna_hash {hashes[np.flatnonzero(failing)[0]]}.')

    # Position check: the cell class each column must have, from the aligned row alone.
    proteins = pd.Index(unique_cds['prot_hash'].unique())
    expected_by_protein = np.full((len(proteins), width), GAP_CELL, dtype=np.int8)
    for i, prot_hash in enumerate(proteins):
        is_symbol = np.frombuffer(aligned[prot_hash].encode('ascii'), dtype=np.uint8) != ord('-')
        stop_column = int(np.flatnonzero(is_symbol)[-1]) + 1
        expected_by_protein[i, stop_column + 1:] = AFTER_STOP_CELL
        expected_by_protein[i, :width - 1][is_symbol] = CODON_CELL
        expected_by_protein[i, stop_column] = CODON_CELL
    expected = expected_by_protein[proteins.get_indexer(unique_cds['prot_hash'])]

    code_cells = np.full(codes.shape, CODON_CELL, dtype=np.int8)
    code_cells[codes == GAP_CODE] = GAP_CELL
    code_cells[codes == AFTER_STOP_CODE] = AFTER_STOP_CELL
    text_class = {GAP_TEXT: GAP_CELL, AFTER_STOP_TEXT: AFTER_STOP_CELL}
    vocabulary_cells = np.array([text_class.get(text, CODON_CELL) for text in vocabulary],
                                dtype=np.int8)
    text_cells = vocabulary_cells[text_ids]
    for name, observed in (('position check (codes)', code_cells),
                           ('position check (table)', text_cells)):
        failing = (observed != expected).any(axis=1)
        if failing.any():
            fail(name, failing)

    # Row check, codes: the codon cells read in column order are the CDS's own codes.
    for cds_length, group in unique_cds.groupby('cds_length'):
        rows = group.index.to_numpy()
        n_codons = int(cds_length) // 3
        source_codes, _ = encode_codon(sequences_to_byte_matrix(group['cds_dna_seq'].tolist()),
                                       genslm_codes)
        group_codes = codes[rows]
        is_codon = code_cells[rows] == CODON_CELL
        failing = np.zeros(n_cds, dtype=bool)
        failing[rows] = is_codon.sum(axis=1) != n_codons
        if failing.any():
            fail('row check (codes)', failing)
        observed_codes = group_codes[is_codon].reshape(len(group), n_codons)
        failing[rows] = (observed_codes != source_codes).any(axis=1)
        if failing.any():
            fail('row check (codes)', failing)

    # Row check, table: the codon cells joined in column order are the uppercase CDS.
    texts = np.array(vocabulary, dtype=object)
    is_codon_text = vocabulary_cells == CODON_CELL
    failing = np.zeros(n_cds, dtype=bool)
    for row, cds_dna_seq in enumerate(unique_cds['cds_dna_seq']):
        row_ids = text_ids[row]
        joined = ''.join(texts[row_ids[is_codon_text[row_ids]]])
        failing[row] = joined != cds_dna_seq.upper()
    if failing.any():
        fail('row check (table)', failing)
    return n_cds


def write_outputs(paths: dict, codes: np.ndarray, text_ids: np.ndarray, vocabulary: list[str],
                  unique_cds: pd.DataFrame, metadata: dict) -> None:
    """Write the cache files and the readable table of one protein.

    Args:
      paths: from `output_paths`.
      codes: (n CDS, W + 1) uint8.
      text_ids: (n CDS, W + 1) int16 indices into `vocabulary`.
      vocabulary: cell texts.
      unique_cds: one row per CDS, in the row order of `codes`.
      metadata: written as the metadata JSON.

    Returns:
      None.
    """
    np.savez_compressed(paths['.npz'], codes=codes)
    index = pd.DataFrame({'cds_dna_hash': unique_cds['cds_dna_hash'].to_numpy(),
                          'row': np.arange(len(unique_cds))})
    index.to_parquet(paths['_index.parquet'], index=False)

    # One dictionary-encoded text column per site, so 30,000 rows x 760 sites stay small.
    vocabulary_array = pa.array(vocabulary, type=pa.string())
    columns = {'cds_dna_hash': pa.array(unique_cds['cds_dna_hash'].to_numpy(), type=pa.string())}
    for column in range(codes.shape[1]):
        indices = pa.array(np.ascontiguousarray(text_ids[:, column]))
        columns[f'site_{column + 1}'] = pa.DictionaryArray.from_arrays(indices, vocabulary_array)
    pq.write_table(pa.table(columns), paths['_table.parquet'])

    with open(paths['_metadata.json'], 'w') as f:
        json.dump(metadata, f, indent=2)


def build_protein(short: str, function: str, alignment_dir: Path, alignment_dir_setting: str,
                  genslm_codes: dict, config_bundle: str) -> dict:
    """Build, check and write one protein's aligned codon features.

    Args:
      short: short protein name, used in the file names.
      function: full function string.
      alignment_dir: the protein's final alignment directory, resolved.
      alignment_dir_setting: the directory as `site.alignment_dirs` gives it, for the metadata.
      genslm_codes: `{codon: id}` from `load_genslm_codon_codes`.
      config_bundle: recorded in the metadata.

    Returns:
      The metadata that was written.

    Raises:
      ValueError: an input check, the placement or a row check fails.
    """
    print(f'\n{short}: {alignment_dir}')
    unique_cds, aligned, hashes = load_selection(alignment_dir, function)
    alignment_width = len(next(iter(aligned.values())))
    print(f'  {len(unique_cds):,} unique CDS, {len(aligned):,} aligned proteins, '
          f'alignment width {alignment_width}')

    codes, triplets, code_map = place_codons(unique_cds, aligned, genslm_codes)
    if max(code_map.values()) >= GAP_CODE:
        raise ValueError(f'codon codes reach {max(code_map.values())}, which collides with '
                         f'GAP_CODE {GAP_CODE}.')
    text_ids, vocabulary = build_table_vocabulary(triplets)
    n_rows_validated = check_rows(codes, text_ids, vocabulary, unique_cds, aligned, genslm_codes)
    print(f'  row and position checks passed on all {n_rows_validated:,} rows')

    n_gap = int((codes == GAP_CODE).sum())
    n_after_stop = int((codes == AFTER_STOP_CODE).sum())
    n_unk = int((codes == CODON_UNK_CODE).sum())
    print(f'  matrix {codes.shape[0]:,} x {codes.shape[1]:,}: {n_gap:,} GAP cells, '
          f'{n_after_stop:,} AFTER_STOP cells, {n_unk:,} unk cells; '
          f'{len(vocabulary):,} distinct cell texts')

    full_code_map = {str(k): int(v) for k, v in code_map.items()}
    full_code_map[GAP_TEXT] = GAP_CODE
    full_code_map[AFTER_STOP_TEXT] = AFTER_STOP_CODE
    metadata = {
        'unit': UNIT,
        'protein': short,
        'function': function,
        'n_sites': int(codes.shape[1]),
        'n_sequences': int(codes.shape[0]),
        'source_column': 'cds_dna_seq',
        'index_key': 'cds_dna_hash',
        'code_map': full_code_map,
        'other_code': int(CODON_UNK_CODE),
        'n_sites_on_other_code': n_unk,
        'site_mode': 'aligned',
        'alignment_dir': alignment_dir_setting,
        'aligned_fasta_sha256': hashes['aligned_proteins.fasta'],
        'selected_cds_sha256': hashes['selected_cds.parquet'],
        'stop_codon_placement_rule': STOP_CODON_PLACEMENT_RULE,
        'n_rows_validated': n_rows_validated,
        'config_bundle': config_bundle,
        'code': get_git_info(),
        'timestamp': datetime.now().isoformat(),
    }
    paths = output_paths(alignment_dir, short)
    write_outputs(paths, codes, text_ids, vocabulary, unique_cds, metadata)
    print(f'  wrote {aligned_site_cache_stem(UNIT, short)}' + ' / '.join(OUTPUT_SUFFIXES))
    return metadata


def main() -> None:
    total_timer = Timer()
    parser = argparse.ArgumentParser(
        description='Compute per-site codon features placed in protein-alignment columns')
    parser.add_argument('--config_bundle', type=str, required=True,
                        help='Bundle that sets site.mode: aligned and site.alignment_dirs.')
    parser.add_argument('--proteins', type=str, nargs='+', default=None,
                        help='Short names to build; defaults to every protein in '
                             'site.alignment_dirs.')
    parser.add_argument('--force_recompute', action='store_true',
                        help='Replace outputs that already exist.')
    args = parser.parse_args()

    config = get_virus_config_hydra(args.config_bundle, config_path=str(project_root / 'conf'))
    site_cfg = config.get('site') or {}
    if site_cfg.get('mode') != 'aligned':
        raise ValueError(f"{args.config_bundle} sets site.mode={site_cfg.get('mode')!r}; this "
                         f"script builds aligned features and needs site.mode: aligned.")
    alignment_dirs = {str(k): str(v) for k, v in dict(site_cfg['alignment_dirs']).items()}

    proteins = args.proteins or list(alignment_dirs)
    unknown = [p for p in proteins if p not in alignment_dirs]
    if unknown:
        raise ValueError(f'{unknown} are not in site.alignment_dirs of {args.config_bundle}, '
                         f'which has {sorted(alignment_dirs)}.')
    short_to_function = {v: k for k, v in get_function_short_name_map(config).items()}
    missing_function = [p for p in proteins if p not in short_to_function]
    if missing_function:
        raise ValueError(f'{missing_function} are not in the virus config function_short_names.')
    resolved_dirs = {p: project_root / alignment_dirs[p] for p in proteins}

    existing = [str(path) for p in proteins
                for path in output_paths(resolved_dirs[p], p).values() if path.exists()]
    if existing and not args.force_recompute:
        raise FileExistsError(f'Outputs already exist: {existing}; pass --force_recompute to '
                              f'replace them.')

    print(f"\n{'='*40}")
    print('Per-site codon features in protein-alignment columns')
    print(f'Config bundle: {args.config_bundle}')
    print(f'proteins: {proteins}')
    print(f"{'='*40}")

    genslm_codes = load_genslm_codon_codes(GENSLM_TOKENIZER)
    for short in proteins:
        build_protein(short, short_to_function[short], resolved_dirs[short],
                      alignment_dirs[short], genslm_codes, args.config_bundle)

    print(f'\nDone. Finished {Path(__file__).name}.')
    total_timer.display_timer()


if __name__ == '__main__':
    main()
