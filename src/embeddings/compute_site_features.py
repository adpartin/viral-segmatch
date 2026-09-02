"""
Compute per-site features for coding sequences (Stage 2c).

One integer column per position along the CDS, so a feature-importance score points at a place in
the sequence. K-mer counts cannot do that: a count records how many times a subsequence occurs,
not where. Sibling of `compute_kmer_features.py` and `compute_esm2_embeddings.py` in this
directory. See docs/plans/2026-08-28_per_site_nt_features_plan.md step 3.

The unit comes from `site.unit` in the Hydra config (`conf/site/default.yaml`):

  nt     one nucleotide.        Reads cds_dna_final -> cds_dna_seq.  5 codes (ACGT + other).
  codon  one 3-letter group.    Reads cds_dna_seq, in threes.        65 codes (GenSLM ids + unk).
  aa     one amino acid.        Reads cds_dna_final -> prot_seq.     22 codes (20 + stop + other).

Codon and aa cover the SAME positions, because 3 x protein length == CDS length exactly (verified
on every row of cds_dna_final). Codon site i and amino-acid site i are the same place; only the
value differs, so codon keeps silent changes and aa discards them.

One matrix per protein, not one for the corpus: the width is the CDS length, which differs by
protein. Only records that are a complete CDS at the protein's pinned length take part -- position
200 is otherwise a different place in different records. Proteins with no pinned length in
`conf/virus/<virus>.yaml` `cds_length` are skipped, and PB1 and NS1 are absent from that table on
purpose because neither has one length across subtypes and years.

Outputs (per protein and unit, to the embeddings dir alongside the k-mer cache):
  site_features_{unit}_{SHORT}.npz            uint8 codes, one row per unique CDS
  site_features_{unit}_{SHORT}_index.parquet  cds_dna_hash -> row
  site_features_{unit}_{SHORT}_metadata.json  code map, site count, kept/dropped counts

Keyed by `cds_dna_hash` for every unit, including `aa`. Two different CDS that translate to the
same protein therefore get two identical aa rows, which costs a little space and buys one join key
and one row order across all three units.

Every build decodes a sample of rows back to the source sequence and raises on any mismatch, so a
wrong code map cannot reach a model silently.

CPU-only. Existence-check caching per protein; recompute with `--force_recompute`.

Usage:
    python src/embeddings/compute_site_features.py --config_bundle flu_ha_na_h3n2_2024_random_cv4_pinned_length
    python src/embeddings/compute_site_features.py --config_bundle flu_ha_na --unit codon --proteins HA NA
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config_hydra import (  # noqa: E402
    get_function_short_name_map,
    get_virus_config_hydra,
    print_config_summary,
)
from src.utils.path_utils import build_embeddings_paths  # noqa: E402
from src.utils.site_utils import sequences_to_byte_matrix  # noqa: E402
from src.utils.timer_utils import Timer  # noqa: E402

UNITS = ('nt', 'codon', 'aa')

# TODO. Several things are probably defined somewhere in the codebase. We need to define their
# canonical location and import them from there, rather than redefining them here. These could
# include: NT_ALPHABET, NT_OTHER_CODE, AA_ALPHABET, AA_STOP_CODE, AA_OTHER_CODE, CODON_UNK_CODE.

# Nucleotide codes. The values are labels for a categorical column, so the numbering is arbitrary;
# it is fixed here and recorded in the metadata so a cache stays readable. Anything outside ACGT --
# the IUPAC ambiguity codes, which are rare -- shares one "other" code rather than getting 11 more.
NT_ALPHABET = 'ACGT'
NT_OTHER_CODE = len(NT_ALPHABET)

# Amino-acid codes: the 20 standard residues, then the stop character, then everything else (X,
# the "residue unknown" character, lands here).
AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
AA_STOP_CODE = len(AA_ALPHABET)
AA_OTHER_CODE = AA_STOP_CODE + 1

# GenSLM's unknown-token id. A codon holding any non-ACGT character gets this, since it names no
# codon in their vocabulary.
CODON_UNK_CODE = 3

GENSLM_TOKENIZER = project_root / 'genslm_vocab/tokenizer_config.json'

# Draws the decode-check sample. Fixed rather than taken from master_seed, so the check covers
# the same rows on every run and a failure is reproducible from the message alone.
VERIFY_SEED = 0


def load_genslm_codon_codes(tokenizer_config: Path) -> dict:
    """Read the codon-to-integer map out of GenSLM's tokenizer config.

    Their ids are adopted rather than invented so that the same feature cache can later be fed to
    GenSLM without rebuilding it. The order is not alphabetical -- it starts GGC, GCC, ATC, GAC and
    puts the three stops last -- which does not matter for a categorical column but does matter to
    their model.

    Args:
      tokenizer_config: path to `genslm_vocab/tokenizer_config.json`.

    Returns:
      `{codon: id}` for all 64 codons, upper-cased.

    Raises:
      FileNotFoundError: the config is not there.
      ValueError: the config does not hold exactly 64 three-letter ACGT tokens.
    """
    if not tokenizer_config.exists():
        raise FileNotFoundError(
            f"load_genslm_codon_codes: {tokenizer_config} not found. It carries the codon ids "
            f"that site.unit='codon' encodes with.")
    tokens = json.loads(tokenizer_config.read_text())['added_tokens_decoder']
    codes = {}
    for token_id, spec in tokens.items():
        content = str(spec['content']).upper()
        if len(content) == 3 and set(content) <= set(NT_ALPHABET):
            codes[content] = int(token_id)
    if len(codes) != 64:
        raise ValueError(
            f"load_genslm_codon_codes: expected 64 codons in {tokenizer_config}, found "
            f"{len(codes)}.")
    return codes


def _char_lookup_table(alphabet: str, other_code: int, extra: dict = None) -> np.ndarray:
    """Build a 256-entry ASCII-to-code table, so encoding is one array index, not a Python loop.

    Args:
      alphabet: characters taking codes 0, 1, 2, ... in order.
      other_code: code for every character not otherwise assigned.
      extra: additional `{character: code}` entries, applied after the alphabet.

    Returns:
      uint8 array of length 256.
    """
    table = np.full(256, other_code, dtype=np.uint8)
    for code, char in enumerate(alphabet):
        table[ord(char)] = code
    for char, code in (extra or {}).items():
        table[ord(char)] = code
    return table


def encode_nt(byte_matrix: np.ndarray) -> tuple[np.ndarray, dict]:
    """Encode a nucleotide byte matrix to one code per position.

    Args:
      byte_matrix: (n, L) upper-cased ASCII bytes of CDS DNA.

    Returns:
      `(codes, code_map)`. `codes` is (n, L) uint8; `code_map` is `{character: code}` including
      the catch-all `'other'` entry.
    """
    table = _char_lookup_table(NT_ALPHABET, NT_OTHER_CODE)
    codes = table[byte_matrix]
    code_map = {char: code for code, char in enumerate(NT_ALPHABET)}
    code_map['other'] = NT_OTHER_CODE
    return codes, code_map


def encode_aa(byte_matrix: np.ndarray) -> tuple[np.ndarray, dict]:
    """Encode a protein byte matrix to one code per residue.

    Args:
      byte_matrix: (n, L) upper-cased ASCII bytes of protein sequence, stop character included.

    Returns:
      `(codes, code_map)`. `codes` is (n, L) uint8; `code_map` is `{character: code}` with `'*'`
      for the stop and `'other'` for anything else (X, the unknown-residue character, lands there).
    """
    table = _char_lookup_table(AA_ALPHABET, AA_OTHER_CODE, extra={'*': AA_STOP_CODE})
    codes = table[byte_matrix]
    code_map = {char: code for code, char in enumerate(AA_ALPHABET)}
    code_map['*'] = AA_STOP_CODE
    code_map['other'] = AA_OTHER_CODE
    return codes, code_map


def encode_codon(byte_matrix: np.ndarray, genslm_codes: dict) -> tuple[np.ndarray, dict]:
    """Encode a nucleotide byte matrix to one GenSLM codon id per 3-letter group.

    Goes through the nucleotide codes rather than slicing strings: a codon whose three bases are
    all ACGT has a base-4 index, which indexes a 64-entry table of GenSLM ids. Any codon holding a
    non-ACGT character cannot name a codon and gets `<unk>`.

    Args:
      byte_matrix: (n, L) upper-cased ASCII bytes of CDS DNA; L must be a multiple of 3.
      genslm_codes: `{codon: id}` from `load_genslm_codon_codes`.

    Returns:
      `(codes, code_map)`. `codes` is (n, L/3) uint8; `code_map` is `{codon: id}` plus `'unk'`.

    Raises:
      ValueError: L is not a multiple of 3.
    """
    n, length = byte_matrix.shape
    if length % 3 != 0:
        raise ValueError(
            f"encode_codon: CDS length {length} is not a multiple of 3, so it does not divide "
            f"into codons.")
    nt_codes, _ = encode_nt(byte_matrix)
    triples = nt_codes.reshape(n, length // 3, 3).astype(np.int16)

    # base-4 index with A=0, C=1, G=2, T=3; 4 marks a non-ACGT base, so any codon holding one
    # exceeds the table and is sent to <unk>.
    base4 = triples[:, :, 0] * 16 + triples[:, :, 1] * 4 + triples[:, :, 2]
    resolvable = (triples < len(NT_ALPHABET)).all(axis=2)

    base4_to_genslm = np.full(64, CODON_UNK_CODE, dtype=np.uint8)
    for codon, code in genslm_codes.items():
        index = sum(NT_ALPHABET.index(base) * 4 ** (2 - i) for i, base in enumerate(codon))
        base4_to_genslm[index] = code

    codes = np.full((n, length // 3), CODON_UNK_CODE, dtype=np.uint8)
    codes[resolvable] = base4_to_genslm[base4[resolvable]]
    code_map = dict(genslm_codes)
    code_map['unk'] = CODON_UNK_CODE
    return codes, code_map


def verify_decodes(codes: np.ndarray, sequences: list[str], code_map: dict, unit: str,
                   other_code: int, n_verify: int) -> int:
    """Decode a sample of rows back to their source sequence and raise on any mismatch.

    A wrong code map would otherwise reach a model as plausible-looking integers. Characters that
    the encoding folds into one catch-all code cannot round-trip, so those positions are checked
    the other way: the source character there must be one the map does not name.

    Args:
      codes: (n, n_sites) encoded matrix.
      sequences: the source sequences the matrix was built from, same row order.
      code_map: `{symbol: code}` used to encode, including the catch-all entry.
      unit: `nt`, `codon` or `aa`; sets how many source characters one site spans.
      other_code: the catch-all code, whose positions are checked rather than decoded.
      n_verify: how many rows to check.

    Returns:
      Number of rows checked.

    Raises:
        ValueError: a decoded site disagrees with the source, or a catch-all code sits at a
            position whose source symbol the map does name.
    """
    span = 3 if unit == 'codon' else 1
    decode = {code: symbol for symbol, code in code_map.items()
              if code != other_code and len(symbol) == span}
    named = set(decode.values())

    rng = np.random.default_rng(VERIFY_SEED)
    rows = rng.choice(len(sequences), size=min(n_verify, len(sequences)), replace=False)
    for row in rows:
        source = sequences[int(row)].upper()
        for site in range(codes.shape[1]):
            symbol = source[site * span:(site + 1) * span]
            code = int(codes[row, site])
            if code == other_code:
                if symbol in named:
                    raise ValueError(
                        f"verify_decodes: row {row} site {site} encoded as the catch-all code "
                        f"{other_code}, but the source symbol {symbol!r} is one the {unit} map "
                        f"names.")
                continue
            if decode[code] != symbol:
                raise ValueError(
                    f"verify_decodes: row {row} site {site} decodes to {decode[code]!r} but the "
                    f"source is {symbol!r} ({unit} codes are wrong).")
    return len(rows)


def build_protein(cds: pd.DataFrame, short: str, function: str, pinned_nt: int, unit: str,
                  genslm_codes: dict, output_dir: Path, input_file: Path,
                  n_verify: int) -> dict:
    """Encode one protein's CDS population and write its three cache files.

    Args:
      cds: full cds_dna_final frame; narrowed to `function` here.
      short: short protein name, used in the file names.
      function: full function string to select on.
      pinned_nt: the protein's pinned CDS length in nucleotides.
      unit: `nt`, `codon` or `aa`.
      genslm_codes: `{codon: id}`, used only when `unit == 'codon'`.
      output_dir: where the three files go.
      input_file: recorded in the metadata for provenance.
      n_verify: rows to decode-check.

    Returns:
      The metadata dict that was written.

    Raises:
      ValueError: no record survives the completeness and length filter.
    """
    of_function = cds[cds['function'] == function]
    unique_cds = of_function.drop_duplicates('cds_dna_hash')
    complete = unique_cds[unique_cds['is_complete_cds']]
    kept = complete[complete['cds_length'] == pinned_nt].reset_index(drop=True)
    if kept.empty:
        raise ValueError(
            f"{short}: no complete CDS at the pinned length {pinned_nt} nt among "
            f"{len(unique_cds):,} unique CDS. Check conf/virus/<virus>.yaml cds_length.")

    print(f"\n{short} ({unit}): {len(unique_cds):,} unique CDS -> complete {len(complete):,} "
          f"-> at {pinned_nt} nt {len(kept):,}")

    # aa reads the protein, the other two read the DNA; the RECORDS are the same either way,
    # because the filter is on the CDS. So codon site i and aa site i are the same place.
    source_col = 'prot_seq' if unit == 'aa' else 'cds_dna_seq'
    sequences = kept[source_col].tolist()
    byte_matrix = sequences_to_byte_matrix(sequences)

    if unit == 'nt':
        codes, code_map = encode_nt(byte_matrix)
        other_code = NT_OTHER_CODE
    elif unit == 'aa':
        codes, code_map = encode_aa(byte_matrix)
        other_code = AA_OTHER_CODE
    else:
        codes, code_map = encode_codon(byte_matrix, genslm_codes)
        other_code = CODON_UNK_CODE

    n_checked = verify_decodes(codes, sequences, code_map, unit, other_code, n_verify)
    n_other = int((codes == other_code).sum())
    print(f"  matrix {codes.shape[0]:,} x {codes.shape[1]:,} uint8 "
          f"({codes.nbytes / 1e6:.1f} MB in memory); {n_other:,} sites on the catch-all code "
          f"({100 * n_other / codes.size:.4f}%)")
    print(f"  decode check passed on {n_checked} rows")

    stem = f'site_features_{unit}_{short}'
    np.savez_compressed(output_dir / f'{stem}.npz', codes=codes)
    index = pd.DataFrame({'cds_dna_hash': kept['cds_dna_hash'].values,
                          'row': np.arange(len(kept))})
    index.to_parquet(output_dir / f'{stem}_index.parquet', index=False)

    metadata = {
        'unit': unit,
        'protein': short,
        'function': function,
        'pinned_nt': pinned_nt,
        'n_sites': int(codes.shape[1]),
        'n_sequences': int(codes.shape[0]),
        'source_column': source_col,
        'index_key': 'cds_dna_hash',
        'code_map': {str(k): int(v) for k, v in code_map.items()},
        'other_code': int(other_code),
        'n_sites_on_other_code': n_other,
        'n_unique_cds': int(len(unique_cds)),
        'n_dropped_incomplete': int(len(unique_cds) - len(complete)),
        'n_dropped_off_pinned_length': int(len(complete) - len(kept)),
        'n_rows_decode_checked': int(n_checked),
        'input_file': str(input_file),
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / f'{stem}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  wrote {stem}.npz / _index.parquet / _metadata.json")
    return metadata


def main() -> None:
    total_timer = Timer()
    parser = argparse.ArgumentParser(description='Compute per-site features for coding sequences')
    parser.add_argument('--config_bundle', type=str, required=True,
                        help='Config bundle to use (e.g., flu_ha_na).')
    parser.add_argument('--unit', type=str, default=None, choices=list(UNITS),
                        help='Site unit; defaults to site.unit from the bundle.')
    parser.add_argument('--proteins', type=str, nargs='+', default=None,
                        help='Short names to build; defaults to every protein with a pinned '
                             'cds_length in the virus config.')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Path to cds_dna_final.parquet. Derived from config if absent.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output dir. Derived from config if absent.')
    parser.add_argument('--n_verify', type=int, default=25,
                        help='Rows per protein to decode back to the source sequence.')
    parser.add_argument('--force_recompute', action='store_true',
                        help='Rebuild proteins that are already cached.')
    args = parser.parse_args()

    config = get_virus_config_hydra(args.config_bundle, config_path=str(project_root / 'conf'))
    print_config_summary(config)

    virus_name = config.virus.virus_name
    data_version = config.virus.data_version
    site_cfg = config.get('site') or {}
    unit = args.unit or str(site_cfg.get('unit', 'nt'))
    if unit not in UNITS:
        raise ValueError(f"site.unit must be one of {list(UNITS)}; got {unit!r}.")

    pinned_lengths = getattr(config.virus, 'cds_length', None)
    if pinned_lengths is None:
        raise ValueError(
            f"conf/virus/{virus_name}.yaml has no `cds_length` block. Per-site features need one "
            f"pinned length per protein, since every sequence must sit at the same length.")
    pinned_lengths = {str(k): int(v['nt']) for k, v in dict(pinned_lengths).items()}

    short_to_function = {v: k for k, v in get_function_short_name_map(config).items()}
    proteins = args.proteins or sorted(pinned_lengths)
    unknown = [p for p in proteins if p not in pinned_lengths]
    if unknown:
        raise ValueError(
            f"no cds_length pinned for {unknown}. conf/virus/{virus_name}.yaml cds_length has "
            f"{sorted(pinned_lengths)}; PB1 and NS1 are absent on purpose, because neither has "
            f"one length across subtypes and years.")

    paths = build_embeddings_paths(project_root=project_root, virus_name=virus_name,
                                   data_version=data_version, run_suffix="", config=config)
    input_file = Path(args.input_file) if args.input_file else \
        paths['input_file'].parent / 'cds_dna_final.parquet'
    output_dir = Path(args.output_dir) if args.output_dir else paths['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*40}")
    print("Stage 2c: Per-site features")
    print(f"Virus: {virus_name}")
    print(f"Config bundle: {args.config_bundle}")
    print(f"site.unit: {unit}")
    print(f"proteins: {proteins}")
    print(f"{'='*40}")
    print(f"\ninput_file: {input_file}")
    print(f"output_dir: {output_dir}")

    to_build = list(proteins)
    if not args.force_recompute:
        cached = [p for p in to_build
                  if all((output_dir / f'site_features_{unit}_{p}{suffix}').exists()
                         for suffix in ('.npz', '_index.parquet', '_metadata.json'))]
        if cached:
            print(f"\nAlready cached (use --force_recompute to rebuild): {cached}")
            to_build = [p for p in to_build if p not in cached]
    if not to_build:
        print("\nNothing to build.")
        return

    print(f"\nLoad source data from: {input_file}")
    required = ['function', 'cds_dna_hash', 'cds_dna_seq', 'prot_seq', 'cds_length',
                'is_complete_cds']
    available = set(pq.ParquetFile(input_file).schema_arrow.names)
    missing = [c for c in required if c not in available]
    if missing:
        raise ValueError(
            f"{input_file} is missing {missing}. `is_complete_cds` is written by Stage 1.5; "
            f"rebuild with `python src/preprocess/extract_cds_dna.py`.")
    cds = pd.read_parquet(input_file, columns=required)
    print(f"Loaded {len(cds):,} records")

    genslm_codes = load_genslm_codon_codes(GENSLM_TOKENIZER) if unit == 'codon' else {}

    for short in to_build:
        function = short_to_function.get(short)
        if function is None:
            raise ValueError(
                f"{short!r} is not in conf/virus/{virus_name}.yaml function_short_names.")
        build_protein(cds, short, function, pinned_lengths[short], unit, genslm_codes,
                      output_dir, input_file, args.n_verify)

    print(f"\nDone. Finished {Path(__file__).name}.")
    total_timer.display_timer()


if __name__ == '__main__':
    main()
