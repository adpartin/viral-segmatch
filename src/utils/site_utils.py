"""Per-site feature primitives: build the site matrix, read the cache, make pair features.

Sibling of `kmer_utils.py`, for the cache `src/embeddings/compute_site_features.py` writes: one
uint8 matrix per (protein, site unit), one row per unique CDS, keyed by `cds_dna_hash`. The aligned
codon caches that `src/embeddings/compute_protein_aligned_codon_features.py` writes beside each
protein alignment have the same layout and are read by `read_aligned_codon_cache`.

`sequences_to_byte_matrix` and `column_entropy` live here rather than in either caller because
three places need them -- the builder, the entropy map and the importance map -- and per-site
entropy must mean one thing across all of them.

Per-site features are the one feature source where slot A and slot B live in different spaces --
slot A is HA position 1..1701, slot B is NA position 1..1410. HA position 500 and NA position 500
have nothing to do with each other, and the two are not even the same length. So `concat` is the
only interaction that means anything here, and there is nothing for a slot transform to normalise:
a category code is a label, not a magnitude. `_pair_features.load_pair_features_for_baselines`
enforces both.

Column layout is fixed and derivable without reading the matrices, which is what lets a
feature-importance score be traced back to a position -- see `site_feature_columns`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from src.utils.experiment_utils import file_sha256
from src.utils.path_utils import build_embeddings_paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENCODINGS = ('ordinal', 'onehot')
# Where a site's position comes from: a position along a CDS at the pinned length, or a column of
# the protein alignment.
SITE_MODES = ('pinned', 'aligned')
SITE_CACHE_SUFFIXES = ('.npz', '_index.parquet', '_metadata.json')
# Which slots' columns end up in the pair matrix. 'both' is the real feature set; 'a' and 'b'
# are the one-side ablation, which should score near chance -- the label is a fact about a
# PAIR, and the same sequence appears on both sides of it.
SLOT_CHOICES = ('both', 'a', 'b')

# The two aligned-cache cells that hold no codon, with their text in the readable table and the
# metadata's code_map, and their code in the matrix. Both codes lie above every GenSLM codon id
# (0-96) and fit in uint8. The writer adds both to code_map, so every code in the matrix is
# declared there.
GAP_TEXT = 'GAP'
AFTER_STOP_TEXT = 'AFTER_STOP'
GAP_CODE = 97
AFTER_STOP_CODE = 98


def sequences_to_byte_matrix(sequences: list[str]) -> np.ndarray:
    """Stack equal-length sequences into an (n, L) array of upper-cased ASCII bytes.

    Args:
      sequences: equal-length sequences.

    Returns:
      uint8 array of shape (n sequences, L characters).

    Raises:
      ValueError: the list is empty or the sequences are not all one length.
    """
    if not sequences:
        raise ValueError("sequences_to_byte_matrix: no sequences given.")
    lengths = {len(s) for s in sequences}
    if len(lengths) != 1:
        raise ValueError(
            f"sequences_to_byte_matrix: {len(lengths)} different lengths {sorted(lengths)}; "
            f"per-site features need one length.")
    flat = ''.join(sequences).upper().encode('ascii', 'replace')
    arr = np.frombuffer(flat, dtype=np.uint8).reshape(len(sequences), lengths.pop())
    return arr


def column_entropy(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shannon entropy in bits down each column of a matrix of small integer values.

    Works on either representation of a site matrix -- ASCII bytes from
    `sequences_to_byte_matrix`, or the integer codes from the feature cache -- so the entropy of
    a position means the same thing wherever it is computed. Note the unit sets the ceiling: 2
    bits for 4 nucleotides, 6 for 64 codons, so entropies are comparable within a unit and not
    across units.

    Args:
      matrix: (n rows, L columns) array of values in 0..255. Rows should be unique sequences, not
          occurrences; a heavily sampled strain would otherwise decide the answer.

    Returns:
      `(entropy_bits, n_values)`, both length L. `n_values` is how many distinct values appear in
      that column, which separates "one rare variant" from "genuinely mixed".

    Raises:
      ValueError: the matrix has no rows.
    """
    if matrix.shape[0] == 0:
        raise ValueError("column_entropy: matrix has no rows.")
    n = matrix.shape[0]
    # One boolean pass per observed value. Site alphabets are small -- 5 codes for nt, 65 for
    # codon -- so this is a handful of passes, not 256.
    values = np.unique(matrix)
    counts = np.stack([(matrix == v).sum(axis=0) for v in values]).astype(np.float64)
    proportions = counts / n

    # Shannon entropy term per value: -p * log2(p), with the standard convention that a value
    # that never appears (p=0) contributes 0, not NaN. `where=has_proportion` on log2 also stops
    # numpy warning about log2(0), since that entry is overwritten by np.where regardless.
    has_proportion = proportions > 0
    log2_proportions = np.log2(proportions, where=has_proportion)
    terms = np.where(has_proportion, -proportions * log2_proportions, 0.0)

    entropy_bits = terms.sum(axis=0)
    n_values = (counts > 0).sum(axis=0)
    return entropy_bits, n_values


class SiteCache(NamedTuple):
    """One protein's site codes plus what is needed to place and read them.

    Attributes:
      protein: short protein name, e.g. `HA`.
      unit: `nt`, `codon` or `aa`.
      codes: (n unique CDS, n sites) uint8 matrix.
      hash_to_row: `cds_dna_hash` -> row index into `codes`.
      metadata: the cache's `_metadata.json`, which carries the code map.
    """
    protein: str
    unit: str
    codes: np.ndarray
    hash_to_row: dict
    metadata: dict


def site_cache_stem(unit: str, protein: str, pinned_nt: int) -> str:
    """The file stem of one protein's per-site feature cache at one pinned length.

    The pin is part of the name because it decides the contents: a cache holds one column per
    site of a CDS of exactly `pinned_nt` nucleotides, and the same protein takes a different pin
    in a different population. Human H3N2 PB1 is 2,277 nt from 2024 while every other population
    measured is 2,274 nt, so the two caches have to be able to sit side by side.

    Args:
      unit: `nt`, `codon` or `aa`.
      protein: short protein name, e.g. `HA`.
      pinned_nt: the pinned CDS length the cache was built at.

    Returns:
      The stem, e.g. `site_features_codon_PB1_2274nt`.
    """
    return f'site_features_{unit}_{protein}_{pinned_nt}nt'


def aligned_site_cache_stem(unit: str, protein: str) -> str:
    """The file stem of one protein's per-site feature cache whose sites are alignment columns.

    The name carries no pin, because the protein alignment, not a CDS length, sets the columns.
    The cache is written beside the alignment it comes from, so its directory says which
    alignment.

    Args:
      unit: site unit; `src/embeddings/compute_protein_aligned_codon_features.py` writes `codon`.
      protein: short protein name, e.g. `HA`.

    Returns:
      The stem, e.g. `site_features_codon_HA_aligned`.
    """
    stem = f'site_features_{unit}_{protein}_aligned'
    return stem


def site_mode(config) -> str:
    """The run's `site.mode`, or `pinned` when the config does not set it.

    Args:
      config: run configuration. Configs written before aligned caches existed have no mode.

    Returns:
      `pinned` or `aligned`.

    Raises:
      ValueError: the mode is neither.
    """
    mode = str((config.get('site') or {}).get('mode', 'pinned'))
    if mode not in SITE_MODES:
        raise ValueError(f'site.mode must be one of {list(SITE_MODES)}; got {mode!r}.')
    return mode


def require_pinned_site_mode(config, caller: str) -> None:
    """Raise if a pinned-only caller receives a different site mode.

    Args:
      config: run configuration. Older configs omit the mode.
      caller: name to include in the error.

    Raises:
      ValueError: the mode is not `pinned`.
    """
    mode = site_mode(config)
    if mode != 'pinned':
        raise ValueError(f'{caller} supports pinned caches only. Got site.mode={mode!r}.')


def site_cache_paths(site_dir: Path, unit: str, protein: str, pinned_nt: int) -> dict:
    """The three files of one protein's per-site feature cache.

    Args:
      site_dir: directory holding the cache, normally the embeddings output dir.
      unit: `nt`, `codon` or `aa`.
      protein: short protein name, e.g. `HA`.
      pinned_nt: the pinned CDS length the cache was built at.

    Returns:
      Suffix (`.npz`, `_index.parquet`, `_metadata.json`) -> path.
    """
    stem = site_cache_stem(unit, protein, pinned_nt)
    paths = {suffix: Path(site_dir) / f'{stem}{suffix}' for suffix in SITE_CACHE_SUFFIXES}
    return paths


def resolve_site_cache_pin(site_dir: Path, unit: str, protein: str) -> int:
    """The pinned length of the one cache on disk for this protein and unit.

    For callers that have no config to read a pin from, such as the exploratory plotting scripts.
    A protein with caches at two pins has no single answer, so this says which exist rather than
    choosing one.

    Args:
      site_dir: directory holding the cache.
      unit: `nt`, `codon` or `aa`.
      protein: short protein name, e.g. `HA`.

    Returns:
      The pinned length.

    Raises:
      FileNotFoundError: no cache for this protein and unit.
      ValueError: caches at more than one pin, so the caller has to name one.
    """
    pattern = f'site_features_{unit}_{protein}_*nt_metadata.json'
    found = sorted(Path(site_dir).glob(pattern))
    pins = sorted(int(path.name.rsplit('_', 2)[-2][:-2]) for path in found)
    if not pins:
        raise FileNotFoundError(
            f"no per-site cache for {protein} ({unit}) in {site_dir}; looked for {pattern}. "
            f"Build it with `python src/embeddings/compute_site_features.py --config_bundle "
            f"<bundle> --unit {unit} --proteins {protein}`.")
    if len(pins) > 1:
        raise ValueError(
            f"{protein} ({unit}) has caches at {pins} nt in {site_dir}; pass the pinned length "
            f"to say which one to read.")
    return pins[0]


def load_site_cache(site_dir: Path, unit: str, protein: str,
                    pinned_nt: Optional[int] = None) -> SiteCache:
    """Load one protein's per-site feature cache.

    Args:
      site_dir: directory holding the cache, normally the embeddings output dir.
      unit: `nt`, `codon` or `aa`.
      protein: short protein name, e.g. `HA`.
      pinned_nt: the pinned CDS length to read. None takes the only cache on disk for this
          protein and unit, and raises when there are several.

    Returns:
      The `SiteCache` for that protein, unit and pin.

    Raises:
      FileNotFoundError: any of the three cache files is missing.
      ValueError: the directory contains an aligned cache for this protein and unit,
          the matrix and the index disagree on how many rows there are, the cache's own
          metadata records a different pin than its name, or `pinned_nt` is None and the protein
          has caches at several pins.
    """
    site_dir = Path(site_dir)
    aligned_metadata = site_dir / f'{aligned_site_cache_stem(unit, protein)}_metadata.json'
    # Some pinned-only callers request nucleotide features even in an aligned codon directory.
    if aligned_metadata.is_file():
        raise ValueError(f'load_site_cache supports pinned caches only. Found {aligned_metadata}.')
    if unit != 'codon':
        aligned_codon = site_dir / f'{aligned_site_cache_stem("codon", protein)}_metadata.json'
        pinned_pattern = f'site_features_{unit}_{protein}_*nt_metadata.json'
        if aligned_codon.is_file() and not any(site_dir.glob(pinned_pattern)):
            raise ValueError(f'load_site_cache supports pinned caches only. Found {aligned_codon}.')
    if pinned_nt is None:
        pinned_nt = resolve_site_cache_pin(site_dir, unit, protein)
    stem = site_cache_stem(unit, protein, pinned_nt)
    paths = site_cache_paths(site_dir, unit, protein, pinned_nt)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        on_disk = sorted(int(path.name.rsplit('_', 2)[-2][:-2])
                         for path in Path(site_dir).glob(
                             f'site_features_{unit}_{protein}_*nt_metadata.json'))
        have = f" Caches on disk for {protein} ({unit}): {on_disk} nt." if on_disk else ''
        raise FileNotFoundError(
            f"per-site cache for {protein} ({unit}) at {pinned_nt} nt is incomplete; missing "
            f"{missing}.{have} Build it with `python src/embeddings/compute_site_features.py "
            f"--config_bundle <bundle> --unit {unit} --proteins {protein}`.")

    metadata = json.loads(paths['_metadata.json'].read_text())
    mode = metadata.get('site_mode', 'pinned')
    if mode != 'pinned':
        raise ValueError(f'load_site_cache supports pinned caches only. '
                         f'{paths["_metadata.json"]} declares site_mode={mode!r}.')
    codes = np.load(paths['.npz'])['codes']
    index = pd.read_parquet(paths['_index.parquet'])
    if len(index) != codes.shape[0]:
        raise ValueError(
            f"{stem}: index has {len(index):,} rows but the matrix has {codes.shape[0]:,}. The "
            f"cache is inconsistent; rebuild it with --force_recompute.")
    if int(metadata['pinned_nt']) != int(pinned_nt):
        raise ValueError(
            f"{stem}: the file name says {pinned_nt} nt but its metadata records "
            f"{metadata['pinned_nt']} nt. Rebuild it with --force_recompute.")
    hash_to_row = dict(zip(index['cds_dna_hash'], index['row'].astype(int)))
    return SiteCache(protein=protein, unit=unit, codes=codes, hash_to_row=hash_to_row,
                     metadata=metadata)


def read_site_cache_index(stem: Path, n_rows: int) -> pd.Series:
    """Read a site cache's hash-to-row mapping and check that it covers each matrix row once.

    Args:
      stem: the cache's path without its suffix.
      n_rows: number of rows in the cache's code matrix.

    Returns:
      Row number per `cds_dna_hash`, indexed by the hash.

    Raises:
      ValueError: a hash repeats or is missing, the row numbers are not integers, or they are not
          exactly 0 to `n_rows - 1`.
    """
    index = pd.read_parquet(f'{stem}_index.parquet', columns=['cds_dna_hash', 'row'])
    rows = index.set_index('cds_dna_hash', verify_integrity=True)['row']
    if (rows.index.hasnans or not pd.api.types.is_integer_dtype(rows)
            or not np.array_equal(np.sort(rows.to_numpy()), np.arange(n_rows))):
        raise ValueError(f'{stem}: invalid hash-to-row index')
    return rows


def read_aligned_codon_cache(directory: Path, protein: str) -> SiteCache:
    """Read one protein's aligned codon cache and check it against the alignment beside it.

    The metadata must name an aligned codon cache of `protein` keyed by `cds_dna_hash`, with GAP
    and AFTER_STOP at their fixed codes. The matrix must be uint8 at the shape the metadata
    records, with every code declared in the code map. Its width must be the alignment width W
    from `alignment_validation.json` plus the extra column after the alignment. The aligned FASTA
    and the selected CDS on disk must have the sha256 the cache recorded when it was built, and
    the validation report must describe that FASTA.

    Args:
      directory: the protein's final alignment directory, which holds the cache.
      protein: short protein name, e.g. `HA`.

    Returns:
      The `SiteCache`, with unit `codon`.

    Raises:
      FileNotFoundError: a cache file, the validation report or a source file is missing.
      KeyError: the validation report has no `aligned_width`.
      ValueError: any of the checks above fails, or the index does not cover each matrix row once.
    """
    directory = Path(directory)
    stem = directory / aligned_site_cache_stem('codon', protein)
    metadata = json.loads(Path(f'{stem}_metadata.json').read_text())
    identity = {'site_mode': 'aligned', 'unit': 'codon', 'protein': protein,
                'index_key': 'cds_dna_hash'}
    for field, expected in identity.items():
        if metadata.get(field) != expected:
            raise ValueError(f'{stem}: expected {field}={expected!r}, got {metadata.get(field)!r}')

    code_map = metadata.get('code_map')
    fixed_codes = {GAP_TEXT: GAP_CODE, AFTER_STOP_TEXT: AFTER_STOP_CODE}
    if not isinstance(code_map, dict) or any(code_map.get(text) != code
                                             for text, code in fixed_codes.items()):
        raise ValueError(f'{stem}: expected {GAP_TEXT}={GAP_CODE} and '
                         f'{AFTER_STOP_TEXT}={AFTER_STOP_CODE} in code_map')
    ids = list(code_map.values())
    if any(type(code) is not int or not 0 <= code <= 255 for code in ids) or len(set(ids)) != len(ids):
        raise ValueError(f'{stem}: code_map must contain distinct uint8 codes')

    with np.load(f'{stem}.npz', allow_pickle=False) as archive:
        codes = archive['codes']
    shape = (metadata.get('n_sequences'), metadata.get('n_sites'))
    if codes.dtype != np.uint8 or codes.shape != shape:
        raise ValueError(f'{stem}: expected uint8 matrix of shape {shape}, got {codes.dtype} {codes.shape}')
    declared = np.zeros(256, dtype=bool)
    declared[ids] = True
    if not declared[codes].all():
        raise ValueError(f'{stem}: matrix contains a code missing from code_map')

    # The source files must still hold the bytes the cache was built from.
    sources = {'aligned_fasta_sha256': 'aligned_proteins.fasta',
               'selected_cds_sha256': 'selected_cds.parquet'}
    current = {field: file_sha256(directory / name) for field, name in sources.items()}
    for field, name in sources.items():
        if metadata.get(field) != current[field]:
            raise ValueError(f'{stem}: {name} has sha256 {current[field]}, but the cache was built '
                             f'from {metadata.get(field)}. Rebuild the cache.')

    # The width is the alignment's W columns plus the extra column after the last one. W comes
    # from the validation report, so that report must describe the current FASTA.
    validation = json.loads((directory / 'alignment_validation.json').read_text())
    if validation.get('aligned_fasta_sha256') != current['aligned_fasta_sha256']:
        raise ValueError(f'{directory}: alignment_validation.json describes a different '
                         f'aligned_proteins.fasta')
    expected_width = int(validation['aligned_width']) + 1
    if codes.shape[1] != expected_width:
        raise ValueError(f'{stem}: {codes.shape[1]} sites, but the alignment has '
                         f'{expected_width - 1} columns, so the width must be {expected_width}')

    rows = read_site_cache_index(stem, len(codes))
    hash_to_row = dict(zip(rows.index, rows.astype(int)))
    cache = SiteCache(protein=protein, unit='codon', codes=codes, hash_to_row=hash_to_row,
                      metadata=metadata)
    return cache


def _configured_alignment_dir(config, protein: str) -> Path:
    """The protein's final alignment directory from `site.alignment_dirs`, under the project root.

    Args:
      config: run configuration in aligned mode.
      protein: short protein name, e.g. `HA`.

    Returns:
      The directory. An absolute entry is returned unchanged.

    Raises:
      ValueError: `site.alignment_dirs` has no entry for the protein.
    """
    alignment_dirs = dict(config.site.get('alignment_dirs') or {})
    if protein not in alignment_dirs:
        raise ValueError(f'site.alignment_dirs has no entry for {protein}; it has '
                         f'{sorted(alignment_dirs)}.')
    directory = PROJECT_ROOT / str(alignment_dirs[protein])
    return directory


def _configured_pinned_cache(config, protein: str) -> tuple[Path, int]:
    """The embeddings directory and the protein's pinned length in `virus.cds_length`.

    Args:
      config: run configuration in pinned mode.
      protein: short protein name, e.g. `HA`.

    Returns:
      `(site_dir, pinned_nt)`.

    Raises:
      ValueError: `virus.cds_length` pins no length for the protein.
    """
    site_dir = build_embeddings_paths(
        project_root=PROJECT_ROOT, virus_name=config.virus.virus_name,
        data_version=config.virus.data_version, run_suffix='', config=config)['output_dir']
    pins = dict(config.virus.get('cds_length') or {})
    if protein not in pins:
        raise ValueError(f'virus.cds_length pins no length for {protein}; it has {sorted(pins)}.')
    pinned_nt = int(pins[protein]['nt'])
    return site_dir, pinned_nt


def configured_site_cache_paths(config, protein: str) -> dict:
    """The three files of the site cache that the run's config selects for one protein.

    In pinned mode the cache is in the embeddings directory and is named by the protein's pinned
    length in `virus.cds_length`. In aligned mode it is the codon cache in the protein's
    `site.alignment_dirs` directory.

    Args:
      config: run configuration.
      protein: short protein name, e.g. `HA`.

    Returns:
      Suffix (`.npz`, `_index.parquet`, `_metadata.json`) -> path.

    Raises:
      ValueError: `site.mode` is not recognised, or the protein has no pinned length (pinned
          mode) or no alignment directory (aligned mode).
    """
    if site_mode(config) == 'aligned':
        directory = _configured_alignment_dir(config, protein)
        stem = aligned_site_cache_stem('codon', protein)
        paths = {suffix: directory / f'{stem}{suffix}' for suffix in SITE_CACHE_SUFFIXES}
    else:
        site_dir, pinned_nt = _configured_pinned_cache(config, protein)
        paths = site_cache_paths(site_dir, str(config.site.unit), protein, pinned_nt)
    return paths


def load_configured_site_cache(config, protein: str) -> SiteCache:
    """Load the site cache that the run's config selects for one protein.

    Pinned mode reads it with `load_site_cache`. Aligned mode reads it with
    `read_aligned_codon_cache`, which also checks it against the alignment beside it. Both read
    the files `configured_site_cache_paths` names.

    Args:
      config: run configuration.
      protein: short protein name, e.g. `HA`.

    Returns:
      The protein's `SiteCache`.

    Raises:
      ValueError: `site.mode` is not recognised, the protein has no pinned length or alignment
          directory, or the cache fails its reader's checks.
      FileNotFoundError: a cache file is missing.
    """
    if site_mode(config) == 'aligned':
        cache = read_aligned_codon_cache(_configured_alignment_dir(config, protein), protein)
    else:
        site_dir, pinned_nt = _configured_pinned_cache(config, protein)
        cache = load_site_cache(site_dir, str(config.site.unit), protein, pinned_nt)
    return cache


def declared_codes(cache: SiteCache) -> list[int]:
    """The full set of codes a unit can produce, sorted, from the cache metadata.

    One-hot width comes from this rather than from the values a split happens to contain, so
    train, val and test get identical columns even when a rare code appears in only one of them.

    Args:
      cache: the cache whose metadata declares the code map.

    Returns:
      Sorted unique code values.
    """
    return sorted({int(v) for v in cache.metadata['code_map'].values()})


def site_feature_columns(cache_a: SiteCache, cache_b: SiteCache, encoding: str,
                         slots: str = 'both') -> pd.DataFrame:
    """Describe every output column, so an importance score can name a position.

    The layout is fixed: slot A's sites in order, then slot B's. Under `ordinal` that is one
    column per site; under `onehot` it is one column per (site, code), with the codes in the
    sorted order `declared_codes` returns.

    Args:
      cache_a: slot A's cache.
      cache_b: slot B's cache.
      encoding: `ordinal` or `onehot`.
      slots: which slots contribute columns -- `both`, or `a` / `b` for the one-side ablation.

    Returns:
      One row per column with `column`, `slot`, `protein` and 1-based `site`. For pinned
      caches, the site is a position along the CDS in the cache's unit. For aligned caches,
      it is a protein-alignment column or the extra column after the alignment. Under
      `onehot`, `code` gives the value that column indicates.

    Raises:
      ValueError: `encoding` or `slots` is not a recognised value.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"site encoding must be one of {list(ENCODINGS)}; got {encoding!r}.")
    if slots not in SLOT_CHOICES:
        raise ValueError(f"site slots must be one of {list(SLOT_CHOICES)}; got {slots!r}.")
    rows = []
    for slot, cache in _selected_slots(cache_a, cache_b, slots):
        n_sites = cache.codes.shape[1]
        codes = declared_codes(cache) if encoding == 'onehot' else [None]
        for site in range(1, n_sites + 1):
            for code in codes:
                rows.append({'slot': slot, 'protein': cache.protein, 'site': site, 'code': code})
    out = pd.DataFrame(rows)
    out.insert(0, 'column', np.arange(len(out)))
    return out


def _selected_slots(cache_a: SiteCache, cache_b: SiteCache, slots: str) -> list:
    """The (slot letter, cache) pairs that contribute columns, in output order.

    One place decides this, so the column layout and the feature matrix cannot disagree about
    which slots are present or what order they come in.

    Args:
      cache_a: slot A's cache.
      cache_b: slot B's cache.
      slots: `both`, `a` or `b`.

    Returns:
      A list of `(slot letter, cache)` tuples.
    """
    available = {'a': cache_a, 'b': cache_b}
    letters = ('a', 'b') if slots == 'both' else (slots,)
    return [(letter, available[letter]) for letter in letters]


def _slot_matrix(cache: SiteCache, hashes: pd.Series, slot: str, encoding: str) -> np.ndarray:
    """Look one slot's sequences up in the cache and encode them.

    Args:
      cache: that slot's cache.
      hashes: `cds_dna_hash` per pair, in pair order.
      slot: `a` or `b`, used in the error text only.
      encoding: `ordinal` or `onehot`.

    Returns:
      float32 array, (n pairs, n sites) under `ordinal` or (n pairs, n sites x n codes) under
      `onehot`.

    Raises:
      KeyError: a hash is not in the cache.
    """
    rows = hashes.map(cache.hash_to_row)
    missing = rows.isna()
    if missing.any():
        n_missing = int(missing.sum())
        example = hashes[missing].iloc[0]
        raise KeyError(
            f"slot {slot} ({cache.protein}): {n_missing:,} of {len(rows):,} pairs have a "
            f"cds_dna_hash missing from the {cache.unit} cache index, e.g. {example}.")
    codes = cache.codes[rows.to_numpy(dtype=int)]
    if encoding == 'ordinal':
        return codes.astype(np.float32)

    # One column per (site, declared code). Built against the declared code set, not the observed
    # one, so every split has the same width.
    code_to_slot = {code: i for i, code in enumerate(declared_codes(cache))}
    n_pairs, n_sites = codes.shape
    n_codes = len(code_to_slot)
    lookup = np.full(256, -1, dtype=np.int64)
    for code, position in code_to_slot.items():
        lookup[code] = position
    onehot = np.zeros((n_pairs, n_sites * n_codes), dtype=np.float32)
    column = np.arange(n_sites) * n_codes + lookup[codes]
    onehot[np.arange(n_pairs)[:, None], column] = 1.0
    return onehot


def get_site_pair_features(pairs_df: pd.DataFrame, cache_a: SiteCache, cache_b: SiteCache,
                           encoding: str = 'ordinal',
                           slots: str = 'both') -> tuple[np.ndarray, np.ndarray]:
    """Build the pair feature matrix from two per-protein site caches.

    Slot A's sites come first, then slot B's -- `concat`, the only interaction the two spaces
    admit. Every pair must resolve; unlike the k-mer path this does not skip a pair it cannot look
    up, because dropping rows would quietly change the evaluation set.

    `slots='a'` or `'b'` keeps one side's columns and drops the other. That is an ablation, not a
    feature set anyone should train for real: the label says whether two sequences came from the
    same isolate, and one sequence alone cannot answer it, since the same sequence appears in both
    matched and mismatched rows. A one-side run scoring near chance is the expected result and
    confirms the model has to relate the two sides; scoring well above chance would mean something
    about one sequence on its own predicts the label, which could only come from how the pairs
    were built.

    Args:
      pairs_df: pair table with `cds_dna_hash_a`, `cds_dna_hash_b` and `label`.
      cache_a: slot A's cache.
      cache_b: slot B's cache.
      encoding: `ordinal` or `onehot`.
      slots: `both` (default), or `a` / `b` for the one-side ablation.

    Returns:
      `(features, labels)`. `features` is float32 (n pairs, n columns); `labels` is (n pairs,).

    Raises:
      ValueError: a required column is missing, the two caches disagree on the unit, or
          `encoding` / `slots` is not a recognised value.
      KeyError: a pair's hash is not in the cache.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"site encoding must be one of {list(ENCODINGS)}; got {encoding!r}.")
    if slots not in SLOT_CHOICES:
        raise ValueError(f"site slots must be one of {list(SLOT_CHOICES)}; got {slots!r}.")
    if cache_a.unit != cache_b.unit:
        raise ValueError(
            f"the two slots were built with different site units: {cache_a.protein} is "
            f"{cache_a.unit!r} and {cache_b.protein} is {cache_b.unit!r}.")
    required = ['cds_dna_hash_a', 'cds_dna_hash_b', 'label']
    missing = [c for c in required if c not in pairs_df.columns]
    if missing:
        raise ValueError(
            f"get_site_pair_features: pair table missing {missing}. cds_dna_hash_a/b are written "
            f"when the dataset is built with split_strategy.pair_key_alphabet=nt_cds.")

    blocks = [_slot_matrix(cache, pairs_df[f'cds_dna_hash_{letter}'], letter, encoding)
              for letter, cache in _selected_slots(cache_a, cache_b, slots)]
    features = np.hstack(blocks) if len(blocks) > 1 else blocks[0]
    labels = pairs_df['label'].to_numpy()
    return features, labels
