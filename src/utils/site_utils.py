"""Per-site feature primitives: build the site matrix, read the cache, make pair features.

Sibling of `kmer_utils.py`, for the cache `src/embeddings/compute_site_features.py` writes: one
uint8 matrix per (protein, site unit), one row per unique CDS, keyed by `cds_dna_hash`.

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
from typing import NamedTuple

import numpy as np
import pandas as pd

ENCODINGS = ('ordinal', 'onehot')
# Which slots' columns end up in the pair matrix. 'both' is the real feature set; 'a' and 'b'
# are the one-side ablation, which should score near chance -- the label is a fact about a
# PAIR, and the same sequence appears on both sides of it.
SLOT_CHOICES = ('both', 'a', 'b')


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
    return np.frombuffer(flat, dtype=np.uint8).reshape(len(sequences), lengths.pop())


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


def load_site_cache(site_dir: Path, unit: str, protein: str) -> SiteCache:
    """Load one protein's per-site feature cache.

    Args:
      site_dir: directory holding the cache, normally the embeddings output dir.
      unit: `nt`, `codon` or `aa`.
      protein: short protein name, e.g. `HA`.

    Returns:
      The `SiteCache` for that protein and unit.

    Raises:
      FileNotFoundError: any of the three cache files is missing.
      ValueError: the matrix and the index disagree on how many rows there are.
    """
    site_dir = Path(site_dir)
    stem = f'site_features_{unit}_{protein}'
    paths = {suffix: site_dir / f'{stem}{suffix}'
             for suffix in ('.npz', '_index.parquet', '_metadata.json')}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"per-site cache for {protein} ({unit}) is incomplete; missing {missing}. Build it "
            f"with `python src/embeddings/compute_site_features.py --config_bundle <bundle> "
            f"--unit {unit} --proteins {protein}`.")

    codes = np.load(paths['.npz'])['codes']
    index = pd.read_parquet(paths['_index.parquet'])
    metadata = json.loads(paths['_metadata.json'].read_text())
    if len(index) != codes.shape[0]:
        raise ValueError(
            f"{stem}: index has {len(index):,} rows but the matrix has {codes.shape[0]:,}. The "
            f"cache is inconsistent; rebuild it with --force_recompute.")
    hash_to_row = dict(zip(index['cds_dna_hash'], index['row'].astype(int)))
    return SiteCache(protein=protein, unit=unit, codes=codes, hash_to_row=hash_to_row,
                     metadata=metadata)


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
      One row per column with `column`, `slot`, `protein`, `site` (1-based along the CDS) and,
      under `onehot`, the `code` that column indicates.

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
    if rows.isna().any():
        n_missing = int(rows.isna().sum())
        example = hashes[rows.isna()].iloc[0]
        raise KeyError(
            f"slot {slot} ({cache.protein}): {n_missing:,} of {len(rows):,} pairs have a "
            f"cds_dna_hash the per-site cache does not hold, e.g. {example}. The cache covers "
            f"complete CDS at the pinned length, so a dataset built without "
            f"dataset.require_complete_cds_at_pinned_length will not line up with it.")
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
