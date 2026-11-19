"""Compare saved aligned and pinned codon caches by CDS hash.

Run from the repository root with `python -m src.analysis.check_aligned_codon_caches`.
The comparisons are specified in step 7 of docs/plans/2026-11-17_aligned_codon_features_plan.md.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.utils.site_utils import aligned_site_cache_stem, site_cache_stem

ROOT = Path(__file__).resolve().parents[2]
SITE_CONFIG = ROOT / 'conf/site/aligned_4host_3subtype_2010_2025.yaml'
PINNED_DIR = ROOT / 'data/embeddings/flu/July_2025'
COMPARISONS = {'PB2': (2280,), 'PA': (2151,), 'NP': (1497,), 'M1': (759,),
               'PB1': (2277, 2274), 'NS1': (693,)}


def read_index(stem: Path, n_rows: int) -> pd.Series:
    """Read the saved hash-to-row mapping and verify that it covers each matrix row once."""
    index = pd.read_parquet(f'{stem}_index.parquet', columns=['cds_dna_hash', 'row'])
    rows = index.set_index('cds_dna_hash', verify_integrity=True)['row']
    if (rows.index.hasnans or not pd.api.types.is_integer_dtype(rows)
            or not np.array_equal(np.sort(rows.to_numpy()), np.arange(n_rows))):
        raise ValueError(f'{stem}: invalid hash-to-row index')
    return rows


def read_aligned_codon_cache(directory: Path, protein: str) -> tuple[np.ndarray, pd.Series]:
    """Read one aligned codon cache and verify its metadata, code matrix and index.

    Args:
      directory: directory containing the saved cache.
      protein: short name used in the cache filename and metadata.

    Returns:
      The uint8 code matrix and its CDS hash to row-number mapping.
    """
    stem = directory / aligned_site_cache_stem('codon', protein)
    metadata = json.loads(Path(f'{stem}_metadata.json').read_text())
    identity = {'site_mode': 'aligned', 'unit': 'codon', 'protein': protein,
                'index_key': 'cds_dna_hash'}
    for field, expected in identity.items():
        if metadata.get(field) != expected:
            raise ValueError(f'{stem}: expected {field}={expected!r}, got {metadata.get(field)!r}')

    code_map = metadata.get('code_map')
    if not isinstance(code_map, dict) or code_map.get('GAP') != 97 or code_map.get('AFTER_STOP') != 98:
        raise ValueError(f'{stem}: expected GAP=97 and AFTER_STOP=98 in code_map')
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
    rows = read_index(stem, len(codes))
    return codes, rows


def main() -> None:
    directories = OmegaConf.load(SITE_CONFIG).alignment_dirs
    total = 0
    for protein, lengths in COMPARISONS.items():
        directory = ROOT / directories[protein]
        selected = pd.read_parquet(directory / 'selected_cds.parquet',
                                   columns=['cds_dna_hash', 'cds_length']).drop_duplicates()
        selected = selected.set_index('cds_dna_hash', verify_integrity=True)
        aligned, aligned_rows = read_aligned_codon_cache(directory, protein)
        if not selected.index.symmetric_difference(aligned_rows.index).empty:
            raise ValueError(f'{directory}: aligned index and final selection cover different CDS')

        for length in lengths:
            hashes = selected.index[selected['cds_length'] == length]
            pinned_stem = PINNED_DIR / site_cache_stem('codon', protein, length)
            with np.load(f'{pinned_stem}.npz') as archive:
                pinned = archive['codes']
            pinned_rows = read_index(pinned_stem, len(pinned))
            missing = hashes.difference(pinned_rows.index)
            if hashes.empty or len(missing):
                raise ValueError(f'{protein} {length} nt: {len(hashes)} selected, '
                                 f'{len(missing)} missing from pinned cache')
            width = length // 3
            extra = (protein, length) == ('PB1', 2274)
            if pinned.shape[1:] != (width,) or aligned.shape[1:] != (width + int(extra),):
                raise ValueError(f'{protein} {length} nt: unexpected widths, '
                                 f'pinned {pinned.shape}, aligned {aligned.shape}')
            observed = aligned[aligned_rows.loc[hashes].to_numpy()]
            expected = pinned[pinned_rows.loc[hashes].to_numpy()]
            mismatch = (observed[:, :width] != expected).any(axis=1)
            if extra:
                mismatch |= observed[:, width] != 98
            if mismatch.any():
                raise ValueError(f'{protein} {length} nt: {mismatch.sum()} rows differ, '
                                 f'first hash {hashes[mismatch][0]}')
            total += len(hashes)
            print(f'{protein} {length} nt: {len(hashes):,} matched', flush=True)
    print(f'Total: {total:,} matched')


if __name__ == '__main__':
    main()
