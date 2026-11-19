"""Compare saved aligned and pinned codon caches by CDS hash.

Run from the repository root with `python -m src.analysis.check_aligned_codon_caches`.
The comparisons are specified in step 7 of docs/plans/2026-11-17_aligned_codon_features_plan.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.utils.site_utils import (
    AFTER_STOP_CODE,
    read_aligned_codon_cache,
    read_site_cache_index,
    site_cache_stem,
)

ROOT = Path(__file__).resolve().parents[2]
SITE_CONFIG = ROOT / 'conf/site/aligned_4host_3subtype_2010_2025.yaml'
PINNED_DIR = ROOT / 'data/embeddings/flu/July_2025'
COMPARISONS = {'PB2': (2280,), 'PA': (2151,), 'NP': (1497,), 'M1': (759,),
               'PB1': (2277, 2274), 'NS1': (693,)}


def main() -> None:
    directories = OmegaConf.load(SITE_CONFIG).alignment_dirs
    total = 0
    for protein, lengths in COMPARISONS.items():
        directory = ROOT / directories[protein]
        selected = pd.read_parquet(directory / 'selected_cds.parquet',
                                   columns=['cds_dna_hash', 'cds_length']).drop_duplicates()
        selected = selected.set_index('cds_dna_hash', verify_integrity=True)
        cache = read_aligned_codon_cache(directory, protein)
        aligned = cache.codes
        aligned_rows = pd.Series(cache.hash_to_row)
        if not selected.index.symmetric_difference(aligned_rows.index).empty:
            raise ValueError(f'{directory}: aligned index and final selection cover different CDS')

        for length in lengths:
            hashes = selected.index[selected['cds_length'] == length]
            pinned_stem = PINNED_DIR / site_cache_stem('codon', protein, length)
            with np.load(f'{pinned_stem}.npz') as archive:
                pinned = archive['codes']
            pinned_rows = read_site_cache_index(pinned_stem, len(pinned))
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
                mismatch |= observed[:, width] != AFTER_STOP_CODE
            if mismatch.any():
                raise ValueError(f'{protein} {length} nt: {mismatch.sum()} rows differ, '
                                 f'first hash {hashes[mismatch][0]}')
            total += len(hashes)
            print(f'{protein} {length} nt: {len(hashes):,} matched', flush=True)
    print(f'Total: {total:,} matched')


if __name__ == '__main__':
    main()
