"""Exercise the standalone reader with a two-row cache."""
import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.check_aligned_codon_caches import read_aligned_codon_cache


@pytest.fixture
def cache(tmp_path):
    stem = tmp_path / 'site_features_codon_example_aligned'
    codes = np.array([[40, 98], [97, 40]], dtype=np.uint8)
    metadata = {'site_mode': 'aligned', 'unit': 'codon', 'protein': 'example',
                'index_key': 'cds_dna_hash', 'n_sequences': 2, 'n_sites': 2,
                'code_map': {'AAA': 40, 'GAP': 97, 'AFTER_STOP': 98}}
    np.savez(f'{stem}.npz', codes=codes)
    pd.DataFrame({'cds_dna_hash': ['second', 'first'], 'row': [1, 0]}).to_parquet(
        f'{stem}_index.parquet', index=False)
    path = tmp_path / f'{stem.name}_metadata.json'
    path.write_text(json.dumps(metadata))
    return stem, path, metadata


def test_reader_uses_saved_row_numbers(cache):
    stem, _, _ = cache
    codes, rows = read_aligned_codon_cache(stem.parent, 'example')
    assert codes[rows['second']].tolist() == [97, 40]


@pytest.mark.parametrize(('field', 'value', 'error'), [
    ('site_mode', 'pinned', 'site_mode'), ('protein', 'wrong', 'protein'),
    ('unit', 'nt', 'unit'), ('n_sites', 3, 'shape'), ('index_key', 'row', 'index_key'),
    ('code_map', {'GAP': 97, 'AFTER_STOP': 98}, 'missing from code_map'),
    ('code_map', {'GAP': 97, 'AFTER_STOP': 97}, 'AFTER_STOP=98'),
])
def test_rejects_inconsistent_metadata(cache, field, value, error):
    stem, path, metadata = cache
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=error):
        read_aligned_codon_cache(stem.parent, 'example')


def test_rejects_duplicate_row_numbers(cache):
    stem, _, _ = cache
    pd.DataFrame({'cds_dna_hash': ['first', 'second'], 'row': [0, 0]}).to_parquet(
        f'{stem}_index.parquet', index=False)
    with pytest.raises(ValueError, match='hash-to-row index'):
        read_aligned_codon_cache(stem.parent, 'example')


def test_rejects_matrix_with_wrong_dtype(cache):
    stem, _, _ = cache
    np.savez(f'{stem}.npz', codes=np.full((2, 2), 40, dtype=float))
    with pytest.raises(ValueError, match='uint8'):
        read_aligned_codon_cache(stem.parent, 'example')
