"""Exercise the aligned codon cache reader with a two-row cache beside a one-column alignment."""
import json

import numpy as np
import pandas as pd
import pytest

from src.utils.experiment_utils import file_sha256
from src.utils.site_utils import read_aligned_codon_cache


@pytest.fixture
def cache(tmp_path):
    fasta = tmp_path / 'aligned_proteins.fasta'
    fasta.write_text('>first\nK\n>second\nK\n')
    selected = tmp_path / 'selected_cds.parquet'
    pd.DataFrame({'cds_dna_hash': ['first', 'second']}).to_parquet(selected, index=False)
    (tmp_path / 'alignment_validation.json').write_text(json.dumps(
        {'aligned_width': 1, 'aligned_fasta_sha256': file_sha256(fasta)}))

    stem = tmp_path / 'site_features_codon_example_aligned'
    codes = np.array([[40, 98], [97, 40]], dtype=np.uint8)
    metadata = {'site_mode': 'aligned', 'unit': 'codon', 'protein': 'example',
                'index_key': 'cds_dna_hash', 'n_sequences': 2, 'n_sites': 2,
                'code_map': {'AAA': 40, 'GAP': 97, 'AFTER_STOP': 98},
                'aligned_fasta_sha256': file_sha256(fasta),
                'selected_cds_sha256': file_sha256(selected)}
    np.savez(f'{stem}.npz', codes=codes)
    pd.DataFrame({'cds_dna_hash': ['second', 'first'], 'row': [1, 0]}).to_parquet(
        f'{stem}_index.parquet', index=False)
    path = tmp_path / f'{stem.name}_metadata.json'
    path.write_text(json.dumps(metadata))
    return stem, path, metadata


def test_reader_returns_site_cache_with_saved_row_numbers(cache):
    stem, _, _ = cache
    site_cache = read_aligned_codon_cache(stem.parent, 'example')
    assert (site_cache.protein, site_cache.unit) == ('example', 'codon')
    assert site_cache.codes[site_cache.hash_to_row['second']].tolist() == [97, 40]


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


def test_rejects_width_other_than_alignment_width_plus_one(cache):
    stem, _, _ = cache
    report = stem.parent / 'alignment_validation.json'
    validation = json.loads(report.read_text())
    validation['aligned_width'] = 2
    report.write_text(json.dumps(validation))
    with pytest.raises(ValueError, match='width must be 3'):
        read_aligned_codon_cache(stem.parent, 'example')


@pytest.mark.parametrize('name', ['aligned_proteins.fasta', 'selected_cds.parquet'])
def test_rejects_changed_source_file(cache, name):
    stem, _, _ = cache
    source = stem.parent / name
    source.write_bytes(source.read_bytes() + b'\n')
    with pytest.raises(ValueError, match=f'{name} has sha256'):
        read_aligned_codon_cache(stem.parent, 'example')


def test_accepts_source_files_rewritten_with_identical_bytes(cache):
    stem, _, _ = cache
    for name in ('aligned_proteins.fasta', 'selected_cds.parquet'):
        source = stem.parent / name
        source.write_bytes(source.read_bytes())
    assert read_aligned_codon_cache(stem.parent, 'example').codes.shape == (2, 2)


def test_rejects_validation_report_of_another_fasta(cache):
    stem, _, _ = cache
    report = stem.parent / 'alignment_validation.json'
    validation = json.loads(report.read_text())
    validation['aligned_fasta_sha256'] = '0' * 64
    report.write_text(json.dumps(validation))
    with pytest.raises(ValueError, match='describes a different'):
        read_aligned_codon_cache(stem.parent, 'example')
