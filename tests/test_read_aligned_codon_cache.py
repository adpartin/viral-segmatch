"""Exercise the aligned codon cache reader with a two-row cache beside a one-column alignment."""
import json

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.utils.site_utils import (
    configured_site_cache_paths,
    load_configured_site_cache,
    read_aligned_codon_cache,
)


@pytest.fixture
def cache(tmp_path, write_aligned_codon_cache):
    return write_aligned_codon_cache(tmp_path, 'example')


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


def aligned_config(alignment_dirs: dict) -> OmegaConf:
    return OmegaConf.create({'site': {'mode': 'aligned', 'unit': 'codon',
                                      'alignment_dirs': alignment_dirs}})


def test_configured_aligned_cache_is_read_from_the_alignment_directory(cache):
    stem, path, _ = cache
    config = aligned_config({'example': str(stem.parent)})
    paths = configured_site_cache_paths(config, 'example')
    assert paths == {suffix: stem.parent / f'{stem.name}{suffix}'
                     for suffix in ('.npz', '_index.parquet', '_metadata.json')}
    assert paths['_metadata.json'] == path
    assert load_configured_site_cache(config, 'example').codes.shape == (2, 2)


def test_configured_aligned_cache_needs_an_alignment_directory(cache):
    stem, _, _ = cache
    config = aligned_config({'other': str(stem.parent)})
    with pytest.raises(ValueError, match='no entry for example'):
        configured_site_cache_paths(config, 'example')
    with pytest.raises(ValueError, match='no entry for example'):
        load_configured_site_cache(config, 'example')
