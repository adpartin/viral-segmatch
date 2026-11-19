"""Fixtures shared by more than one test module."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.experiment_utils import file_sha256


def _write_aligned_codon_cache(directory: Path, protein: str) -> tuple:
    """Write a two-row aligned codon cache beside a one-column alignment.

    The directory gets every file `read_aligned_codon_cache` reads: the cache's three files,
    `aligned_proteins.fasta`, `selected_cds.parquet` and `alignment_validation.json`, all
    consistent with each other.

    Args:
      directory: the final alignment directory to write into; created if absent.
      protein: short protein name recorded in the cache's file name and metadata.

    Returns:
      `(stem, metadata_path, metadata)`.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fasta = directory / 'aligned_proteins.fasta'
    fasta.write_text('>first\nK\n>second\nK\n')
    selected = directory / 'selected_cds.parquet'
    pd.DataFrame({'cds_dna_hash': ['first', 'second']}).to_parquet(selected, index=False)
    (directory / 'alignment_validation.json').write_text(json.dumps(
        {'aligned_width': 1, 'aligned_fasta_sha256': file_sha256(fasta)}))

    stem = directory / f'site_features_codon_{protein}_aligned'
    codes = np.array([[40, 98], [97, 40]], dtype=np.uint8)
    metadata = {'site_mode': 'aligned', 'unit': 'codon', 'protein': protein,
                'index_key': 'cds_dna_hash', 'n_sequences': 2, 'n_sites': 2,
                'code_map': {'AAA': 40, 'GAP': 97, 'AFTER_STOP': 98},
                'aligned_fasta_sha256': file_sha256(fasta),
                'selected_cds_sha256': file_sha256(selected)}
    np.savez(f'{stem}.npz', codes=codes)
    pd.DataFrame({'cds_dna_hash': ['second', 'first'], 'row': [1, 0]}).to_parquet(
        f'{stem}_index.parquet', index=False)
    metadata_path = directory / f'{stem.name}_metadata.json'
    metadata_path.write_text(json.dumps(metadata))
    return stem, metadata_path, metadata


@pytest.fixture
def write_aligned_codon_cache():
    """`write_aligned_codon_cache(directory, protein)` writes a valid two-row aligned codon cache."""
    return _write_aligned_codon_cache
