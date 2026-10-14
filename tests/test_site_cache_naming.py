"""Unit tests for the per-site cache naming in src/utils/site_utils.py.

The pinned length is part of a cache's file name because it decides the contents: one protein
takes different pins in different populations, and human H3N2 PB1 at 2,277 nt and the 2,274 nt
form of every other population have to be able to sit side by side.

Run: python -m pytest tests/test_site_cache_naming.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.utils.site_utils import (
    load_site_cache,
    resolve_site_cache_pin,
    site_cache_paths,
    site_cache_stem,
)


def write_cache(site_dir: Path, unit: str, protein: str, pinned_nt: int,
                metadata_pin: int = None):
    """A three-file cache like the builder writes, small enough to be instant."""
    site_dir.mkdir(parents=True, exist_ok=True)
    paths = site_cache_paths(site_dir, unit, protein, pinned_nt)
    n_sites = pinned_nt // 3
    np.savez_compressed(paths['.npz'], codes=np.zeros((2, n_sites), dtype=np.uint8))
    pd.DataFrame({'cds_dna_hash': ['a', 'b'], 'row': [0, 1]}).to_parquet(
        paths['_index.parquet'], index=False)
    paths['_metadata.json'].write_text(json.dumps({
        'unit': unit, 'protein': protein,
        'pinned_nt': pinned_nt if metadata_pin is None else metadata_pin,
        'n_sites': n_sites, 'code_map': {'AAA': 1}}))
    return paths


def test_stem_carries_unit_protein_and_pin():
    assert site_cache_stem('codon', 'PB1', 2274) == 'site_features_codon_PB1_2274nt'
    assert site_cache_stem('nt', 'HA', 1701) == 'site_features_nt_HA_1701nt'


def test_two_pins_of_one_protein_coexist(tmp_path):
    write_cache(tmp_path, 'codon', 'PB1', 2274)
    write_cache(tmp_path, 'codon', 'PB1', 2277)

    short = load_site_cache(tmp_path, 'codon', 'PB1', 2274)
    long = load_site_cache(tmp_path, 'codon', 'PB1', 2277)
    assert short.codes.shape[1] == 758
    assert long.codes.shape[1] == 759
    assert short.metadata['pinned_nt'] == 2274


def test_resolution_needs_a_pin_when_there_are_two(tmp_path):
    write_cache(tmp_path, 'codon', 'PB1', 2274)
    write_cache(tmp_path, 'codon', 'PB1', 2277)
    with pytest.raises(ValueError, match=r'\[2274, 2277\]'):
        resolve_site_cache_pin(tmp_path, 'codon', 'PB1')
    with pytest.raises(ValueError, match=r'\[2274, 2277\]'):
        load_site_cache(tmp_path, 'codon', 'PB1')


def test_resolution_finds_the_only_cache(tmp_path):
    write_cache(tmp_path, 'codon', 'HA', 1701)
    assert resolve_site_cache_pin(tmp_path, 'codon', 'HA') == 1701
    assert load_site_cache(tmp_path, 'codon', 'HA').codes.shape[1] == 567


def test_missing_cache_names_what_it_looked_for(tmp_path):
    with pytest.raises(FileNotFoundError, match='site_features_codon_NP_'):
        resolve_site_cache_pin(tmp_path, 'codon', 'NP')
    write_cache(tmp_path, 'codon', 'NP', 1497)
    # Asking for a pin that was never built says so, and says which pin is there.
    with pytest.raises(FileNotFoundError, match=r'1500 nt is incomplete.*\[1497\] nt'):
        load_site_cache(tmp_path, 'codon', 'NP', 1500)


def test_name_and_metadata_must_agree(tmp_path):
    write_cache(tmp_path, 'codon', 'PB1', 2274, metadata_pin=2277)
    with pytest.raises(ValueError, match='file name says 2274'):
        load_site_cache(tmp_path, 'codon', 'PB1', 2274)


def test_one_unit_does_not_see_another(tmp_path):
    write_cache(tmp_path, 'codon', 'HA', 1701)
    with pytest.raises(FileNotFoundError):
        resolve_site_cache_pin(tmp_path, 'nt', 'HA')
