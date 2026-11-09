"""Tests for host name normalization in `src/preprocess/flu_genomes_eda.py`.

Covers:
  1. Missing, empty and 'null' values carry no host
  2. 'human' and 'environment' reach their canonical capitalization from any case
  3. An all-lowercase label is title-cased; an already-capitalized one is left alone
  4. Every HOST_NORMALIZATION entry is applied
  5. HOST_NORMALIZATION targets are themselves canonical (no two-step mapping)
  6. report_host_name_collisions finds a case/punctuation pair and stays quiet once it is mapped
  7. Real-corpus check: the stored `host_common_name` column is already clean

Run: python tests/test_host_name_normalization.py
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.preprocess.flu_genomes_eda import (  # noqa: E402
    HOST_NORMALIZATION,
    _host_name_key,
    _normalize_host_value,
    report_host_name_collisions,
)

CURATED_HOST_COLUMN = (PROJ / 'data/processed/flu/metadata_eda'
                       / 'flu_genomes_metadata_parsed.csv')


@pytest.mark.parametrize('value', [None, float('nan'), '', '   ', 'null', 'Null', 'NULL'])
def test_values_carrying_no_host_become_none(value):
    assert _normalize_host_value(value) is None


@pytest.mark.parametrize('value', ['human', 'Human', 'HUMAN', ' human '])
def test_human_is_canonicalized(value):
    assert _normalize_host_value(value) == 'Human'


@pytest.mark.parametrize('value', ['environment', 'Environment', 'ENVIRONMENT'])
def test_environment_is_canonicalized(value):
    assert _normalize_host_value(value) == 'Environment'


def test_lowercase_label_is_title_cased():
    assert _normalize_host_value('red fox') == 'Red Fox'


def test_already_capitalized_label_is_left_alone():
    assert _normalize_host_value('Great-Horned Owl') == 'Great-Horned Owl'
    assert _normalize_host_value("Bewick's Swan") == "Bewick's Swan"


@pytest.mark.parametrize('variant,canonical', sorted(HOST_NORMALIZATION.items()))
def test_every_normalization_entry_is_applied(variant, canonical):
    assert _normalize_host_value(variant) == canonical


@pytest.mark.parametrize('canonical', sorted(set(HOST_NORMALIZATION.values())))
def test_normalization_targets_are_themselves_canonical(canonical):
    """A target must not be a key, or normalizing would need two passes."""
    assert _normalize_host_value(canonical) == canonical


def test_host_name_key_ignores_case_and_punctuation():
    assert _host_name_key('Blue-Winged Teal') == _host_name_key('blue winged  teal')
    assert _host_name_key('Northern_Pintail') == _host_name_key('Northern Pintail')
    assert _host_name_key('Duck') != _host_name_key('Wood Duck')


def test_collision_report_flags_an_unmapped_pair():
    unresolved = report_host_name_collisions(pd.Series(['Snowy Owl', 'Snowy owl']))
    assert unresolved == {'snowy owl': ['Snowy Owl', 'Snowy owl']}


def test_collision_report_is_quiet_for_a_mapped_pair():
    mapped = next(iter(HOST_NORMALIZATION))
    both = pd.Series([mapped, HOST_NORMALIZATION[mapped]])
    assert report_host_name_collisions(both) == {}


def test_collision_report_ignores_missing_values():
    assert report_host_name_collisions(pd.Series([None, '', 'null', 'Human'])) == {}


@pytest.mark.skipif(not CURATED_HOST_COLUMN.exists(),
                    reason='parsed metadata not available')
def test_stored_animal_names_are_already_clean():
    """`flu_genomes_eda.py` writes the cleaned name, so normalizing again changes nothing."""
    hosts = pd.read_csv(CURATED_HOST_COLUMN, dtype=str, keep_default_na=False,
                        na_values=[''], usecols=['host_common_name'])['host_common_name']
    assert report_host_name_collisions(hosts) == {}
    normalized = hosts.map(_normalize_host_value)
    assert normalized.dropna().nunique() == hosts.dropna().nunique()
    assert set(normalized.dropna()) == set(hosts.dropna())
    assert 'Cattle' not in set(hosts.dropna())


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
