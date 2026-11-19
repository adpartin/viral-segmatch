"""Tests for the aligned-mode dataset checks.

Covers `check_population_inside_alignments` in src/datasets/_pair_helpers.py and the aligned-mode
rule in `_validate_v2_config` in src/datasets/dataset_segment_pairs_v2.py.

Run: python -m pytest tests/test_aligned_dataset_checks.py
"""
import json

import pytest
from omegaconf import OmegaConf

from src.datasets._pair_helpers import check_population_inside_alignments
from src.datasets.dataset_segment_pairs_v2 import _validate_v2_config

TASK2_FILTERS = {'host': ['Human', 'Swine', 'Chicken', 'Duck'],
                 'hn_subtype': ['H3N2', 'H1N1', 'H5N1'], 'year': None, 'year_range': [2010, 2025]}


@pytest.fixture
def alignment_dirs(tmp_path):
    """Two final alignments of the Task 2 population, as `align_proteins.py` records it."""
    dirs = {}
    for protein in ('PB1', 'HA'):
        directory = tmp_path / protein
        directory.mkdir()
        manifest = {'filters': {**TASK2_FILTERS, 'is_complete_cds': True,
                                'length_share_above': 0.02}}
        (directory / 'selection_manifest.json').write_text(json.dumps(manifest))
        dirs[protein] = directory
    return dirs


@pytest.mark.parametrize('filters', [
    {'host': ['Human'], 'hn_subtype': ['H3N2'], 'year': [2024]},
    {'host': 'Human', 'hn_subtype': 'H3N2', 'year': 2024},
    {'host': ['Human', 'Duck'], 'hn_subtype': ['H5N1'], 'year_range': [2015, 2025]},
    TASK2_FILTERS,
    OmegaConf.create({'host': ['Human'], 'hn_subtype': ['H3N2'], 'year': [2024]}),
])
def test_population_inside_passes(alignment_dirs, filters):
    check_population_inside_alignments(filters, alignment_dirs)


@pytest.mark.parametrize(('change', 'message'), [
    ({'host': ['Human', 'Other Bird']}, r"PB1: dataset host values \['Other Bird'\]"),
    ({'hn_subtype': ['H9N2']}, r"HA: dataset hn_subtype values \['H9N2'\]"),
    ({'year': [2009, 2024]}, r"dataset year values \[2009\]"),
    ({'year': None, 'year_range': [2020, 2026]}, r"dataset year values \[2026\]"),
    ({'hn_subtype': None}, r"leaves \['hn_subtype'\] unset"),
    ({'year': None}, r"leaves \['year'\] unset"),
    ({'geo_location': ['USA']}, 'dataset.geo_location is set'),
    ({'passage': 'MDCK'}, 'dataset.passage is set'),
])
def test_population_outside_raises(alignment_dirs, change, message):
    filters = {'host': ['Human'], 'hn_subtype': ['H3N2'], 'year': [2024], **change}
    with pytest.raises(ValueError, match=message):
        check_population_inside_alignments(filters, alignment_dirs)


def test_unset_alignment_axis_allows_every_value(alignment_dirs):
    manifest_path = alignment_dirs['HA'] / 'selection_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['filters']['host'] = None
    manifest_path.write_text(json.dumps(manifest))
    filters = {'host': ['Other Bird'], 'hn_subtype': ['H3N2'], 'year': [2024]}
    with pytest.raises(ValueError) as error:
        check_population_inside_alignments(filters, alignment_dirs)
    assert 'PB1: dataset host' in str(error.value)
    assert 'HA: dataset host' not in str(error.value)


def v2_config(mode, require_pinned):
    config = {'dataset': {'schema_pair': ['A', 'B'], 'pair_mode': 'schema_ordered',
                          'hard_partition_isolates': True,
                          'split_strategy': {'mode': 'random'},
                          'require_complete_cds_at_pinned_length': require_pinned}}
    if mode is not None:
        config['site'] = {'mode': mode}
    return OmegaConf.create(config)


@pytest.mark.parametrize(('mode', 'require_pinned'), [
    ('aligned', False), ('aligned', None), ('pinned', True), (None, True),
])
def test_v2_config_accepts(mode, require_pinned):
    _validate_v2_config(v2_config(mode, require_pinned))


def test_v2_config_rejects_pinned_length_filter_in_aligned_mode():
    with pytest.raises(ValueError, match='require_complete_cds_at_pinned_length must be false'):
        _validate_v2_config(v2_config('aligned', True))
