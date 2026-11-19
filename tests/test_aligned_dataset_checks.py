"""Tests for the aligned-mode dataset checks.

Covers `check_population_inside_alignments` in src/datasets/_pair_helpers.py, the aligned-mode
rule in `_validate_v2_config` in src/datasets/dataset_segment_pairs_v2.py, the 2D-CD builder's
rejection of aligned mode in src/datasets/dataset_pairs_cc.py, and `check_aligned_dataset` in
src/models/train_pair_baselines.py.

Run: python -m pytest tests/test_aligned_dataset_checks.py
"""
import json

import pytest
from omegaconf import OmegaConf

from src.datasets._pair_helpers import check_population_inside_alignments
from src.datasets.dataset_pairs_cc import _resolve_spec
from src.datasets.dataset_segment_pairs_v2 import _validate_v2_config
from src.models.train_pair_baselines import check_aligned_dataset

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


def test_cc_builder_rejects_aligned_mode():
    config = OmegaConf.create({'site': {'mode': 'aligned'},
                               'dataset': {'split_strategy': {'mode': 'cluster_disjoint_cc'}}})
    with pytest.raises(ValueError, match="does not support site.mode='aligned'"):
        _resolve_spec(None, config)


ALIGNMENT_DIRS = {'PB1': 'alignments/PB1', 'HA': 'alignments/HA', 'NS1': 'alignments/NS1'}


def site_config(site: dict):
    return OmegaConf.create({'site': site})


def save_dataset_config(tmp_path, site: dict):
    path = tmp_path / 'resolved_config.yaml'
    OmegaConf.save(site_config(site), path)
    return path


def test_aligned_dataset_check_skips_pinned_fits():
    check_aligned_dataset(site_config({'mode': 'pinned'}), None, ('PB1', 'HA'))
    check_aligned_dataset(site_config({}), None, ('PB1', 'HA'))


def test_aligned_dataset_check_passes_on_same_alignments(tmp_path):
    training = site_config({'mode': 'aligned', 'alignment_dirs': ALIGNMENT_DIRS})
    # A different NS1 alignment does not matter to a PB1-HA fit.
    dataset = save_dataset_config(tmp_path, {'mode': 'aligned', 'alignment_dirs': {
        **ALIGNMENT_DIRS, 'NS1': 'other/NS1'}})
    check_aligned_dataset(training, dataset, ('PB1', 'HA'))


@pytest.mark.parametrize(('dataset_site', 'message'), [
    ({'unit': 'codon'}, "records site.mode='pinned'"),
    ({'mode': 'pinned'}, "records site.mode='pinned'"),
    ({'mode': 'aligned', 'alignment_dirs': {**ALIGNMENT_DIRS, 'HA': 'other/HA'}},
     r"'HA': \('other/HA', 'alignments/HA'\)"),
])
def test_aligned_dataset_check_rejects_other_builds(tmp_path, dataset_site, message):
    training = site_config({'mode': 'aligned', 'alignment_dirs': ALIGNMENT_DIRS})
    dataset = save_dataset_config(tmp_path, dataset_site)
    with pytest.raises(ValueError, match=message):
        check_aligned_dataset(training, dataset, ('PB1', 'HA'))


def test_aligned_dataset_check_needs_the_dataset_config():
    training = site_config({'mode': 'aligned', 'alignment_dirs': ALIGNMENT_DIRS})
    with pytest.raises(ValueError, match='needs --dataset_resolved_config'):
        check_aligned_dataset(training, None, ('PB1', 'HA'))
