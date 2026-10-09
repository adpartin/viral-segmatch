"""Tests for the manifest-level checks of `src/analysis/audit_pair_datasets.py`.

Covers:
  1. A complete manifest built with the audited spec passes
  2. A different spec, a failed build, a duplicate pair, a missing pair and an unselected pair are
     each reported
  3. Without a pair filter, every capacity pair of the population needs a dataset
  4. run_pair_and_population builds the pair label and the capacity table's population label

Run: python -m pytest tests/test_audit_pair_datasets.py
"""
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.analysis.audit_pair_datasets import (  # noqa: E402
    manifest_failures,
    run_pair_and_population,
)

SPEC = {'path': 'spec.yaml', 'sha256': 's' * 64, 'expect': {'dataset.n_folds': 2}}
FOLDS = ['fold_0', 'fold_1']
CONFIG = {
    'virus': {'function_short_names': {'RNA-dependent RNA polymerase PB2 subunit': 'PB2',
                                       'Hemagglutinin precursor': 'HA'}},
    'dataset': {'schema_pair': ['RNA-dependent RNA polymerase PB2 subunit',
                                'Hemagglutinin precursor'],
                'host': ['Human'], 'hn_subtype': ['H3N2'], 'year': [2024], 'year_range': None},
}


def _manifest(datasets: dict, pairs, results=None) -> dict:
    return {
        'experiment_spec': {'path': 'spec.yaml', 'sha256': SPEC['sha256']},
        'n_folds': 2, 'cache_files': {}, 'bundle_prefix': 'flu_28p_codon_', 'pairs': pairs,
        'results': results or {bundle: 0 for bundle in datasets},
        'dataset_directories': {bundle: entry['dataset_dir'] for bundle, entry in datasets.items()},
        'datasets': datasets,
    }


def _entry(pair, dataset_dir='/nonexistent'):
    return {'pair': pair, 'split_dirs': FOLDS, 'dataset_dir': dataset_dir}


def test_complete_manifest_passes():
    manifest = _manifest({'flu_28p_codon_pb2_ha': _entry('pb2_ha')}, ['pb2_ha'])
    assert manifest_failures(manifest, SPEC, pd.DataFrame()) == []


def test_each_problem_is_reported():
    datasets = {'flu_28p_codon_pb2_ha': _entry('pb2_ha'),
                'flu_28p_codon_pb2_ha_copy': _entry('pb2_ha'),
                'flu_28p_codon_ha_na': _entry('ha_na')}
    manifest = _manifest(datasets, ['pb2_ha', 'pb2_np'],
                         results={'flu_28p_codon_pb2_ha': 0, 'flu_28p_codon_pb2_ha_copy': 0,
                                  'flu_28p_codon_ha_na': 0, 'flu_28p_codon_pb2_np': 1})
    manifest['experiment_spec']['sha256'] = 'x' * 64
    failures = ' | '.join(manifest_failures(manifest, SPEC, pd.DataFrame()))
    for fragment in ('experiment spec differs', 'pb2_np: build exited 1', 'pb2_ha: 2 datasets',
                     'pb2_np: no dataset', 'ha_na: not a selected pair'):
        assert fragment in failures


def test_unfiltered_manifest_needs_every_capacity_pair(tmp_path):
    OmegaConf.save(OmegaConf.create(CONFIG), tmp_path / 'resolved_config.yaml')
    manifest = _manifest({'flu_28p_codon_pb2_ha': _entry('pb2_ha', str(tmp_path))}, None)
    capacity = pd.DataFrame({'population': ['Human-H3N2-2024', 'Human-H3N2-2024', 'Human-H1N1-2024'],
                             'Schema pair': ['PB2-HA', 'PB2-NP', 'PB1-PA']})
    failures = manifest_failures(manifest, SPEC, capacity)
    assert failures == ["capacity pairs of Human-H3N2-2024 without a dataset: ['pb2_np']"]


def test_run_pair_and_population():
    assert run_pair_and_population(OmegaConf.create(CONFIG)) == ('PB2-HA', 'Human-H3N2-2024')
