"""Tests for `src/utils/dataset_manifest.py` and the trainer's input checksums.

Covers:
  1. read_split_dirs reads cv_info.json, falls back to the root for a single split, and rejects
     an empty directory
  2. dataset_file_checksums keys files by their path in the run directory, and changed_files
     reports missing and changed files
  3. fold_count_failures compares the manifest's fold count and split directories with the spec
  4. fit_input_checksums produces exactly the dict the trainer's input_checksums records, with no
     site caches and with aligned site caches
  5. pair_token turns a pair label into the token used in names
  6. site_cache_files names the same cache files the trainer reads, in pinned and aligned mode
  7. read_pair_table_for_rewrite writes a pair table back with its identifiers, 'NA' and labels
     unchanged

Run: python -m pytest tests/test_dataset_manifest.py
"""
import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.models.train_pair_baselines import input_checksums  # noqa: E402
from src.utils.dataset_manifest import (  # noqa: E402
    SPLIT_FILES,
    changed_files,
    dataset_file_checksums,
    fit_input_checksums,
    fold_count_failures,
    pair_cache_checksums,
    pair_token,
    read_pair_table_for_rewrite,
    read_split_dirs,
    site_cache_files,
)
from src.utils.experiment_utils import file_sha256  # noqa: E402
from src.utils.site_utils import configured_site_cache_paths  # noqa: E402

SHORT_NAMES = {'PB1 function': 'PB1', 'HA function': 'HA'}


def _make_dataset(root: Path, n_folds: int) -> Path:
    """A minimal CV dataset: resolved_config.yaml, cv_info.json and the split files per fold."""
    root.mkdir()
    (root / 'resolved_config.yaml').write_text('dataset:\n  n_folds: %d\n' % n_folds)
    fold_dirs = [f'fold_{i}' for i in range(n_folds)]
    (root / 'cv_info.json').write_text(json.dumps({'fold_dirs': fold_dirs}))
    for fold in fold_dirs:
        (root / fold).mkdir()
        for name in SPLIT_FILES:
            (root / fold / name).write_text(f'{fold}/{name}\n')
    return root


def test_read_split_dirs(tmp_path):
    assert read_split_dirs(_make_dataset(tmp_path / 'cv', 3)) == ['fold_0', 'fold_1', 'fold_2']
    single = tmp_path / 'single'
    single.mkdir()
    (single / 'train_pairs.csv').write_text('x\n')
    assert read_split_dirs(single) == ['.']
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(ValueError):
        read_split_dirs(empty)


def test_pair_table_rewrite_keeps_values(tmp_path):
    """A pair table read for rewriting must come back out byte-for-byte on its values.

    `ctg_a` holds contig identifiers that look numeric, so inferred types would write
    `11320.955490` back as `11320.95549`, and `func_a` holds `NA` for Neuraminidase, which a
    default read turns into NaN.
    """
    source = tmp_path / 'train_pairs.csv'
    source.write_text('pair_key,ctg_a,ctg_b,func_a,label\n'
                      'k1,11320.955490,2044004.10,NA,1\n'
                      'k2,11320.683010,11320.68301,Hemagglutinin precursor,0\n')
    table = read_pair_table_for_rewrite(source)

    assert list(table.func_a) == ['NA', 'Hemagglutinin precursor']
    assert list(table.loc[table.label == 0, 'pair_key']) == ['k2']

    rewritten = tmp_path / 'rewritten.csv'
    table.to_csv(rewritten, index=False)
    assert rewritten.read_text() == source.read_text()


def test_checksums_and_changes(tmp_path):
    root = _make_dataset(tmp_path / 'ds', 2)
    files = dataset_file_checksums(root, ['fold_0', 'fold_1'])
    assert 'resolved_config.yaml' in files and 'fold_1/test_pairs.csv' in files
    assert len(files) == 1 + 2 * len(SPLIT_FILES)
    assert changed_files(files, base=root) == []
    (root / 'fold_0' / 'train_pairs.csv').write_text('edited\n')
    (root / 'fold_1' / 'val_pairs.csv').unlink()
    problems = changed_files(files, base=root)
    assert len(problems) == 2
    assert any('fold_0/train_pairs.csv' in p and 'changed' in p for p in problems)
    assert any('fold_1/val_pairs.csv' in p and 'missing' in p for p in problems)


def test_fold_count_failures():
    spec = {'path': 'spec.yaml', 'expect': {'dataset.n_folds': 5}}
    good = {'n_folds': 5, 'datasets': {'b': {'split_dirs': [f'fold_{i}' for i in range(5)]}}}
    assert fold_count_failures(good, spec) == []
    bad = {'n_folds': 4, 'datasets': {'b': {'split_dirs': [f'fold_{i}' for i in range(4)]}}}
    assert len(fold_count_failures(bad, spec)) == 2


def test_fit_inputs_match_trainer(tmp_path):
    root = _make_dataset(tmp_path / 'ds', 2)
    entry = {'dataset_dir': str(root), 'files': dataset_file_checksums(root, ['fold_0', 'fold_1'])}
    expected = fit_input_checksums(entry, 'fold_1', {})
    recorded = input_checksums(root / 'fold_1', [], root / 'resolved_config.yaml')
    assert recorded == expected


def test_pair_token():
    assert pair_token('PB2-HA') == 'pb2_ha'
    assert pair_token('M1-NS1') == 'm1_ns1'


def site_config(tmp_path, mode: str):
    """A site-feature run config for PB1-HA in `mode`, with every cache under `tmp_path`."""
    return OmegaConf.create({
        'paths': {'data_dir': str(tmp_path)},
        'virus': {'virus_name': 'flu', 'data_version': 'v1',
                  'cds_length': {'PB1': {'nt': 2274}, 'HA': {'nt': 1701}},
                  'function_short_names': SHORT_NAMES},
        'site': {'mode': mode, 'unit': 'codon',
                 'alignment_dirs': {'PB1': str(tmp_path / 'PB1'), 'HA': str(tmp_path / 'HA')}},
        'training': {'feature_source': 'site'},
        'dataset': {'schema_pair': list(SHORT_NAMES)},
    })


def trainer_cache_files(config) -> list:
    """The cache files `train_pair_baselines` checksums for a PB1-HA fit."""
    files = [path for protein in ('PB1', 'HA')
             for path in configured_site_cache_paths(config, protein).values()]
    return files


@pytest.mark.parametrize('mode', ['pinned', 'aligned'])
def test_site_cache_files_match_the_trainer_cache_files(tmp_path, mode):
    config = site_config(tmp_path, mode)
    trainer_files = [path.resolve() for path in trainer_cache_files(config)]
    assert site_cache_files(config) == trainer_files


def test_site_cache_files_follow_site_mode(tmp_path):
    aligned = site_cache_files(site_config(tmp_path, 'aligned'))
    assert [path.parent for path in aligned] == [tmp_path / 'PB1'] * 3 + [tmp_path / 'HA'] * 3
    assert aligned[0].name == 'site_features_codon_PB1_aligned.npz'
    pinned = site_cache_files(site_config(tmp_path, 'pinned'))
    assert pinned[0] == tmp_path / 'embeddings' / 'flu' / 'v1' / 'site_features_codon_PB1_2274nt.npz'
    kmer = site_config(tmp_path, 'aligned')
    kmer.training.feature_source = 'kmer'
    assert site_cache_files(kmer) == []


def test_aligned_fit_inputs_match_trainer(tmp_path):
    root = _make_dataset(tmp_path / 'ds', 2)
    config = site_config(tmp_path, 'aligned')
    for path in site_cache_files(config):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'{path.name}\n')
    manifest = {'cache_files': {str(path): file_sha256(path) for path in site_cache_files(config)}}
    entry = {'dataset_dir': str(root), 'files': dataset_file_checksums(root, ['fold_0', 'fold_1'])}
    expected = fit_input_checksums(entry, 'fold_1', pair_cache_checksums(manifest, config))
    recorded = input_checksums(root / 'fold_1', trainer_cache_files(config),
                               root / 'resolved_config.yaml')
    assert recorded == expected
    assert len(recorded) == 1 + len(SPLIT_FILES) + 6
