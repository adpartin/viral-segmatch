"""Tests for `extraction_manifest` in scripts/write_extraction_manifest.py.

Run: python -m pytest tests/test_write_extraction_manifest.py
"""
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / 'scripts'))

from write_extraction_manifest import extraction_manifest, group_suffix  # noqa: E402

from src.utils.dataset_manifest import SPLIT_FILES, dataset_file_checksums  # noqa: E402
from src.utils.experiment_utils import file_sha256  # noqa: E402

GROUP = {'host': 'Human', 'hn_subtype': 'H3N2', 'year': '2024'}
SPLIT_DIRS = ['fold_0', 'fold_1']


def write_dataset(dataset_dir: Path, text: str) -> None:
    """A dataset directory holding `resolved_config.yaml` and every split file, all with `text`."""
    dataset_dir.mkdir()
    (dataset_dir / 'resolved_config.yaml').write_text(text)
    for split_dir in SPLIT_DIRS:
        (dataset_dir / split_dir).mkdir()
        for name in SPLIT_FILES:
            (dataset_dir / split_dir / name).write_text(f'{text} {split_dir} {name}')


def source_run(tmp_path: Path, pairs=('pb1_ha', 'pb2_pb1')) -> dict:
    """A source manifest of `pairs`, each with its extraction written beside it."""
    cache = tmp_path / 'cache.npz'
    cache.write_text('cache')
    datasets = {}
    for pair in pairs:
        source_dir = tmp_path / f'run_{pair}'
        write_dataset(source_dir, f'source {pair}')
        files = dataset_file_checksums(source_dir, SPLIT_DIRS)
        datasets[f'bundle_{pair}'] = {'pair': pair, 'dataset_dir': str(source_dir), 'n_folds': 2,
                                      'split_dirs': SPLIT_DIRS, 'files': files}
        extracted_dir = tmp_path / f'run_{pair}_{group_suffix(GROUP)}'
        write_dataset(extracted_dir, f'extracted {pair}')
        info = {'source_dataset': str(source_dir), 'group': GROUP, 'split_dirs': SPLIT_DIRS,
                'source_sha256': dict(files, **{'cv_info.json': 'abc'}),
                'code': {'commit': 'extraction-commit'}}
        (extracted_dir / 'extraction_info.json').write_text(json.dumps(info))
    manifest = {'dataset_prefix': 'run_', 'manifest_path': str(tmp_path / 'run_manifest.json'),
                'manifest_sha256': 'source-sha', 'experiment_spec': {'path': 'spec.yaml'},
                'overrides': ['dataset.n_folds=2'], 'bundle_prefix': 'bundle_', 'n_folds': 2,
                'cache_files': {str(cache): file_sha256(cache)}, 'datasets': datasets}
    return manifest


def edit_extraction_info(tmp_path: Path, pair: str, **changes) -> None:
    path = tmp_path / f'run_{pair}_{group_suffix(GROUP)}' / 'extraction_info.json'
    info = json.loads(path.read_text())
    info.update(changes)
    path.write_text(json.dumps(info))


def test_group_suffix_joins_the_values_in_lower_case():
    assert group_suffix(GROUP) == 'human_h3n2_2024'


def test_manifest_lists_every_extraction_with_its_own_checksums(tmp_path):
    source = source_run(tmp_path)
    manifest, failures = extraction_manifest(source, GROUP)
    assert failures == []
    assert manifest['dataset_prefix'] == 'run_human_h3n2_2024_'
    assert manifest['source_manifest'] == {'path': source['manifest_path'], 'sha256': 'source-sha'}
    for key in ('experiment_spec', 'overrides', 'bundle_prefix', 'n_folds', 'cache_files'):
        assert manifest[key] == source[key]
    assert sorted(manifest['datasets']) == ['bundle_pb1_ha', 'bundle_pb2_pb1']
    entry = manifest['datasets']['bundle_pb1_ha']
    extracted_dir = tmp_path / 'run_pb1_ha_human_h3n2_2024'
    assert entry['dataset_dir'] == str(extracted_dir)
    assert entry['files'] == dataset_file_checksums(extracted_dir, SPLIT_DIRS)
    assert entry['files'] != source['datasets']['bundle_pb1_ha']['files']
    assert entry['extraction_commit'] == 'extraction-commit'


def test_a_missing_extraction_is_reported(tmp_path):
    source = source_run(tmp_path, pairs=('pb1_ha',))
    source['datasets']['bundle_ha_na'] = dict(source['datasets']['bundle_pb1_ha'],
                                              pair='ha_na', dataset_dir=str(tmp_path / 'run_ha_na'))
    manifest, failures = extraction_manifest(source, GROUP)
    assert failures == [f"{tmp_path / 'run_ha_na_human_h3n2_2024'}: missing"]
    assert list(manifest['datasets']) == ['bundle_pb1_ha']


def test_an_extraction_of_other_files_is_reported(tmp_path):
    source = source_run(tmp_path)
    (tmp_path / 'run_pb1_ha' / 'fold_1' / 'test_pairs.csv').write_text('rewritten')
    source['datasets']['bundle_pb1_ha']['files'] = dataset_file_checksums(tmp_path / 'run_pb1_ha',
                                                                          SPLIT_DIRS)
    _, failures = extraction_manifest(source, GROUP)
    assert failures == ["run_pb1_ha_human_h3n2_2024: the extraction read 1 source files with "
                        "checksums other than the source manifest records, e.g. "
                        "['fold_1/test_pairs.csv']"]


def test_another_source_group_or_split_list_is_reported(tmp_path):
    source = source_run(tmp_path)
    edit_extraction_info(tmp_path, 'pb1_ha', source_dataset=str(tmp_path / 'run_pb2_pb1'),
                         group=dict(GROUP, year='2023'), split_dirs=['fold_0'])
    manifest, failures = extraction_manifest(source, GROUP)
    assert len(failures) == 3
    assert failures[0].startswith('run_pb1_ha_human_h3n2_2024: extracted from ')
    assert 'extracted group' in failures[1]
    assert "split directories ['fold_0']" in failures[2]
    assert list(manifest['datasets']) == ['bundle_pb2_pb1']


def test_a_changed_cache_file_is_reported(tmp_path):
    source = source_run(tmp_path)
    (tmp_path / 'cache.npz').write_text('rebuilt')
    _, failures = extraction_manifest(source, GROUP)
    assert failures == [f"{tmp_path / 'cache.npz'}: changed since the manifest was written"]
