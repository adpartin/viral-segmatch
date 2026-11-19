"""Validate the final aligned site settings from a bundle and its command-line overrides."""
import pytest

from src.utils.config_hydra import compose_config, get_virus_config_hydra


@pytest.fixture
def config_dir(tmp_path):
    bundles = tmp_path / 'bundles'
    bundles.mkdir()
    (bundles / 'example.yaml').write_text(
        'virus: {}\nsite: {mode: aligned, unit: codon, encoding: onehot}\n')
    return str(tmp_path)


def test_direct_bundle_load_rejects_unsupported_encoding(config_dir):
    with pytest.raises(ValueError, match="Got site.unit='codon' and site.encoding='onehot'"):
        get_virus_config_hydra('example', config_path=config_dir)


@pytest.mark.parametrize(('mode', 'unit', 'encoding'), [
    ('aligned', 'codon', 'ordinal'),
    ('pinned', 'nt', 'onehot'),
    ('pinned', 'aa', 'ordinal'),
])
def test_supported_final_settings(config_dir, mode, unit, encoding):
    overrides = [f'site.mode={mode}', f'site.unit={unit}', f'site.encoding={encoding}']
    config = compose_config('example', overrides, config_path=config_dir)
    assert (config.site.mode, config.site.unit, config.site.encoding) == (mode, unit, encoding)


@pytest.mark.parametrize(('unit', 'encoding'), [
    ('nt', 'ordinal'), ('aa', 'ordinal'), ('codon', 'onehot'), ('nt', 'onehot'),
])
def test_aligned_settings_report_supplied_values(config_dir, unit, encoding):
    overrides = [f'site.unit={unit}', f'site.encoding={encoding}']
    with pytest.raises(ValueError) as error:
        compose_config('example', overrides, config_path=config_dir)
    assert f"Got site.unit={unit!r} and site.encoding={encoding!r}" in str(error.value)
