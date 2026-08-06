"""Contract tests for the opt-in `dataset.molecule` -> alphabet-axis resolution
(`config_hydra._resolve_molecule_alphabets`). Plan §13. Runs in the segmatch env;
pytest-compatible and also runnable directly: `python tests/test_molecule_resolution.py`.
"""
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from omegaconf import OmegaConf  # noqa: E402
from src.utils.config_hydra import _resolve_molecule_alphabets  # noqa: E402


def _cfg(d):
    return OmegaConf.create(d)


def test_molecule_derives_all_three_when_absent():
    c = _cfg({'dataset': {'molecule': 'nt_cds', 'split_strategy': {'mode': 'cluster_disjoint'}},
              'kmer': {'k': 6}})
    _resolve_molecule_alphabets(c)
    assert c.dataset.split_strategy.cluster_alphabet == 'nt_cds'
    assert c.dataset.split_strategy.pair_key_alphabet == 'nt_cds'
    assert c.kmer.alphabet == 'nt_cds'


def test_nt_ctg_molecule_all_three():
    c = _cfg({'dataset': {'molecule': 'nt_ctg', 'split_strategy': {'mode': 'cluster_disjoint'}},
              'kmer': {'k': 6}})
    _resolve_molecule_alphabets(c)
    assert c.dataset.split_strategy.cluster_alphabet == 'nt_ctg'
    assert c.dataset.split_strategy.pair_key_alphabet == 'nt_ctg'
    assert c.kmer.alphabet == 'nt_ctg'


def test_explicit_cluster_override_diverges_then_guard_blocks():
    # explicit cluster_alphabet diverges from molecule -> mismatch -> raise (no allow flag)
    c = _cfg({'dataset': {'molecule': 'nt_cds',
                          'split_strategy': {'mode': 'cluster_disjoint', 'cluster_alphabet': 'nt_ctg'}},
              'kmer': {'k': 6}})
    raised = False
    try:
        _resolve_molecule_alphabets(c)
    except ValueError:
        raised = True
    assert raised, "expected guard to block molecule/cluster mismatch"


def test_allow_alphabet_mismatch_permits_divergence():
    c = _cfg({'dataset': {'molecule': 'nt_cds', 'allow_alphabet_mismatch': True,
                          'split_strategy': {'mode': 'cluster_disjoint', 'cluster_alphabet': 'nt_ctg'}},
              'kmer': {'k': 6}})
    _resolve_molecule_alphabets(c)  # must NOT raise
    assert c.dataset.split_strategy.cluster_alphabet == 'nt_ctg'   # explicit override kept
    assert c.dataset.split_strategy.pair_key_alphabet == 'nt_cds'  # derived
    assert c.kmer.alphabet == 'nt_cds'                              # authoritative


def test_matching_explicit_override_is_fine():
    c = _cfg({'dataset': {'molecule': 'nt_cds',
                          'split_strategy': {'mode': 'cluster_disjoint', 'pair_key_alphabet': 'nt_cds'}},
              'kmer': {'k': 6}})
    _resolve_molecule_alphabets(c)  # all agree -> no raise
    assert c.dataset.split_strategy.pair_key_alphabet == 'nt_cds'


def test_no_molecule_is_noop():
    c = _cfg({'dataset': {'split_strategy': {'mode': 'seq_disjoint'}},
              'kmer': {'alphabet': 'nt_ctg', 'k': 6}})
    _resolve_molecule_alphabets(c)
    assert OmegaConf.select(c, 'dataset.split_strategy.cluster_alphabet') is None
    assert OmegaConf.select(c, 'dataset.split_strategy.pair_key_alphabet') is None
    assert c.kmer.alphabet == 'nt_ctg'  # untouched


def test_invalid_molecule_raises():
    c = _cfg({'dataset': {'molecule': 'protein'}, 'kmer': {'k': 6}})
    raised = False
    try:
        _resolve_molecule_alphabets(c)
    except ValueError:
        raised = True
    assert raised, "expected ValueError for unknown molecule"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
