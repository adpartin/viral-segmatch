"""Contract tests: kmer_utils occurrence-col helpers delegate to the schema registry.

Runs in the segmatch env (kmer_utils imports scipy/pandas). Pytest-compatible;
also runnable directly: `python tests/test_kmer_utils.py`.
"""
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils import kmer_utils, schema  # noqa: E402


def test_occurrence_col_matches_registry():
    for a in schema.ALPHABETS:
        assert kmer_utils._occurrence_col(a) == schema.SCHEMA[a].occurrence_col
    # explicit: nt_ctg k-mers key on the contig id; nt_cds + aa on brc_fea_id
    assert kmer_utils._occurrence_col("nt_ctg") == "genbank_ctg_id"
    assert kmer_utils._occurrence_col("nt_cds") == "brc_fea_id"
    assert kmer_utils._occurrence_col("aa") == "brc_fea_id"


def test_pair_side_col_matches_registry():
    for a in schema.ALPHABETS:
        for side in ("a", "b"):
            assert kmer_utils._pair_side_col(a, side) == schema.pair_occ_col(a, side)
    assert kmer_utils._pair_side_col("nt_ctg", "a") == "ctg_a"
    assert kmer_utils._pair_side_col("aa", "b") == "brc_b"


def test_legacy_nt_alphabet_rejected():
    # 'nt' is no longer a valid alphabet (it split into nt_ctg / nt_cds).
    try:
        kmer_utils._occurrence_col("nt")
    except ValueError:
        return
    raise AssertionError("expected ValueError for legacy 'nt' alphabet")


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
