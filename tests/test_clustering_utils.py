"""Contract tests for clustering_utils: the prot_hash invariant + the registry-
derived column map. Runs in the segmatch env. Pytest-compatible; also runnable
directly: `python tests/test_clustering_utils.py`.
"""
import hashlib
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils import schema  # noqa: E402
from src.utils.clustering_utils import compute_prot_hash, _COLS_BY_ALPHABET  # noqa: E402


def test_compute_prot_hash_is_plain_md5_no_strip():
    # CRITICAL join invariant: prot_hash == md5(prot_seq) with NO rstrip. This is
    # exactly what protein_final.prot_hash and clusters_aa.prot_hash are built on;
    # a future rstrip('*') "fix" would silently break every cluster join (the
    # rstrip belongs in clean_aa_for_mmseqs, applied to the FASTA AFTER hashing).
    for s in ("ACDEFG", "ACDEFG*", "MK*K", "ACDE"):
        assert compute_prot_hash(s) == hashlib.md5(s.encode()).hexdigest()
    # a trailing stop ('*') must change the hash (i.e. it is not stripped away)
    assert compute_prot_hash("ACDE*") != compute_prot_hash("ACDE")


def test_cols_by_alphabet_matches_registry():
    # The clustering exporter's column map is derived from the single source of
    # truth; it must equal the registry for every alphabet.
    assert set(_COLS_BY_ALPHABET) == set(schema.ALPHABETS)
    for a, s in schema.SCHEMA.items():
        assert _COLS_BY_ALPHABET[a] == {"seq_col": s.seq_col, "hash_col": s.hash_col}


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
