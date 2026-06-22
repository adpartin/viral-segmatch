"""Tests for the canonical per-alphabet schema registry (src/utils/schema.py).

Dependency-free: run with `python tests/test_schema.py` (also pytest-compatible).
"""
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils import schema  # noqa: E402


def test_alphabets():
    assert schema.ALPHABETS == ("aa", "nt_cds", "nt_ctg")
    assert set(schema.SCHEMA) == set(schema.ALPHABETS)


def test_hash_cols():
    assert schema.hash_col("aa") == "prot_hash"
    assert schema.hash_col("nt_cds") == "cds_dna_hash"
    assert schema.hash_col("nt_ctg") == "ctg_dna_hash"


def test_seq_cols():
    assert schema.seq_col("aa") == "prot_seq"
    assert schema.seq_col("nt_cds") == "cds_dna_seq"
    assert schema.seq_col("nt_ctg") == "ctg_dna_seq"


def test_ab_helpers():
    assert schema.hash_col_ab("nt_cds") == ("cds_dna_hash_a", "cds_dna_hash_b")
    assert schema.seq_col_ab("aa") == ("prot_seq_a", "prot_seq_b")
    assert schema.pair_occ_col("nt_ctg", "a") == "ctg_a"
    assert schema.pair_occ_col("aa", "b") == "brc_b"
    assert schema.pair_occ_col("nt_cds", "a") == "brc_a"  # CDS<->protein 1-1


def test_files_and_artifacts():
    assert schema.SCHEMA["aa"].file_basename == "protein_final"
    assert schema.SCHEMA["nt_cds"].file_basename == "cds_dna_final"
    assert schema.SCHEMA["nt_ctg"].file_basename == "ctg_dna_final"
    assert schema.SCHEMA["nt_cds"].cluster_dir == "clusters_nt_cds"
    assert schema.SCHEMA["nt_ctg"].cluster_dir == "clusters_nt_ctg"
    assert schema.SCHEMA["nt_ctg"].kmer_basename == "kmer_features_nt_ctg"
    assert schema.SCHEMA["nt_cds"].occurrence_col == "brc_fea_id"
    assert schema.SCHEMA["nt_ctg"].occurrence_col == "genbank_ctg_id"


def test_build_pair_columns():
    cols = schema.build_pair_columns()
    assert len(cols) == 24, cols
    # new molecule-level names present
    for c in ["prot_seq_a", "ctg_dna_seq_a", "prot_hash_a", "ctg_dna_hash_a",
              "cds_dna_hash_a", "prot_seq_b", "prot_hash_b"]:
        assert c in cols, c
    # old names gone
    for old in ["seq_a", "dna_seq_a", "seq_hash_a", "dna_hash_a", "seq_b", "dna_hash_b"]:
        assert old not in cols, old
    # invariant (non-per-alphabet) columns retained
    for c in ["pair_key", "assembly_id_a", "brc_a", "ctg_a", "seg_a", "func_a",
              "label", "neg_regime", "metadata_match_count"]:
        assert c in cols, c


def test_unknown_alphabet():
    try:
        schema.require("protein")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown alphabet")


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
