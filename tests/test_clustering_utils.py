"""Contract tests for clustering_utils: the prot_hash invariant, the registry-
derived column map, and the nt_ctg ungate + contig->function 1-1 join. Runs in
the segmatch env. Pytest-compatible; also runnable directly:
`python tests/test_clustering_utils.py`.
"""
import hashlib
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils import schema  # noqa: E402
from src.utils.clustering_utils import (  # noqa: E402
    _ACTIVE_ALPHABETS,
    _COLS_BY_ALPHABET,
    _clean_for_mmseqs,
    attach_function_to_contigs,
    compute_prot_hash,
)


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


def test_nt_ctg_active_and_clean_passthrough():
    # nt_ctg is no longer gated: it is in the active set, its registry columns are
    # the contig molecule columns, and cleaning is nt pass-through (no UTR trim).
    assert "nt_ctg" in _ACTIVE_ALPHABETS
    assert _COLS_BY_ALPHABET["nt_ctg"] == {
        "seq_col": "ctg_dna_seq", "hash_col": "ctg_dna_hash"}
    # contig DNA passes through unchanged (IUPAC codes kept; no '*' to strip).
    assert _clean_for_mmseqs("ACGTNRYACGT", "nt_ctg") == "ACGTNRYACGT"


def _write_function_source(tmpdir, rows):
    p = Path(tmpdir) / "fn_source.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def test_attach_function_to_contigs_clean_1to1():
    # A clean 1-1 join attaches `function` and preserves every contig row + its
    # value columns, in order.
    ctg = pd.DataFrame({
        "assembly_id": ["a1", "a1", "a2"],
        "genbank_ctg_id": ["c1", "c2", "c1"],
        "ctg_dna_hash": ["h1", "h2", "h3"],
    })
    with tempfile.TemporaryDirectory() as d:
        fsrc = _write_function_source(d, {
            "assembly_id": ["a1", "a1", "a2"],
            "genbank_ctg_id": ["c1", "c2", "c1"],
            "function": ["HA", "NA", "HA"],
        })
        out = attach_function_to_contigs(ctg, fsrc)
    assert list(out["ctg_dna_hash"]) == ["h1", "h2", "h3"]
    assert list(out["function"]) == ["HA", "NA", "HA"]


def test_attach_function_to_contigs_rejects_ambiguous_source():
    # A function source with a duplicated (assembly_id, genbank_ctg_id) key would
    # let a contig map to >1 function -> must raise (the 1-1 guarantee).
    ctg = pd.DataFrame({
        "assembly_id": ["a1"], "genbank_ctg_id": ["c1"], "ctg_dna_hash": ["h1"]})
    with tempfile.TemporaryDirectory() as d:
        fsrc = _write_function_source(d, {
            "assembly_id": ["a1", "a1"],
            "genbank_ctg_id": ["c1", "c1"],
            "function": ["HA", "NA"],
        })
        raised = False
        try:
            attach_function_to_contigs(ctg, fsrc)
        except (ValueError, AssertionError):
            raised = True
    assert raised, "expected a raise on an ambiguous (duplicate-key) function source"


def test_attach_function_to_contigs_drops_unmatched():
    # Contigs with no matching function row (non-major contigs) are dropped via the
    # inner join; matched rows survive intact.
    ctg = pd.DataFrame({
        "assembly_id": ["a1", "a2"],
        "genbank_ctg_id": ["c1", "cX"],
        "ctg_dna_hash": ["h1", "hX"],
    })
    with tempfile.TemporaryDirectory() as d:
        fsrc = _write_function_source(d, {
            "assembly_id": ["a1"], "genbank_ctg_id": ["c1"], "function": ["HA"]})
        out = attach_function_to_contigs(ctg, fsrc)
    assert list(out["ctg_dna_hash"]) == ["h1"]
    assert list(out["function"]) == ["HA"]


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
