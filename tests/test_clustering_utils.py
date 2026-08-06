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
    _build_mmseqs_clust_cmd,
    _build_mmseqs_search_cmd,
    _clean_for_mmseqs,
    attach_function_to_contigs,
    compute_prot_hash,
    connected_components_from_hits,
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


def test_build_cmd_default_is_backcompat():
    # Guardrail: default args must reproduce the historical Set-Cover / linclust
    # command byte-for-byte, so existing clusters_* parquets stay reproducible.
    cmd = _build_mmseqs_clust_cmd(
        mmseqs_bin="mmseqs", subcmd="easy-linclust",
        fasta_path="f.fasta", out_prefix="out", tmp_dir="tmp", dbtype="1",
        min_seq_id=0.95, coverage=0.8, cov_mode=0, cluster_mode=0,
    )
    assert cmd == [
        "mmseqs", "easy-linclust", "f.fasta", "out", "tmp",
        "--min-seq-id", "0.95", "-c", "0.8", "--cov-mode", "0", "--dbtype", "1",
        "--cluster-mode", "0", "--seq-id-mode", "0", "--similarity-type", "2",
        "-e", "0.001", "--threads", "16",
    ]
    assert "-s" not in cmd and "--single-step-clustering" not in cmd


def test_build_cmd_ood_emits_connected_component_flags():
    # The OOD recipe emits the opt-in flags: connected-component (--cluster-mode 1),
    # sensitive prefilter (-s), single-step, and a high --max-seqs.
    cmd = _build_mmseqs_clust_cmd(
        mmseqs_bin="mmseqs", subcmd="easy-cluster",
        fasta_path="f.fasta", out_prefix="out", tmp_dir="tmp", dbtype="1",
        min_seq_id=0.95, coverage=0.8, cov_mode=0, cluster_mode=1,
        sensitivity=7.5, single_step_clustering=True, max_seqs=100000,
    )
    assert cmd[cmd.index("--cluster-mode") + 1] == "1"
    assert cmd[cmd.index("-s") + 1] == "7.5"
    assert "--single-step-clustering" in cmd
    assert cmd[cmd.index("--max-seqs") + 1] == "100000"


def test_build_search_cmd_default_is_s75():
    # Default OOD all-vs-all search: -s 7.5 prefilter, with the SAME id/cov rule the
    # verifier uses so a cluster set built from these hits verifies against one graph.
    cmd = _build_mmseqs_search_cmd(
        mmseqs_bin="mmseqs", fasta_path="f.fasta", out_tsv="h.tsv", tmp_dir="tmp",
        dbtype="1", min_seq_id=0.99, coverage=0.8, cov_mode=0, sensitivity=7.5,
    )
    assert cmd == [
        "mmseqs", "easy-search", "f.fasta", "f.fasta", "h.tsv", "tmp",
        "--min-seq-id", "0.99", "-c", "0.8", "--cov-mode", "0", "--dbtype", "1",
        "--seq-id-mode", "0", "-e", "0.001",
        "--format-output", "query,target,fident,qcov,tcov",
        "-s", "7.5", "--threads", "16",
    ]


def test_build_search_cmd_exhaustive_uses_prefilter_mode_2():
    # --exhaustive -> --prefilter-mode 2 (nofilter, provably complete); it wins over -s.
    # Also checks nt dbtype, --max-seqs, and the GPU flags.
    cmd = _build_mmseqs_search_cmd(
        mmseqs_bin="mmseqs", fasta_path="f.fasta", out_tsv="h.tsv", tmp_dir="tmp",
        dbtype="2", min_seq_id=0.95, coverage=0.8, cov_mode=0,
        sensitivity=7.5, prefilter_mode=2, max_seqs=100000, gpu=1,
    )
    assert cmd[cmd.index("--prefilter-mode") + 1] == "2"
    assert "-s" not in cmd
    assert cmd[cmd.index("--dbtype") + 1] == "2"
    assert cmd[cmd.index("--max-seqs") + 1] == "100000"
    assert cmd[cmd.index("--gpu") + 1] == "1" and "--createdb-mode" in cmd


def test_build_search_cmd_nt_emits_search_type_3():
    # Nucleotide easy-search needs --search-type 3 (aa omits it; protein is unambiguous).
    aa = _build_mmseqs_search_cmd(
        mmseqs_bin="mmseqs", fasta_path="f", out_tsv="h", tmp_dir="t",
        dbtype="1", min_seq_id=0.99, coverage=0.8, cov_mode=0, sensitivity=7.5,
    )
    assert "--search-type" not in aa
    nt = _build_mmseqs_search_cmd(
        mmseqs_bin="mmseqs", fasta_path="f", out_tsv="h", tmp_dir="t",
        dbtype="2", min_seq_id=0.99, coverage=0.8, cov_mode=0, sensitivity=7.5, search_type=3,
    )
    assert nt[nt.index("--search-type") + 1] == "3"


def test_connected_components_from_hits_union_find():
    # Graph: a-b, b-c (component {a,b,c}); d-e ({d,e}); f isolated (singleton). The
    # hit TSV is directed (both a-b and b-a) with the 5 verifier columns.
    rows = "a\tb\t1\t1\t1\nb\ta\t1\t1\t1\nb\tc\t1\t1\t1\nd\te\t1\t1\t1\n"
    with tempfile.TemporaryDirectory() as d:
        tsv = Path(d) / "hits.tsv"
        tsv.write_text(rows)
        out = connected_components_from_hits(
            tsv, ["a", "b", "c", "d", "e", "f"], alphabet="aa", cluster_id_prefix="X")
    hash_col = _COLS_BY_ALPHABET["aa"]["hash_col"]
    cid = dict(zip(out[hash_col], out["cluster_id"]))
    assert cid["a"] == cid["b"] == cid["c"]           # same component -> same id
    assert cid["d"] == cid["e"]
    assert len({cid["a"], cid["d"], cid["f"]}) == 3   # 3 distinct components
    assert cid["a"] == "X_0"                           # canonical: biggest is _0
    assert len(out) == 6 and out[hash_col].is_unique   # every node once, incl. singleton f


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
