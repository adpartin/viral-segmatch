"""Canonical per-alphabet column / artifact schema — the single source of truth.

The pipeline represents each protein-coding segment three ways:

  - ``aa``     — the protein   (``prot_seq``,    ``prot_hash``)
  - ``nt_cds`` — the CDS DNA   (``cds_dna_seq``, ``cds_dna_hash``)
  - ``nt_ctg`` — the contig DNA(``ctg_dna_seq``, ``ctg_dna_hash``)

Every module that needs a per-alphabet column name, file basename, cluster dir,
membership table, or k-mer cache basename reads it from ``SCHEMA`` here, so the
convention is defined in exactly one place (and changed in exactly one place).
This replaces the ~6 drifted ``alphabet -> column`` maps that previously lived in
``clustering_utils._COLS_BY_ALPHABET``, ``_cc_helpers._MEMB{,_HASH}``,
``dataset_pairs_cc._POS_HASH``, the analysis ``_cv_*._HASH`` copies (×3),
``cluster_source._KEY``, and ``kmer_utils._{occurrence,pair_side}_col`` — the
drift between those copies was the source of the ``dna_hash`` mislabel (the
analysis universe named the CDS hash ``dna_hash`` while production named it
``cds_dna_hash``).

Naming convention (glossary ``aa/nt vs protein/DNA``): the alphabet values
(``aa``/``nt_cds``/``nt_ctg``) are alphabet/residue level — used for the enum,
cluster dirs, and k-mer cache names. The molecule names
(``prot``/``cds_dna``/``ctg_dna``) are molecule/sequence level — used for
sequence columns, hashes, and file basenames. All three hashes are md5; the
ESM-2 cache key is a separate ``sha1(prot_seq)`` namespace and is NOT modelled
here. File *directories* (``data_version``/virus roots) live in ``conf/paths``;
this module owns only the *names* (basenames + columns).
"""
from __future__ import annotations

from dataclasses import dataclass

# Canonical ordering of the three sequence representations.
ALPHABETS: tuple[str, ...] = ("aa", "nt_cds", "nt_ctg")


@dataclass(frozen=True)
class AlphabetSchema:
    """Column / artifact names for one sequence representation."""

    alphabet: str          # 'aa' | 'nt_cds' | 'nt_ctg'
    seq_col: str           # per-sequence column in the *_final source table
    hash_col: str          # md5 hash column
    file_basename: str     # Stage 1/1.5 output file (no extension)
    occurrence_col: str    # k-mer matrix INDEX key (in the source table)
    pair_occ_base: str     # pair-side k-mer join base ('brc'/'ctg' -> '<base>_{a,b}')
    cluster_dir: str       # mmseqs cluster output directory name
    memb_basename: str     # per-isolate cluster-membership parquet (no extension)
    kmer_basename: str     # k-mer feature cache basename (alphabet-tagged)


SCHEMA: dict[str, AlphabetSchema] = {
    "aa": AlphabetSchema(
        alphabet="aa",
        seq_col="prot_seq",
        hash_col="prot_hash",
        file_basename="protein_final",
        occurrence_col="brc_fea_id",
        pair_occ_base="brc",
        cluster_dir="clusters_aa",
        memb_basename="cluster_memb_aa",
        kmer_basename="kmer_features_aa",
    ),
    "nt_cds": AlphabetSchema(
        alphabet="nt_cds",
        seq_col="cds_dna_seq",
        hash_col="cds_dna_hash",
        file_basename="cds_dna_final",
        occurrence_col="brc_fea_id",     # CDS <-> protein <-> brc_fea_id are 1-1
        pair_occ_base="brc",
        cluster_dir="clusters_nt_cds",
        memb_basename="cluster_memb_nt_cds",
        kmer_basename="kmer_features_nt_cds",
    ),
    "nt_ctg": AlphabetSchema(
        alphabet="nt_ctg",
        seq_col="ctg_dna_seq",
        hash_col="ctg_dna_hash",
        file_basename="ctg_dna_final",
        occurrence_col="genbank_ctg_id",
        pair_occ_base="ctg",
        cluster_dir="clusters_nt_ctg",
        memb_basename="cluster_memb_nt_ctg",
        kmer_basename="kmer_features_nt_ctg",
    ),
}


def require(alphabet: str) -> AlphabetSchema:
    """Return the schema for ``alphabet`` or raise with the accepted set."""
    try:
        return SCHEMA[alphabet]
    except KeyError:
        raise ValueError(
            f"unknown alphabet {alphabet!r}; expected one of {ALPHABETS}") from None


def seq_col(alphabet: str) -> str:
    return require(alphabet).seq_col


def hash_col(alphabet: str) -> str:
    return require(alphabet).hash_col


def seq_col_ab(alphabet: str) -> tuple[str, str]:
    """Pair-side sequence columns, e.g. ('prot_seq_a', 'prot_seq_b')."""
    c = require(alphabet).seq_col
    return f"{c}_a", f"{c}_b"


def hash_col_ab(alphabet: str) -> tuple[str, str]:
    """Pair-side hash columns, e.g. ('cds_dna_hash_a', 'cds_dna_hash_b')."""
    c = require(alphabet).hash_col
    return f"{c}_a", f"{c}_b"


def pair_occ_col(alphabet: str, side: str) -> str:
    """Pair-side k-mer join column, e.g. ('aa', 'a') -> 'brc_a'."""
    return f"{require(alphabet).pair_occ_base}_{side}"


def build_pair_columns() -> list[str]:
    """The canonical pair-table schema (`_PAIR_COLUMNS`), built from the registry.

    Mirrors the historical column set exactly, under the new molecule-level names:
    ``seq_a -> prot_seq_a``, ``dna_seq_a -> ctg_dna_seq_a``,
    ``seq_hash_a -> prot_hash_a``, ``dna_hash_a -> ctg_dna_hash_a``;
    ``cds_dna_hash_a`` unchanged. The pair table carries SEQ columns for aa +
    nt_ctg only (the CDS sequence is not materialized per-pair — only its hash),
    and HASH columns for all three alphabets.
    """
    aa, cds, ctg = SCHEMA["aa"], SCHEMA["nt_cds"], SCHEMA["nt_ctg"]
    return [
        "pair_key",
        "assembly_id_a", "assembly_id_b",
        "brc_a", "brc_b",
        "ctg_a", "ctg_b",
        f"{aa.seq_col}_a", f"{aa.seq_col}_b",        # prot_seq_a/b   (was seq_a/b)
        f"{ctg.seq_col}_a", f"{ctg.seq_col}_b",      # ctg_dna_seq_a/b (was dna_seq_a/b)
        "seg_a", "seg_b",
        "func_a", "func_b",
        f"{aa.hash_col}_a", f"{aa.hash_col}_b",      # prot_hash_a/b   (was seq_hash_a/b)
        f"{ctg.hash_col}_a", f"{ctg.hash_col}_b",    # ctg_dna_hash_a/b (was dna_hash_a/b)
        f"{cds.hash_col}_a", f"{cds.hash_col}_b",    # cds_dna_hash_a/b (unchanged)
        "label",
        "neg_regime",
        "metadata_match_count",
    ]
