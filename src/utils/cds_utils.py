"""CDS extraction utilities.

Reconstructs per-row coding DNA (CDS) from the genome and protein metadata
tables produced by Stage 1. The recipe is documented in
`docs/methods/gto_format_reference.md` § 9 and § 5 (`location` schema).

Used by `src/preprocess/extract_cds_dna.py` (the Stage 1.5 driver that
emits `cds_final.parquet`) and by `src/utils/clustering_utils.py` for the
nt-level cluster_disjoint follow-up (Experiment B-nt; see
`docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` § B-nt).

Conventions:
- 1-based inclusive starts (GTO/GenBank) are translated to 0-based half-open
  Python slices via `start - 1 : start - 1 + length`.
- Multi-entry `location` lists are spliced in the order they appear (one
  entry per exon). Flu A exercises this for M2, NEP, M42, NS3, PA-X, and a
  rare PB2 splice variant.
- All flu CDS in our corpus are on the `+` strand, but `-` strand is
  supported via reverse-complement for portability.
"""

from __future__ import annotations

import ast
import hashlib


_COMPLEMENT = {
    'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
    'a': 't', 't': 'a', 'g': 'c', 'c': 'g',
    'N': 'N', 'n': 'n',
    'R': 'Y', 'Y': 'R', 'S': 'S', 'W': 'W', 'K': 'M', 'M': 'K',
    'B': 'V', 'V': 'B', 'D': 'H', 'H': 'D',
    'U': 'A', 'u': 'a',
}


def reverse_complement(dna: str) -> str:
    return ''.join(_COMPLEMENT.get(b, b) for b in reversed(dna))


def parse_location(loc: object) -> list[tuple[str, int, str, int]]:
    """Parse a `protein_final.location` field into a list of exon spans.

    Accepts either the in-memory list form or the CSV repr-string form.
    Each returned tuple is `(contig_id, start_1based, strand, length)`.
    Raises `ValueError` on malformed input.
    """
    if isinstance(loc, str):
        try:
            loc = ast.literal_eval(loc)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"could not parse location string: {loc!r}") from e
    if not isinstance(loc, (list, tuple)) or len(loc) == 0:
        raise ValueError(f"location must be a non-empty list, got {loc!r}")
    out: list[tuple[str, int, str, int]] = []
    for entry in loc:
        if len(entry) != 4:
            raise ValueError(f"location entry must have 4 fields, got {entry!r}")
        ctg, start, strand, length = entry
        out.append((str(ctg), int(start), str(strand), int(length)))
    return out


def extract_cds_dna(contig_dna: str, location: object) -> str:
    """Return the spliced CDS DNA carved out of `contig_dna` per `location`.

    Exons are concatenated in the order they appear in `location`. If
    every exon's strand is `-`, the final concatenation is reverse-
    complemented. Mixed strands within one `location` are rejected.

    Raises `ValueError` if `location` is empty, malformed, mixed-strand,
    or asks for a slice that goes off the end of `contig_dna`.
    """
    exons = parse_location(location)
    strands = {e[2] for e in exons}
    if len(strands) != 1:
        raise ValueError(f"mixed strands within one location: {exons}")
    strand = strands.pop()
    pieces = []
    for _ctg, start, _strand, length in exons:
        end = start - 1 + length
        if start < 1 or end > len(contig_dna):
            raise ValueError(
                f"exon span [{start},{end}] out of contig (len={len(contig_dna)})"
            )
        pieces.append(contig_dna[start - 1 : end])
    cds_dna = ''.join(pieces)
    if strand == '-':
        cds_dna = reverse_complement(cds_dna)
    return cds_dna


_CODON_TABLE_1 = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


_IUPAC_EXPAND = {
    'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'U': 'T',
    'R': 'AG', 'Y': 'CT', 'S': 'CG', 'W': 'AT', 'K': 'GT', 'M': 'AC',
    'B': 'CGT', 'D': 'AGT', 'H': 'ACT', 'V': 'ACG',
    'N': 'ACGT',
}


def _translate_codon(codon: str) -> str:
    """Translate one codon, resolving IUPAC ambiguities synonymously.

    Returns the resolved amino acid if every IUPAC expansion yields the
    same residue (e.g. `YTG -> {CTG, TTG} -> {L}`); otherwise 'X'.
    Unrecognized bases also produce 'X'.
    """
    codon = codon.upper()
    aa = _CODON_TABLE_1.get(codon)
    if aa is not None:
        return aa
    try:
        bases = [_IUPAC_EXPAND[b] for b in codon]
    except KeyError:
        return 'X'
    resolved: set[str] = set()
    for b1 in bases[0]:
        for b2 in bases[1]:
            for b3 in bases[2]:
                aa = _CODON_TABLE_1.get(b1 + b2 + b3)
                if aa is None:
                    return 'X'
                resolved.add(aa)
                if len(resolved) > 1:
                    return 'X'
    return resolved.pop()


def translate_dna(cds_dna: str) -> str:
    """Translate `cds_dna` via NCBI translation table 1 (standard code).

    IUPAC ambiguity bases (R, Y, S, W, K, M, B, D, H, V, N) are resolved
    per codon: if every expansion yields the same amino acid
    (e.g. `ytg -> {CTG, TTG} -> {L}`), that residue is used; otherwise
    the codon becomes 'X'. The terminal stop codon, if any, becomes '*'.
    Raises `ValueError` if `len(cds_dna) % 3 != 0`.

    Flu uses standard code 1 despite the GTO field reporting
    `genetic_code = 11`; see `docs/methods/gto_format_reference.md § 2.1`.
    """
    if len(cds_dna) % 3 != 0:
        raise ValueError(
            f"CDS length {len(cds_dna)} is not a multiple of 3"
        )
    return ''.join(_translate_codon(cds_dna[i : i + 3])
                   for i in range(0, len(cds_dna), 3))


def compute_cds_dna_hash(cds_dna: str) -> str:
    """md5(cds_dna). Distinct from the contig-level `dna_hash` in Stage 1."""
    return hashlib.md5(cds_dna.encode('utf-8')).hexdigest()
