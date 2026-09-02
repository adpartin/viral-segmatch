"""CDS extraction utils.

Reconstructs per-row coding DNA (CDS) from the genome and protein metadata
tables produced by Stage 1. The recipe is documented in
`docs/methods/gto_format_reference.md` § 9 and § 5 (`location` schema).

Used by `src/preprocess/extract_cds_dna.py` (the Stage 1.5 driver that
emits `cds_dna_final.parquet`).

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

import pandas as pd


# Base -> its complement, used by `reverse_complement` for a minus-strand CDS. Case is
# preserved, so a lowercase record stays lowercase. IUPAC ambiguity codes are included because
# a record may contain them: R (A or G) complements to Y (C or T), K (G or T) to M (A or C),
# B (C/G/T) to V (A/C/G), D (A/G/T) to H (A/C/T); N, S and W are their own complements.
# `U` maps to `A`, so an RNA-spelled record complements into DNA and does not round-trip back
# to `U` -- that normalisation is intended.
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
    Raises `ValueError` on malformed input — including BV-BRC sentinel
    annotations where `length <= 0` (most commonly `-1`, used to flag
    incomplete spliced annotations on M42 / M2 / NEP / NS3 / PA-X).
    Such rows are not reconstructible from `location` alone; callers
    should let them fall into the warn-and-skip bucket rather than
    silently produce a truncated CDS.
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
        s, L = int(start), int(length)
        if s < 1:
            raise ValueError(
                f"non-positive start_1based in location entry {entry!r} "
                f"(BV-BRC uses 1-based inclusive coordinates)"
            )
        if L <= 0:
            raise ValueError(
                f"non-positive exon length in location entry {entry!r} "
                f"(BV-BRC sentinel for incomplete / un-reconstructible "
                f"spliced annotations; ~30% of M42 rows on Flu A July 2025)"
            )
        out.append((str(ctg), s, str(strand), L))

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
    strands = {ex[2] for ex in exons}
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

# NCBI translation table 1, the standard genetic code -- which is the one influenza uses, since
# it is translated by host ribosomes. Uppercase DNA triplet -> one-letter amino acid, with the
# three stop codons (TAA, TAG, TGA) mapping to '*'. All 64 ACGT triplets are present, so a
# lookup that misses means the triplet held a character outside ACGT; `_translate_codon` then
# falls through to the IUPAC expansion below.
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

# IUPAC nucleotide code -> the concrete bases it stands for. `_translate_codon` uses this to
# expand an ambiguous codon into every triplet it could be: if all of them give the same amino
# acid the residue is unambiguous despite the ambiguity (TAR is TAA or TAG, both stops), and
# otherwise the residue is 'X'. `U` expands to `T` so RNA spelling translates like DNA.
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


def check_cds_length(
    observed_lengths,
    pinned_nt: int,
    *,
    protein: str,
    min_coverage: float = 0.90,
    ) -> dict:
    """Check a pinned canonical CDS length against the population actually built.

    Per-site features need every sequence at one length, so a run pins the length from
    `conf/virus/flu.yaml` `cds_length` rather than taking the most common length of whatever
    rows it loaded -- a value that can drift between runs and make two importance maps
    non-comparable without anything failing. This re-derives the most common length from the
    data and refuses to continue if it disagrees with the pin, or if the pin covers too little
    of the population to be the canonical form.

    Args:
        observed_lengths: CDS lengths in nucleotides, one per unique sequence, for one protein
            in one population. Pass unique sequences, not rows: a heavily sampled strain would
            otherwise decide the answer.
        pinned_nt: the length from the config for this protein.
        protein: short name, used in the error text only.
        min_coverage: least share of `observed_lengths` that must equal `pinned_nt`.

    Returns:
        `{'pinned_nt', 'observed_mode_nt', 'coverage', 'n'}` -- coverage is the share at
        `pinned_nt`, not at the observed mode.

    Raises:
        ValueError: when `observed_lengths` is empty, when the most common observed length is
            not `pinned_nt`, or when coverage is below `min_coverage`.
    """
    lengths = pd.Series(list(observed_lengths))
    if lengths.empty:
        raise ValueError(f"check_cds_length: no sequences given for {protein}.")
    counts = lengths.value_counts()
    observed_mode = int(counts.index[0])
    coverage = float((lengths == pinned_nt).mean())

    if observed_mode != pinned_nt:
        raise ValueError(
            f"{protein}: config pins cds_length {pinned_nt} nt, but the most common length in "
            f"this population is {observed_mode} nt ({100 * counts.iloc[0] / len(lengths):.1f}% "
            f"of {len(lengths):,} unique sequences; the pinned length covers "
            f"{100 * coverage:.1f}%). The pin is for H3N2 and H1N1 only -- other subtypes "
            f"differ (H5N1 HA is 1704, H9/H7 HA 1683, N8/N6/N9 NA 1413), and PB1 and NS1 have "
            f"no single canonical length. Either narrow the population or add a per-subtype "
            f"entry; do not re-pin one number across subtypes."
        )
    if coverage < min_coverage:
        raise ValueError(
            f"{protein}: cds_length {pinned_nt} nt is the most common length but covers only "
            f"{100 * coverage:.1f}% of {len(lengths):,} unique sequences, below the "
            f"{100 * min_coverage:.0f}% floor. A per-site run would drop the rest, so the "
            f"population is probably a mix of forms rather than one canonical length."
        )
    return {'pinned_nt': pinned_nt, 'observed_mode_nt': observed_mode,
            'coverage': coverage, 'n': int(len(lengths))}


def compute_cds_dna_hash(cds_dna: str) -> str:
    """md5(cds_dna). Distinct from the contig-level `dna_hash` in Stage 1."""
    return hashlib.md5(cds_dna.encode('utf-8')).hexdigest()
