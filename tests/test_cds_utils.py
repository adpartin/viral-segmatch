"""Tests for `src/utils/cds_utils.py`.

Covers:
  1. parse_location accepts both repr-string and list forms
  2. Unspliced extraction (PB2-style single-exon)
  3. Spliced extraction (M2-style two-exon, intron skipped)
  4. Strand `-` reverse-complement
  5. translate_dna round-trips through a CDS
  6. Validation: malformed location, off-end span, mixed strands, bad length
  7. Hash determinism
  8. Real-corpus spot check (sampled rows) — translate-back must match
     the original protein modulo terminal `*` handling.

Run: python tests/test_cds_utils.py
"""
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

import pandas as pd

from src.utils.cds_utils import (
    compute_cds_dna_hash,
    extract_cds_dna,
    parse_location,
    reverse_complement,
    translate_dna,
)


def _synth_contig(prefix_len: int, cds: str, suffix_len: int) -> str:
    """Build a synthetic contig with a 5' UTR, CDS, and 3' UTR."""
    return 'A' * prefix_len + cds + 'A' * suffix_len


def test_parse_location_repr_string():
    s = "[['1406633.10', '16', '+', '2280']]"
    out = parse_location(s)
    assert out == [('1406633.10', 16, '+', 2280)], out


def test_parse_location_list():
    out = parse_location([['c', '14', '+', 26], ['c', 728, '+', 268]])
    assert out == [('c', 14, '+', 26), ('c', 728, '+', 268)], out


def test_parse_location_rejects_empty():
    try:
        parse_location([])
    except ValueError:
        return
    raise AssertionError('expected ValueError on empty location')


def test_parse_location_rejects_wrong_arity():
    try:
        parse_location([['c', '16', '+']])
    except ValueError:
        return
    raise AssertionError('expected ValueError on 3-tuple entry')


def test_unspliced_extraction_matches_doc():
    """PB2-style: location = [[ctg, 16, '+', 2280]] -> contig[15:2295]."""
    cds = 'ATG' + 'AAA' * 758 + 'TAG'        # 2280 nt = 760 codons
    contig = _synth_contig(prefix_len=15, cds=cds, suffix_len=21)
    assert len(contig) == 2316
    location = [['c', '16', '+', '2280']]
    out = extract_cds_dna(contig, location)
    assert out == cds, f'unspliced: got len={len(out)}, expected {len(cds)}'


def test_spliced_extraction_skips_intron():
    """M2-style: exon1 = nt 14..39 (26 nt), exon2 = nt 728..995 (268 nt).

    Total 294 nt; intron of 688 nt between them must be excluded.
    """
    exon1 = 'ATG' + 'C' * 23                 # 26 nt
    intron = 'T' * 688                       # 40..727 inclusive = 688 nt
    exon2 = 'G' * 265 + 'TAA'                # 268 nt
    contig = 'A' * 13 + exon1 + intron + exon2 + 'A' * 32
    location = [['c', 14, '+', 26], ['c', 728, '+', 268]]
    out = extract_cds_dna(contig, location)
    assert out == exon1 + exon2, (
        f'spliced: got len={len(out)} (=== {out[:20]}...{out[-20:]} ===), '
        f'expected {len(exon1 + exon2)}'
    )
    assert 'T' not in out[26:294 - 5], 'intron contamination'


def test_strand_minus_reverse_complements():
    cds_plus = 'ATGAAATAG'
    rc = reverse_complement(cds_plus)
    assert rc == 'CTATTTCAT'
    contig = 'CCC' + cds_plus + 'GGG'
    location = [['c', 4, '-', 9]]
    out = extract_cds_dna(contig, location)
    assert out == rc, f'strand-: got {out}, expected {rc}'


def test_reverse_complement_handles_ambiguity():
    assert reverse_complement('ATGCN') == 'NGCAT'
    assert reverse_complement('R') == 'Y' and reverse_complement('Y') == 'R'


def test_extract_rejects_off_end_span():
    contig = 'A' * 20
    try:
        extract_cds_dna(contig, [['c', 15, '+', 10]])  # span 15..24, contig len 20
    except ValueError:
        return
    raise AssertionError('expected ValueError on off-end span')


def test_extract_rejects_mixed_strands():
    contig = 'A' * 50
    try:
        extract_cds_dna(contig, [['c', 1, '+', 10], ['c', 20, '-', 10]])
    except ValueError:
        return
    raise AssertionError('expected ValueError on mixed strands')


def test_translate_dna_round_trip():
    # ATG (M) + TTT (F) + GGC (G) + TAA (*)
    cds = 'ATGTTTGGCTAA'
    assert translate_dna(cds) == 'MFG*'


def test_translate_dna_ambiguity_becomes_X():
    cds = 'ATGNNNTAA'
    assert translate_dna(cds) == 'MX*'


def test_translate_dna_rejects_non_codon_length():
    try:
        translate_dna('ATGA')
    except ValueError:
        return
    raise AssertionError('expected ValueError on length 4')


def test_compute_cds_dna_hash_deterministic():
    h1 = compute_cds_dna_hash('ATGTAA')
    h2 = compute_cds_dna_hash('ATGTAA')
    assert h1 == h2 and len(h1) == 32, h1
    assert compute_cds_dna_hash('ATGTAG') != h1


def test_real_corpus_spot_check():
    """Sample 200 random rows from protein_final and verify CDS extraction
    + translation matches the stored `prot_seq` (terminal '*' tolerated).
    Spliced functions (M2, NEP, M42, NS3, PA-X) are excluded — Stage 1 has
    a known quirk where some spliced rows have a non-canonical
    annotation; that's tracked in the GTO reference, not a CDS-extractor
    issue.
    """
    proj = Path(__file__).resolve().parents[1]
    prot_csv = proj / 'data/processed/flu/July_2025/protein_final.csv'
    gen_csv = proj / 'data/processed/flu/July_2025/genome_final.csv'
    if not prot_csv.exists() or not gen_csv.exists():
        print('  spot-check skipped: Stage 1 outputs not present')
        return

    # selected_functions from conf/virus/flu.yaml — the 8 majors, all unspliced.
    unspliced_majors = (
        'RNA-dependent RNA polymerase PB2 subunit',
        'RNA-dependent RNA polymerase catalytic core PB1 subunit',
        'RNA-dependent RNA polymerase PA subunit',
        'Hemagglutinin precursor',
        'Nucleocapsid protein',
        'Neuraminidase protein',
        'Matrix protein 1',
        'Non-structural protein 1, interferon antagonist and host mRNA processing inhibitor',
    )
    # assembly_id is a 10-char hex string in the GTO corpus; force str
    # dtype so pandas does not coerce zero-padded numerics to int.
    prot = pd.read_csv(prot_csv,
                       usecols=['assembly_id', 'genbank_ctg_id', 'function',
                                'prot_seq', 'location', 'length'],
                       dtype={'assembly_id': str, 'genbank_ctg_id': str})
    prot = prot[prot['function'].isin(unspliced_majors)]
    sample = prot.sample(n=min(200, len(prot)), random_state=0)
    needed_ctgs = sample[['assembly_id', 'genbank_ctg_id']].drop_duplicates()

    gen = pd.read_csv(gen_csv,
                      usecols=['assembly_id', 'genbank_ctg_id', 'dna_seq'],
                      dtype={'assembly_id': str, 'genbank_ctg_id': str})
    gen = gen.merge(needed_ctgs, on=['assembly_id', 'genbank_ctg_id'], how='inner')
    ctg_lookup = {(a, c): d for a, c, d in
                  zip(gen['assembly_id'], gen['genbank_ctg_id'], gen['dna_seq'])}

    mismatches = 0
    checked = 0
    for _, row in sample.iterrows():
        key = (row['assembly_id'], row['genbank_ctg_id'])
        if key not in ctg_lookup:
            continue
        cds_dna = extract_cds_dna(ctg_lookup[key], row['location'])
        if len(cds_dna) % 3 != 0:
            mismatches += 1
            continue
        aa = translate_dna(cds_dna)
        target = row['prot_seq']
        # Stage 1 stores prot_seq with a trailing '*'; tolerate it.
        if aa == target:
            checked += 1
        elif aa.rstrip('*') == target.rstrip('*'):
            checked += 1
        else:
            mismatches += 1
            if mismatches <= 3:
                print(f'  mismatch on assembly_id={key[0]} '
                      f'function={row["function"][:30]}: '
                      f'aa={aa[:30]}... vs target={target[:30]}...')
    assert mismatches == 0, f'{mismatches} mismatches in {checked + mismatches} checks'
    print(f'  spot-check: {checked}/{checked + mismatches} translate-back rows match')


if __name__ == '__main__':
    tests = [
        test_parse_location_repr_string,
        test_parse_location_list,
        test_parse_location_rejects_empty,
        test_parse_location_rejects_wrong_arity,
        test_unspliced_extraction_matches_doc,
        test_spliced_extraction_skips_intron,
        test_strand_minus_reverse_complements,
        test_reverse_complement_handles_ambiguity,
        test_extract_rejects_off_end_span,
        test_extract_rejects_mixed_strands,
        test_translate_dna_round_trip,
        test_translate_dna_ambiguity_becomes_X,
        test_translate_dna_rejects_non_codon_length,
        test_compute_cds_dna_hash_deterministic,
        test_real_corpus_spot_check,
    ]
    failed = 0
    for t in tests:
        try:
            print(f'... {t.__name__}')
            t()
            print(f'    OK')
        except Exception as e:
            failed += 1
            print(f'    FAIL: {e}')
    if failed:
        print(f'\n{failed} test(s) failed')
        sys.exit(1)
    print(f'\nAll {len(tests)} tests passed.')
