"""
Utility functions for DNA/RNA sequence processing.

DNA Ambiguity Codes (IUPAC)
---------------------------
DNA sequences from BV-BRC (and other databases) may contain IUPAC ambiguity
codes representing positions where the sequencer could not determine the exact
base. The standard (unambiguous) DNA alphabet is {A, C, G, T}. The IUPAC
ambiguity codes are:

    Two-base codes:
        R = A or G      (puRine)
        Y = C or T      (pYrimidine)
        S = G or C      (Strong — 3 hydrogen bonds)
        W = A or T      (Weak — 2 hydrogen bonds)
        K = G or T      (Keto)
        M = A or C      (aMino)

    Three-base codes:
        B = C, G, or T  (not A)
        D = A, G, or T  (not C)
        H = A, C, or T  (not G)
        V = A, C, or G  (not T)

    Fully ambiguous:
        N = A, C, G, or T (aNy base — most common in practice)

In Flu A genomes from BV-BRC, N is by far the most prevalent ambiguity code
(representing general sequencing uncertainty). The two- and three-base codes
are much rarer.

Current implementation
----------------------
summarize_dna_qc() records per-sequence:
  - ambig_count: total number of non-ACGT characters (all IUPAC codes pooled)
  - ambig_frac:  ambig_count / sequence length
  - length:      sequence length
  - gc_content:  (G + C) / sequence length

This is analogous to protein_utils.analyze_protein_ambiguities(), but less
detailed: proteins get per-residue-type breakdowns (X, B, Z, *, etc.) while
DNA currently lumps all ambiguity codes into a single count.

TODOs
-----
1. Add per-code breakdown (like analyze_protein_ambiguities does for proteins):
   count each IUPAC code separately (N, R, Y, etc.) so downstream consumers
   can distinguish "mostly Ns" from "many two-base ambiguities."
2. Finish and test clean_dna_sequences.
3. Consider impact on k-mer features: compute_kmer_features.py silently skips
   any k-mer window containing a non-ACGT character. A single ambiguous base
   at position i causes k windows (positions i-k+1 through i) to be dropped.
   For k=6, one N removes 6 k-mers from the count. With raw counts (normalize
   = 'none'), high-ambiguity sequences will have artificially lower totals.
   L1 normalization corrects for this (converts to frequencies).
"""
import pandas as pd


def summarize_dna_qc(df: pd.DataFrame, seq_col: str = 'dna') -> pd.DataFrame:
    """Perform quality control on DNA sequences."""
    df = df.copy()
    ambig_codes = set("NRYWSKMBDHV")  # IUPAC ambiguous nucleotide codes
    
    def analyze_seq(seq):
        if not isinstance(seq, str) or len(seq) == 0:
            return (0, 0.0, 0, 0.0)
        seq_upper = seq.upper()
        seq_length = len(seq_upper)
        ambig_count = sum(1 for base in seq_upper if base in ambig_codes)
        ambig_frac = ambig_count / seq_length
        gc_content = (seq_upper.count('G') + seq_upper.count('C')) / seq_length
        return (ambig_count, ambig_frac, seq_length, gc_content)
    
    qc = df[seq_col].apply(analyze_seq)
    columns = ['ambig_count', 'ambig_frac', 'length', 'gc_content']
    qc_df = pd.DataFrame(list(qc), columns=columns).reset_index(drop=True)
    return pd.concat([df, qc_df], axis=1)


def clean_dna_sequences(
    df: pd.DataFrame,
    seq_col: str = 'dna',
    max_ambig_frac: float = 0.1
    ) -> pd.DataFrame:
    """Clean DNA sequences by filtering out sequences with too many ambiguous bases."""
    df = df.copy()
    df = summarize_dna_qc(df, seq_col)
    print(f"Sequences with ambig_frac > {max_ambig_frac}: {sum(df['ambig_frac'] > max_ambig_frac)}")
    ambig_df = df[df['ambig_frac'] > max_ambig_frac]
    df = df[df['ambig_frac'] <= max_ambig_frac].reset_index(drop=True)
    return df, ambig_df
