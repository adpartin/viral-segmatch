"""
Utility functions for DNA/RNA sequence processing.

TODO
1. Finish and test these funcs, especially clean_dna_sequences
"""
import pandas as pd


def summarize_dna_qc(df: pd.DataFrame, seq_col: str = 'dna') -> pd.DataFrame:
    """Perform quality control on DNA sequences."""
    df = df.copy()
    ambig_codes = set("NRYWSKMBDHV")  # IUPAC ambiguous nucleotide codes
    
    def analyze_seq(seq):
        if not isinstance(seq, str) or len(seq) == 0:
            return (0, 0.0, 0, 0.0)
        seq_length = len(seq)
        ambig_count = sum(1 for base in seq if base in ambig_codes)
        ambig_frac = ambig_count / seq_length
        gc_content = (seq.count('G') + seq.count('C')) / seq_length
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