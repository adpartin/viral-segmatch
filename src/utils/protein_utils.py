"""
Example run:
python src/utils/protein_utils.py

TODO Use cases:
1. We can develop a pipeline that loads protein data (e.g., GTO files), cleans
    the seqs, copies the cleaned seqs back to files, saves the files with new
    file names, and saves a summary report.
"""

from collections import Counter
from pprint import pprint
import pandas as pd

# Global ambiguity meanings (from Claude’s code, reused for consistency)
AMBIGUITY_MEANINGS = {
    'B': 'Asn or Asp', # A common ambiguity code representing either Asparagine (N) or Aspartic acid (D).
    'Z': 'Gln or Glu', # A common ambiguity code representing either Glutamine (Q) or Glutamic acid (E).
    'J': 'Leu or Ile', # A newer code intended to represent an ambiguity between Leucine (L) and Isoleucine (I). These two amino acids are isomers and can be difficult to distinguish in some mass spectrometry analyses.
    'X': 'Any amino acid', # Most general ambiguity code, meaning the identity of the amino acid at that position is unknown or unspecified.
    'U': 'Selenocysteine', # Selenocysteine is the 21st amino acid.
    'O': 'Pyrrolysine',    # Pyrrolysine is the 22nd amino acid.
    '*': 'Stop codon', # Used for terminal stops (internal stops get custom meaning)
    '-': 'Gap', # In sequence alignments, the hyphen often denotes a gap, meaning a deletion in one sequence relative to another, or a missing segment.
    '.': 'Gap'  # Also can be used to denote a gap, similar to a hyphen, though the hyphen is more common.
}


def analyze_protein_ambiguities(
    prot_df: pd.DataFrame,
    seq_column: str='prot_seq'
    ) -> pd.DataFrame:
    """
    Analyze protein sequences for ambiguous amino acids and provide detailed information.
    
    Args:
        prot_df (pd.DataFrame): DataFrame with protein sequences.
        seq_column (str): Name of the column containing protein sequences.
    
    Returns:
        DataFrame with additional columns:
        - 'has_ambiguities': Bool indicating non-standard amino acids or internal stops
        - 'ambiguous_residues': List of unique non-standard amino acids (excluding *)
        - 'ambiguity_positions': Dict mapping each ambiguous residue to its positions
        - 'ambiguity_count': Total number of ambiguous residues (excluding *)
        - 'has_terminal_stop': Bool indicating if sequence ends with '*'
        - 'has_internal_stop': Bool indicating if sequence has internal '*'
        - 'internal_stop_positions': List of positions of internal stop codons
    """
    prot_df = prot_df.copy()
    standard_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')  # 20 canonical amino acids

    def identify_ambiguities(seq: str):
        """ Identify ambiguous residues in a protein sequence and their positions. """
        if not isinstance(seq, str):
            return False, [], {}, 0, False, False, []

        seq = seq.upper().strip()
        positions = {}  # Positions of ambiguous residues
        internal_stop_positions = []  # Premature Stop Codon
        has_terminal_stop = seq.endswith('*')  # Terminal Stop Codon

        for i, aa in enumerate(seq):
            if aa not in standard_amino_acids:
                if aa == '*' and i == len(seq) - 1:
                    continue  # Don't count terminal stop as ambiguity
                if aa == '*':
                    internal_stop_positions.append(i + 1)  # 1-based indexing
                    continue  # Skip internal * for ambiguous_residues
                if aa not in positions:
                    positions[aa] = [i + 1]  # 1-based indexing for biologists
                else:
                    positions[aa].append(i + 1)

        ambiguous_residues = list(positions.keys())
        ambiguity_count = sum(len(pos_list) for pos_list in positions.values()) # Total ambiguities (excluding terminal stop)
        has_internal_stop = len(internal_stop_positions) > 0

        return (
            bool(ambiguous_residues or internal_stop_positions),
            ambiguous_residues,
            positions,
            ambiguity_count,
            has_terminal_stop,
            has_internal_stop,
            internal_stop_positions
        )
    
    # Apply the function to each sequence
    results = prot_df[seq_column].apply(identify_ambiguities)
    
    # Extract the results
    prot_df['has_ambiguities'] = results.apply(lambda x: x[0])
    prot_df['ambiguous_residues'] = results.apply(lambda x: x[1])
    prot_df['ambiguity_positions'] = results.apply(lambda x: x[2])
    prot_df['ambiguity_count'] = results.apply(lambda x: x[3])
    prot_df['has_terminal_stop'] = results.apply(lambda x: x[4])
    prot_df['has_internal_stop'] = results.apply(lambda x: x[5])
    prot_df['internal_stop_positions'] = results.apply(lambda x: x[6])

    return prot_df


def summarize_ambiguities(prot_df: pd.DataFrame) -> dict:
    """ Generate a summary of ambiguous residues across all sequences. """
    if 'ambiguous_residues' not in prot_df.columns:
        raise ValueError("DataFrame must be processed by analyze_protein_ambiguities() first")

    prot_df = prot_df.copy()
    total_seqs = len(prot_df)
    assert total_seqs > 0, 'Input DataFrame has zero rows.'
    
    seqs_with_ambiguities = prot_df['has_ambiguities'].sum()
    percent_seqs_with_ambiguities = round((seqs_with_ambiguities / total_seqs) * 100, 4)
    total_ambiguities = prot_df['ambiguity_count'].sum()
    terminal_stops = prot_df['has_terminal_stop'].sum()
    internal_stops = prot_df['has_internal_stop'].sum()
    percent_terminal_stops = round((terminal_stops / total_seqs) * 100, 4)
    percent_internal_stops = round((internal_stops / total_seqs) * 100, 4)
    
    # Count occurrence of each ambiguous residue
    all_ambiguities = []
    for residue_list in prot_df['ambiguous_residues']:
        all_ambiguities.extend(residue_list)
    ambiguity_counts = Counter(all_ambiguities)
    
    # Add meanings to ambiguity counts
    non_standard_residue_summary = {
        k: {'count': v, 'meaning': AMBIGUITY_MEANINGS.get(k, 'Unknown') if k != '*' else 'Premature stop codon or error'} 
        for k, v in ambiguity_counts.items()
    }

    return {
        'total_seqs': total_seqs,
        'seqs_with_ambiguities': int(seqs_with_ambiguities),
        'percent_seqs_with_ambiguities': percent_seqs_with_ambiguities,
        'total_ambiguities': int(total_ambiguities),
        'terminal_stops': int(terminal_stops),
        'percent_terminal_stops': percent_terminal_stops,
        'internal_stops': int(internal_stops),
        'percent_internal_stops': percent_internal_stops,
        'non_standard_residue_summary': non_standard_residue_summary
    }


def prepare_sequences_for_esm2(
    prot_df: pd.DataFrame,
    seq_column: str='prot_seq',
    output_column: str='esm2_ready_seq',
    max_internal_stops: float=0.1,
    max_x_residues: float=0.1,
    x_imputation: str='G',
    strip_terminal_stop=True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare protein sequences for ESM-2 by handling (internal and terminal)
    stop codons and ambiguities.

    Strategies to handle for X imputation:
        'G': Replaces X with G
        'remove': Returns None if X is present
        'MLM': Uses ESM-2 to predict residues

    Returns None if non-standard amino acids are found (except for X or *).

    Parameters:
    -----------
    prot_df : pd.DataFrame
        DataFrame with protein sequences, preferably processed by analyze_protein_ambiguities.
    seq_column : str, default='prot_seq'
        Input sequence column.
    output_column : str, default='esm2_ready_seq'
        Output column for ESM-2-ready sequences.
    max_internal_stops : flaot, 
        Maximum allowed internal stop codons in percent of seq length.
    x_imputation : str, default='G'
        Strategy for handling X ('G', 'remove', 'MLM').
    strip_terminal_stop : bool, default=True
        If True, removes terminal *; if False, preserves it.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with ESM-2-ready sequences and additional columns.
    """
    prot_df = prot_df.copy()
    prob_seqs = []

    def process_sequence(seq):
        # breakpoint()
        if not isinstance(seq, str) or len(seq.strip()) == 0:
            return None

        seq = seq.upper().strip()
        eff_length = len(seq) - (1 if seq.endswith('*') else 0) # Ignores terminal stop

        # Exclude sequence with too many internal stops (max_internal_stops threshold)
        internal_stops = (seq[:-1] if seq.endswith('*') else seq).count('*')
        internal_stops_ratio = round(internal_stops / eff_length, 4)
        if internal_stops_ratio > max_internal_stops:
            prob_seqs.append({'prot_seq': seq,
                'problem': f'Exceeds max_internal_stops threshold of {max_internal_stops}'
            })
            print(f"Sequence '{seq}' excluded due to too many internal stops: "
                  f"{internal_stops_ratio} stops (maximum allowed: {max_internal_stops})")
            return None

        # Replace internal stops with X (if the seq passed the max_internal_stops filter)
        if seq.endswith('*'):
            interm_seq = seq[:-1].replace('*', 'X') + ('*' if not strip_terminal_stop else '')
        else:
            interm_seq = seq.replace('*', 'X')

        # Exclude sequence with too many X residues (max_x_residues threshold)
        x_count = interm_seq.count('X')
        x_count_ratio = round(x_count / eff_length, 4)
        if x_count_ratio > max_x_residues:
            prob_seqs.append({'prot_seq': seq,
                'problem': f'Exceeds max_x_residues threshold of {max_x_residues}'
            })
            print(f"Sequence '{seq}' excluded due to too many X residues: "
                  f"{x_count_ratio} (maximum allowed: {max_x_residues})")
            return None

        processed_seq = interm_seq

        # Handle Xs
        if x_imputation == 'G':
            processed_seq = processed_seq.replace('X', 'G')

        elif x_imputation == 'remove' and 'X' in processed_seq:
            prob_seqs.append({'prot_seq': seq, 'problem': 'Contains X residues'})
            print(f"Sequence '{seq}' excluded due to X residues")
            return None  # Exclude sequence with any X

        elif x_imputation == 'MLM' and 'X' in processed_seq:
            # Impute Xs with MLM (using ESM-2)
            import esm
            import torch
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            model.eval()
            with torch.no_grad():
                data = [(('sequence', processed_seq.replace('X', alphabet.mask_token)))]
                batch_converter = alphabet.get_batch_converter()
                _, _, batch_tokens = batch_converter(data)
                results = model(batch_tokens, repr_layers=[33])
                logits = results['logits'][0]
                predicted_seq = list(processed_seq)
                for i, c in enumerate(processed_seq):
                    if c == 'X':
                        probs = torch.softmax(logits[i], dim=-1)
                        predicted_seq[i] = alphabet.get_tok(probs.argmax().item())
                processed_seq = ''.join(predicted_seq)

        # Exclude sequence with non-standard amino acids (and terminal * if strip_terminal_stop=False)
        allowed_chars = 'ACDEFGHIKLMNPQRSTVWY' + ('*' if not strip_terminal_stop else '')
        if any(c not in allowed_chars for c in processed_seq):
            non_standard = set(c for c in processed_seq if c not in allowed_chars)
            prob_seqs.append({'prot_seq': seq, 'problem': f'Non-standard characters: {non_standard}'})
            print(f"Sequence '{seq}' excluded due to non-standard characters "
                  f"(that are not currently handled): {non_standard}")
            return None # Exclude sequence with non-standard characters

        return processed_seq

    
    def get_intermediate_seq(seq: str):
        """ Compute intermediate sequence for x_count_ratio. """
        if not isinstance(seq, str) or len(seq.strip()) == 0:
            return None
        seq = seq.upper().strip()
        if seq.endswith('*'):
            return seq[:-1].replace('*', 'X') + ('*' if not strip_terminal_stop else '')
        return seq.replace('*', 'X')

    prot_df[output_column] = prot_df[seq_column].apply(process_sequence)
    prot_df['x_count_ratio'] = prot_df[seq_column].apply(
        lambda seq: (
            0.0
            if not isinstance(seq, str) or len(seq.strip()) == 0 or
            not (interm_seq := get_intermediate_seq(seq))
            else round(interm_seq.count('X') / (len(seq) - (1 if seq.endswith('*') else 0)), 4)
        )
    )

    # DataFrame with problematic sequences
    problematic_seqs_df = pd.DataFrame(prob_seqs)
    if not problematic_seqs_df.empty:
        cols = ['file', 'brc_fea_id']
        if all(True if i in prot_df.columns else False for i in cols):
            problematic_seqs_df = prot_df.merge(problematic_seqs_df, on=cols)
            # TODO finish this
        else:
            problematic_seqs_df = prot_df.merge(problematic_seqs_df, on='prot_seq', how='inner')
        # problematic_seqs_df.to_csv('problematic_protein_seqs.csv', index=False)

    return prot_df, problematic_seqs_df


def print_replicon_func_count(
    df: pd.DataFrame,
    functions: list[str] = None,
    more_cols: list[str] = None,
    drop_na: bool = True
    ) -> pd.DataFrame:
    """Print counts of [replicon_type, function] combos.
    Args:
        df (pd.DataFrame): DataFrame containing protein data
        functions (list[str]): List of functions to filter by
        more_cols (list[str]): Additional columns to include in the grouping
        drop_na (bool): Whether to drop NA values from the grouping
    
    This func is a reusable diagnostic tool for summarizing protein function
    counts, not specific to preprocessing.
    """
    df = df.copy()
    if functions:
        df = df[df['function'].isin(functions)]
    basic_cols = ['replicon_type', 'function']
    col_list = basic_cols if not more_cols else basic_cols + more_cols
    res = (
        df.groupby(col_list, dropna=drop_na)
        .size()
        .reset_index(name='count')
        .sort_values(['function', 'count'], ascending=False)
        .reset_index(drop=True)
    )
    print(res)
    return res


# Example usage
if __name__ == "__main__":

    # Example data
    data = {
        'prot_seq': [
            'ACDEF*XB*GHIK*X*LMNPQRSTVWY*',  # Various (B, X, internal *, terminal *) - dropped
            'ACDEF*BXZ*GHILMNPQRSTVWY*',     # Various (B, X, Z, internal *, terminal *) - dropped
            'ACDEFGHIKLMNPQRSTVWY*',   # Terminal stop - kept
            'ACDEF*GHIKLMNPQRSTVWY',   # Internal stop - kept
            'AC*****************WY*',  # Various (internal *, terminal *) - dropped
            'ACXXXXXXXXXXXXXXXXXWY',   # Contains X - dropped
            'ACDEFXGHIKLMNPQRSTVWY*',  # Various (X, terminal *) - kept
            'ACDEFBGHIKLMNPQRSTVWY',   # Contains B - dropped
            'ACDEFGHIKLMNPQRSTVWY',    # Clean - kept
        ]
    }
    prot_df = pd.DataFrame(data)
    output_column = 'esm2_ready_seq'
    max_internal_stops = 0.25
    max_x_residues = 0.25

    print(f'{prot_df}\n')

    print('\nCalculate dataset statistics.')
    prot_df = analyze_protein_ambiguities(prot_df)
    summary = summarize_ambiguities(prot_df)
    pprint(summary)
    print(f"Total seqs with terminal stops: {summary['terminal_stops']} ({summary['percent_terminal_stops']:.2f}%)")
    print(f"Total seqs with internal stops: {summary['internal_stops']} ({summary['percent_internal_stops']:.2f}%)")
    print("Non-standard residue distribution (excluding terminal stops):")
    for aa, details in summary['non_standard_residue_summary'].items():
        meaning = details['meaning'] if aa != '*' else 'Premature stop or error'
        print(f"  {aa}: occurred in {details['count']} seqs ({meaning})")

    # Prepare for ESM-2
    # breakpoint()
    print("\nPrepare data for ESM-2 (with x_imputation='G')")
    prot_df_g, problematic_seqs_df = prepare_sequences_for_esm2(
        prot_df.copy(),
        x_imputation='G',
        max_internal_stops=max_internal_stops,
        max_x_residues=max_x_residues
    )
    print("\nAfter preparing protein data with x_imputation='G':")
    print(prot_df_g)

    # Prepare for ESM-2
    # breakpoint()
    print("\nPrepare data for ESM-2 (with x_imputation='remove')")
    prot_df_remove, problematic_seqs_df = prepare_sequences_for_esm2(
        prot_df.copy(),
        x_imputation='remove',
        max_internal_stops=max_internal_stops,
        max_x_residues=max_x_residues
    )
    print("\nAfter preparing with protein data x_imputation='remove':")
    print(prot_df_remove)

    # breakpoint()
    print('\nDone.')