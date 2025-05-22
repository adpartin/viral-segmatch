"""
Utility functions for processing GTO files.

TODO
1. Not sure if still need classify_dup_groups()
2. Log filenames or df with the rows that were excluded in enforce_single_file()
"""
import re
from enum import Enum
from typing import Union

import pandas as pd


class SequenceType(Enum):
    DNA = 'dna'
    PROTEIN = 'prot_seq'


def extract_assembly_info(file_name: str) -> tuple[str, str]:
    """Extract assembly prefix and ID from a file name."""
    match = re.match(r'^(GCA|GCF)_(\d+\.\d+)', file_name)
    return match.groups() if match else (None, None)


def enforce_single_file(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Keep GCF_* over GCA_* for each assembly_id.
    TODO log those to file
    """
    keep_rows = []
    for aid, grp in df.groupby('assembly_id'):
        files = grp['file'].unique()
        if len(files) == 1:
            keep_rows.append(grp)
            continue
        gcf_files = [f for f in files if f.startswith('GCF_')]
        chosen_fname = gcf_files[0] if gcf_files else files[0] # If multiple files, prioritize GCF_
        chosen_file_df = grp[grp['file'] == chosen_fname]
        if verbose:
            print(f"assembly_id {aid}: Multiple files {files}, kept {chosen_fname}")
        keep_rows.append(chosen_file_df)
    return pd.concat(keep_rows, ignore_index=True)


def classify_dup_groups(
    df: pd.DataFrame,
    seq_col_name: Union[SequenceType, str]
    ) -> pd.DataFrame:
    """Classify duplicate sequences based on assembly IDs, record IDs, and prefixes.
    TODO Not sure if still need this
    Raises:
        ValueError: If seq_col_name is not 'dna' or 'prot_seq'
    """
    # Convert string to enum if needed
    if isinstance(seq_col_name, str):
        try:
            seq_col_name = SequenceType(seq_col_name)
        except ValueError:
            raise ValueError("seq_col_name must be either 'dna' or 'prot_seq'")
    
    seq_col = seq_col_name.value
    # Determine the record ID column name (source-specific identifier)
    record_col = {
        SequenceType.DNA: 'genbank_ctg_id',
        SequenceType.PROTEIN: 'brc_fea_id'
    }[seq_col_name]
    
    dup_info = []
    for seq, grp in df.groupby(seq_col): # retrieve rows with the same seq (seq duplicates)
        files = list(grp['file'])                # e.g., GCA_031497195.1.qual.gto
        assembly_ids = list(grp['assembly_id'])  # e.g., 031497195.1
        record_ids = list(grp[record_col])       # genbank_ctg_id or brc_fea_id
        prefixes = list(grp['assembly_prefix'])  # GCA or GCF

        if len(set(assembly_ids)) == 1 and len(set(record_ids)) > 1:
            case = 'Case 1'  # Same assembly, different records
        elif len(set(assembly_ids)) > 1 and len(set(record_ids)) > 1:
            case = 'Case 2'  # Different assemblies, different records
        else:
            case = 'Other'   # Catch-all for rare/unclassified cases        

        dup_info.append({
            seq_col: seq,
            'num_dups': len(grp),
            'files': files,
            'assembly_ids': list(set(assembly_ids)),
            'record_ids': list(set(record_ids)),
            'prefixes': list(set(prefixes)),
            'case': case
        })
    
    return pd.DataFrame(dup_info).sort_values('case').reset_index(drop=True)