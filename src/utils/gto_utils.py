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
    """Extract assembly prefix and ID from a file name.

    Handles two naming conventions:
    1. Bunyavirales: [ASSEMBLY_PREFIX]_[ASSEMBLY_ID].[#].qual.gto (e.g., GCA_031497195.1.qual.gto)
    2. Flu A: [ASSEMBLY_ID].gto (e.g., 1316165.15.gto)

    Args:
        file_name: The GTO file name

    Returns:
        tuple: (assembly_prefix, assembly_id) where assembly_prefix can be None for Flu A files
    """
    # Previous version (Bunyavirales only)
    # match = re.match(r'^(GCA|GCF)_(\d+\.\d+)', file_name)
    # return match.groups() if match else (None, None)

    # Pattern 1: Bunyavirales format - GCA/GCF_123456.1.qual.gto
    match1 = re.match(r'^(GCA|GCF)_(\d+\.\d+)', file_name)
    if match1:
        return match1.groups()

    # Pattern 2: Flu A format - alphanumeric assembly ID.gto (e.g., 123456.gto, 000142e874.gto)
    match2 = re.match(r'^([a-zA-Z0-9]+)\.gto$', file_name)
    if match2:
        return (None, match2.group(1))

    # No match found
    return (None, None)


def enforce_single_file(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Keep GCF_* over GCA_* for each assembly_id (Bunyavirales-specific).
    
    This function was designed for Bunyavirales data where the same assembly_id
    may appear in both GCA_* and GCF_* files. It preferentially keeps GCF_* versions.
    
    ⚠️  WARNING: This function assumes:
    - Files are named with GCA_/GCF_ prefixes
    - Only file-level duplicates exist (not intra-file duplicates)
    - After file selection, no duplicates remain on ['prot_seq', 'assembly_id']
    
    For Flu A data, this function now handles intra-file duplicates automatically.
    
    Args:
        df: DataFrame with protein data
        verbose: If True, print detailed information about file selection
        
    Returns:
        DataFrame with duplicates resolved
        
    Raises:
        ValueError: If duplicates remain after file selection
    """
    seq_col_name = 'prot_seq'

    # Method 1: File selection (GCA/GCF logic)
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

    prot_df = pd.concat(keep_rows, ignore_index=True)

    # Method 2
    """
    dup_cols = [seq_col_name, 'assembly_id', 'function', 'replicon_type']
    groups = prot_df.groupby(dup_cols)['assembly_prefix'].apply(set).reset_index(name='prefixes') # Find all groups (on dup_cols) where both GCA and GCF appear
    dups_to_fix = groups[groups['prefixes'].apply(lambda s: {'GCA', 'GCF'}.issubset(s))][dup_cols] # Select only those groups that contain both 'GCA' and 'GCF'
    to_drop_keys = set(tuple(x) for x in dups_to_fix.values) # Turn that (dup_cols combos) into a set of keys for fast lookup
    
    # Create mask to drop dups
    mask_with_gca_dups = prot_df['assembly_prefix'].eq('GCA') 
    mask_with_drop_keys = prot_df.set_index(dup_cols).index.isin(to_drop_keys)
    mask_drop = mask_with_gca_dups & mask_with_drop_keys
    prot_df = prot_df[~mask_drop].copy()
    print(f"Dropped {mask_drop.sum()} GCA rows. Remaining rows: {len(prot_df)}")
    """

    # Check for remaining duplicates
    dup_cols = [seq_col_name, 'assembly_id']
    remaining_dups = (
        prot_df[prot_df.duplicated(subset=dup_cols, keep=False)]
        .sort_values(dup_cols)
        .reset_index(drop=True).copy()
    )
    
    if not remaining_dups.empty:
        # For Flu A data, this is expected and we need to handle it differently
        print(f"⚠️  Found {len(remaining_dups)} remaining duplicates after (GCF/GCA) file selection")
        print(f"   This may indicate intra-file duplicates (same sequence in same file)")
        print(f"   Attempting to resolve by keeping first occurrence...")
        
        # Keep first occurrence of each duplicate
        prot_df = prot_df.drop_duplicates(subset=dup_cols, keep='first')
        print(f"   Resolved duplicates. Final shape: {prot_df.shape}")
        
        # Final check
        final_dups = prot_df[prot_df.duplicated(subset=dup_cols, keep=False)]
        if not final_dups.empty:
            raise ValueError(f"Could not remove all duplicates. {len(final_dups)} duplicates remain.")
    
    return prot_df


def handle_assembly_duplicates(
    df: pd.DataFrame, 
    seq_col_name: str = 'prot_seq',
    strategy: str = 'keep_first'
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handle duplicate protein sequences within the same assembly_id.

    Three types of duplicates:
    1. Same seq, same assembly_id, different files → Keep one file (GCA/GCF logic)
    2. Same seq, same assembly_id, same file → Keep one record (intra-file duplicates)
    3. Different seq, same assembly_id → Keep all (not duplicates)

    Args:
        df: DataFrame with protein data
        seq_col_name: Name of the sequence column
        strategy: Strategy for handling duplicates ('keep_first', 'keep_last')

    Returns:
        tuple: (cleaned_df, removed_duplicates_df)
    """
    print(f"🔍 Analyzing duplicates on ['{seq_col_name}', 'assembly_id']")
    
    # Find all duplicates
    dup_cols = [seq_col_name, 'assembly_id']
    all_dups = (
        df[df.duplicated(subset=dup_cols, keep=False)]
        .sort_values(dup_cols + ['file', 'brc_fea_id'])
        .copy()  # Keep original index for proper referencing
    )

    if all_dups.empty:
        print("✅ No duplicates found")
        return df, pd.DataFrame()

    print(f"📊 Found {len(all_dups)} duplicate records across "
          f"{all_dups['assembly_id'].nunique()} assemblies")

    # Analyze duplicate types and build cleaned dataframe in one pass
    dup_summary = []
    keep_indices = set()

    for (seq, aid), grp in all_dups.groupby(dup_cols):
        files = grp['file'].unique()
        dup_type = 'different_files' if len(files) > 1 else 'same_file'

        # Determine which record to keep
        if dup_type == 'different_files':
            # GCA/GCF logic: prefer GCF_ files
            gcf_files = [f for f in files if f.startswith('GCF_')] # TODO: there might be a case with a different prefix! Need to handle this.
            chosen_file = gcf_files[0] if gcf_files else files[0]
            keep_idx = grp[grp['file'] == chosen_file].index[0]  # Get the index of the chosen file
        else:
            # Same file: use strategy
            if strategy == 'keep_first':
                keep_idx = grp.index[0]
            else:  # keep_last
                keep_idx = grp.index[-1]

        # Record the index to keep
        keep_indices.add(keep_idx)

        # Record what was kept/removed for summary
        for idx, row in grp.iterrows():
            print(f"file: {row['file']}; assembly_id: {row['assembly_id']}; brc_fea_id: {row['brc_fea_id']}")
            action = 'kept' if idx == keep_idx else 'removed'
            reason = 'gcf_preferred' if (dup_type == 'different_files' and action == 'kept') else f'{strategy}_occurrence'

            dup_summary.append({
                'assembly_id': row['assembly_id'],
                'file': row['file'],
                'function': row['function'],
                'brc_fea_id': row['brc_fea_id'],
                'prot_seq_preview': row[seq_col_name][:30] + "..." if len(row[seq_col_name]) > 30 else row[seq_col_name],
                'duplicate_type': dup_type,
                'action_taken': action,
                'reason': reason
            })

    # Create cleaned dataframe
    dup_summary_df = pd.DataFrame(dup_summary)

    # Create cleaned dataframe: keep all non-duplicate rows + selected duplicate rows
    non_dup_mask = ~df.index.isin(all_dups.index)
    selected_dup_mask = df.index.isin(keep_indices)
    cleaned_df = df[non_dup_mask | selected_dup_mask].copy()

    print(f"📈 Duplicate resolution summary:")
    print(f"   - Different files: {len(dup_summary_df[dup_summary_df['duplicate_type'] == 'different_files'])}")
    print(f"   - Same file: {len(dup_summary_df[dup_summary_df['duplicate_type'] == 'same_file'])}")
    print(f"   - Kept: {len(dup_summary_df[dup_summary_df['action_taken'] == 'kept'])}")
    print(f"   - Removed: {len(dup_summary_df[dup_summary_df['action_taken'] == 'removed'])}")

    return cleaned_df, dup_summary_df


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
            case = 'Case 1'  # Same assembly_id, different records
        elif len(set(assembly_ids)) > 1 and len(set(record_ids)) > 1:
            case = 'Case 2'  # Different assembly_id, different records
        else:
            case = 'Other'   # Catch rare/unclassified cases        

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