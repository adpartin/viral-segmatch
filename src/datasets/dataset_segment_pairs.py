"""
Key columns:
- assembly_id: identifies isolates
- canonical_segment: maps each protein to a segments: L / M / S 
- prot_seq: protein sequence
- function: protein function
- brc_fea_id: unique feature ID (used to distinguish entries)

Splitg strategies:
- partition_by_isolate: ensures strict isolate-based splitting (no leakage)
- partition_by_duplicates: ensures duplicates of the same sequence stay in the same split


- Hard Partition by Isolates: all segments from an isolate (assembly_id) are assigned to a single train/val/test set to prevent leakage
- Hard Partition by Duplicates: identical protein sequences (prot_seq) across different isolates are assigned to the same set, preventing identical sequences from appearing across train/val/test sets
"""

import hashlib
import os
import random
import requests
from itertools import combinations
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

from io import BytesIO
from sklearn.model_selection import train_test_split

seed = 42  # for reproducibility (config!)

filepath = Path(__file__).resolve().parent # .py
# filepath = Path(os.path.abspath('')) # .ipynb
print(f'filepath: {filepath}')

# Settings
use_core_proteins_only = True

## Config
main_data_dir = filepath / '../../data'

# data_version = 'Feb_2025'
data_version = 'April_2025'
virus_name = 'bunya'
processed_data_dir = main_data_dir / 'processed' / virus_name / data_version

task_name = 'segmatch'
datasets_dir = main_data_dir / 'datasets' / virus_name / data_version / task_name

output_dir = datasets_dir
os.makedirs(output_dir, exist_ok=True)

print(f'\nmain_data_dir:      {main_data_dir}')
print(f'processed_data_dir: {processed_data_dir}')
print(f'datasets_dir:       {datasets_dir}')


def create_protein_pairs(
    df: pd.DataFrame,
    neg_to_pos_ratio: int = 1,
    allow_same_func_negatives: bool = True,
    max_same_func_ratio: float = 0.5,
    seed: int = 42,
    ) -> pd.DataFrame:
    """ Create positive and negative protein pairs for the binary classifier.

    Notes:
    1. Symmetric pairs handling (e.g., [seq_a, seq_b] and [seq_b, seq_a])
        - since we use combinations (and not permutations), we don't need to
            worry about this in positive pairs creation
        - for negative pairs, we need to ensure that we don't create duplicates
    2. Same-function pairs (e.g., RdRp from A vs. RdRp from B)
        - Do we want this in positive pairs? Not relevant if we inlcude only core proteins.
        - Do we want this in negative pairs? These pairs are important since same-function
              proteins across isolates are more similar (due to functional conservation)
              than cross-function proteins within an isolate. Excluding them could make the
              task too easy, as the model might rely on functional differences alone.
              However, you want to control their ratio and log their prevalence to ensure
              the dataset isn’t dominated by same-function negatives.

    Args:
        df (pd.DataFrame): df with columns ['assembly_id', 'brc_fea_id',
            'prot_seq', 'segment', 'function', 'seq_hash']
        neg_to_pos_ratio (int): Ratio of negative to positive pairs
        allow_same_func_negatives (bool): If False, exclude same-function
            negative pairs (e.g., RdRp paired with RdRp)
        max_same_func_ratio (float): maximum proportion of same-function
            negative pairs (if allow_same_func_negatives=True)

    Returns:
        pd.DataFrame with columns ['assembly_id_a', 'assembly_id_b', 'brc_a', 'brc_b',
            'seq_a', 'seq_b', 'seg_a', 'seg_b', 'func_a', 'func_b',
            'seq_hash_a', 'seq_hash_b', 'label']

    Notes:
        - Positive pairs are cross-function (e.g., RdRp-GPC, RdRp-N, GPC-N) within the same isolate.
        - Negative pairs are cross-isolate, with optional control over same-function pairs.
        - Symmetric pairs, [seq_a, seq_b] and [seq_b, seq_a], are prevented during negative pair generation.
    """
    
    random.seed(seed)
    isolates = df.groupby('assembly_id') # Group by isolate (assembly_id)

    # Create positive pairs (within-isolate, cross-function)
    pos_pairs = []
    isolates_with_few_proteins = []

    for assembly_id, grp in isolates:
        # print(grp[['file', 'assembly_id', 'canonical_segment', 'replicon_type', 'function', 'brc_fea_id']])

        # Ignore isolates with fewer than 2 proteins (these cannot be used to
        # create positive pairs)
        if len(grp) < 2:
            isolates_with_few_proteins.append(assembly_id)
            continue
        
        pairs = list(combinations(grp.itertuples(), 2)) # [(df_row, df_row), (df_row, df_row), ...]
        for row_a, row_b in pairs:
            # Add pair only if they have different BRC ids and different functions
            if row_a.brc_fea_id != row_b.brc_fea_id and row_a.function != row_b.function:
                dct = {
                    'assembly_id_a': assembly_id,
                    'assembly_id_b': assembly_id,
                    'brc_a': row_a.brc_fea_id,
                    'brc_b': row_b.brc_fea_id,
                    'seq_a': row_a.prot_seq,
                    'seq_b': row_b.prot_seq,
                    'seg_a': row_a.segment,
                    'seg_b': row_b.segment,
                    'func_a': row_a.function,
                    'func_b': row_b.function,
                    'seq_hash_a': row_a.seq_hash,
                    'seq_hash_b': row_b.seq_hash,
                    'label': 1  # Positive pair
                }
                pos_pairs.append(dct)

    pos_df = pd.DataFrame(pos_pairs)
    # print(pos_df[['brc_a', 'brc_b', 'seg_a', 'seg_b', 'func_a', 'func_b']])

    # Log isolates with fewer than 2 proteins
    # If an isolate has only one protein, it won’t generate any positive pairs.
    if isolates_with_few_proteins:
        print(f'Warning: {len(isolates_with_few_proteins)} isolates have fewer \
            than 2 proteins: {isolates_with_few_proteins}')

    def create_negative_pairs(df, num_negatives):
        """ Create negative pairs (cross-isolate) """
        neg_pairs = []
        seen_pairs = set() # Track unique pairs. TODO. why can't we define inside the function?

        # Precompute isolate groups to prevent repeated (trial-and-error) sampling.
        # This ensures 2 different isolates (assembly_id) are sampled during sampling.
        isolate_groups = {aid: list(grp.itertuples()) for aid, grp in df.groupby('assembly_id')}
        isolate_ids = list(isolate_groups.keys())

        same_func_count = 0
        max_same_func = int(num_negatives * max_same_func_ratio) if allow_same_func_negatives else 0

        while len(neg_pairs) < num_negatives:
            aid1, aid2 = random.sample(isolate_ids, 2)  # Sample 2 different isolates
            row_a = random.choice(isolate_groups[aid1]) # Sample a random protein from the 1st isolate
            row_b = random.choice(isolate_groups[aid2]) # Sample a random protein from the 2nd isolate

            # Check if pair is unique and not symmetric
            pair_key = tuple(sorted([row_a.brc_fea_id, row_b.brc_fea_id]))
            if pair_key in seen_pairs:
                continue

            # Check same-function constraint
            is_same_func = row_a.function == row_b.function
            if is_same_func and (not allow_same_func_negatives or same_func_count >= max_same_func):
                continue

            dct = {
                'assembly_id_a': row_a.assembly_id,
                'assembly_id_b': row_b.assembly_id,
                'brc_a': row_a.brc_fea_id,
                'brc_b': row_b.brc_fea_id,
                'seq_a': row_a.prot_seq,
                'seq_b': row_b.prot_seq,
                'seg_a': row_a.segment,
                'seg_b': row_b.segment,
                'func_a': row_a.function,
                'func_b': row_b.function,
                'seq_hash_a': row_a.seq_hash,
                'seq_hash_b': row_b.seq_hash,
                'label': 0  # Negative pair
            }
            neg_pairs.append(dct)
            seen_pairs.add(pair_key)
            if is_same_func:
                same_func_count += 1

        return pd.DataFrame(neg_pairs), same_func_count

    num_negatives = int(len(pos_pairs) * neg_to_pos_ratio)
    neg_df, same_func_count = create_negative_pairs(df, num_negatives)

    # Log dataset stats
    print(f'Positive pairs: {len(pos_df)}')
    print(f'Negative pairs: {len(neg_df)}')
    print(f'Same-function negative pairs: {same_func_count} ({same_func_count/len(neg_df)*100:.2f}%)')
    segment_pair_counts = neg_df.groupby(['seg_a', 'seg_b']).size().rename('count').reset_index()
    print(f'Negative pair segment count:\n{segment_pair_counts}')
    # TODO. Use segment_pair_counts to create histogram

    # Combine positive and negative pairs
    pairs_df = pd.concat([pos_df, neg_df], ignore_index=True)
    return pairs_df


def split_dataset(
    pairs_df: pd.DataFrame,
    df: pd.DataFrame,
    hard_partition_isolates: bool = True,
    hard_partition_duplicates: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Split dataset into train, val, and test sets with configurable partitioning.

    Args:
        pairs_df (pd.DataFrame): DataFrame containing protein pairs with columns
            ['assembly_id_a', 'assembly_id_b', 'brc_a', 'brc_b',
            'seq_a', 'seq_b', 'seg_a', 'seg_b', 'func_a', 'func_b',
            'seq_hash_a', 'seq_hash_b', 'label']
        df (pd.DataFrame): DataFrame containing protein sequences with columns
            ['assembly_id', 'brc_fea_id', 'prot_seq', 'segment', 'function', 'seq_hash']
        hard_partition_isolates (bool): If True, ensures strict isolate-based
            splitting (no leakage). If False, allows random splitting.
        hard_partition_duplicates (bool): If True, ensures identical protein
            sequences (prot_seq) across different isolates are assigned to the
            same set, preventing identical sequences from appearing across
            train/val/test sets. If False, allows random splitting.
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for val.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames for
            training, val, and test sets.
    """

    canonical_segments = ['L', 'M', 'S']
    # canonical_segments = core_funcs.keys()

    if hard_partition_duplicates:
        # Group isolates by identical protein sequences (for each segment)
        seq_groups = {}
        for segment in canonical_segments:
            segment_df = df[df['segment'] == segment]
            for seq_hash, group in segment_df.groupby('seq_hash'):
                isolates = set(group['assembly_id'])
                seq_groups[seq_hash] = isolates

        # Assign isolates to clusters based on shared sequences
        isolate_to_cluster = {}
        cluster_id = 0
        for seq_hash, isolates in seq_groups.items():
            for isolate in isolates:
                if isolate not in isolate_to_cluster:
                    isolate_to_cluster[isolate] = cluster_id
            cluster_id += 1

        # Create cluster-based splitting
        clusters = list(set(isolate_to_cluster.values()))
        train_clusters, temp_clusters = train_test_split(clusters, train_size=train_ratio, random_state=seed)
        val_clusters, test_clusters = train_test_split(temp_clusters, train_size=val_ratio/(1-train_ratio), random_state=seed)

        # Map isolates to sets
        train_isolates = {iso for iso, cid in isolate_to_cluster.items() if cid in train_clusters}
        val_isolates = {iso for iso, cid in isolate_to_cluster.items() if cid in val_clusters}
        test_isolates = {iso for iso, cid in isolate_to_cluster.items() if cid in test_clusters}

    else:
        # Split by isolates (or randomly if not hard_partition_isolates)
        unique_isolates = df['assembly_id'].unique()
        if hard_partition_isolates:
            train_isolates, tmp_isolates = train_test_split(unique_isolates, train_size=train_ratio, random_state=seed)
            val_isolates, test_isolates = train_test_split(tmp_isolates, train_size=val_ratio/(1-train_ratio), random_state=seed)
            train_isolates, val_isolates, test_isolates = set(train_isolates), set(val_isolates), set(test_isolates)
        else:
            # Random split without isolate-based partitioning
            train_pairs, temp_pairs = train_test_split(pairs_df, train_size=train_ratio, random_state=seed)
            val_pairs, test_pairs = train_test_split(temp_pairs, train_size=val_ratio/(1-train_ratio), random_state=seed)
            return train_pairs, val_pairs, test_pairs
    
    # Assign pairs to sets based on isolate membership
    train_pairs = pairs_df[
        (pairs_df['assembly_id_a'].isin(train_isolates)) & 
        (pairs_df['assembly_id_b'].isin(train_isolates))
    ]
    val_pairs = pairs_df[
        (pairs_df['assembly_id_a'].isin(val_isolates)) & 
        (pairs_df['assembly_id_b'].isin(val_isolates))
    ]
    test_pairs = pairs_df[
        (pairs_df['assembly_id_a'].isin(test_isolates)) & 
        (pairs_df['assembly_id_b'].isin(test_isolates))
    ]
    
    # Log stats
    breakpoint()
    total_pairs = len(train_pairs) + len(val_pairs) + len(test_pairs)
    total_isolates = len(train_isolates) + len(val_isolates) + len(test_isolates)
    print(f'Train pairs: {len(train_pairs)} ({len(train_pairs)/total_pairs*100:.2f}%)')
    print(f'Val pairs:   {len(val_pairs)} ({len(val_pairs)/total_pairs*100:.2f}%)')
    print(f'Test pairs:  {len(test_pairs)} ({len(test_pairs)/total_pairs*100:.2f}%)')
    print(f'Train isolates: {len(train_isolates)} ({len(train_isolates)/total_isolates*100:.2f}%)')
    print(f'Val isolates:   {len(val_isolates)} ({len(val_isolates)/total_isolates*100:.2f}%)')
    print(f'Test isolates:  {len(test_isolates)} ({len(test_isolates)/total_isolates*100:.2f}%)')
    return train_pairs, val_pairs, test_pairs


# Load protein data
print('\nLoad filtered protein data.')
fname = 'protein_filtered.csv'
datapath = processed_data_dir / fname
prot_df = pd.read_csv(datapath)

# Optionally restrict to core proteins
if use_core_proteins_only:

    core_funcs = {
        'L': 'RNA-dependent RNA polymerase',
        'M': 'Pre-glycoprotein polyprotein GP complex',
        'S': 'Nucleocapsid protein'
    }
    mask = prot_df.apply(lambda row: row['canonical_segment'] in core_funcs and row['function'] == core_funcs[row['canonical_segment']], axis=1)
    df_core = prot_df[mask].reset_index(drop=True)

    # core_functions = [
    #         'RNA-dependent RNA polymerase',
    #         'Pre-glycoprotein polyprotein GP complex',
    #         'Nucleocapsid protein'
    # ]
    # df2 = prot_df[ prot_df['function'].isin(core_functions) ].reset_index(drop=True)

    # print(df_core.equal(df2))


# Replace '*' with empty string in protein sequences
# The asterisk (*) is denotes a stop codon
df_core['prot_seq'] = df_core['prot_seq'].apply(lambda x: x.replace('*', ''))

# Standardize segment labels using canonical_segment (or infer from replicon_type)
# TODO. Note, this is the same as 'canonical_segment'
df_core['segment'] = df_core['canonical_segment'].fillna(
    df_core['replicon_type'].map({
        'Large RNA Segment': 'L',
        'Segment One': 'L',
        'Medium RNA Segment': 'M',
        'Segment Two': 'M',
        'Small RNA Segment': 'S',
        'Segment Three': 'S'
        # Add more mappings as needed based on dataset exploration
    })
)

# Add sequence hash for duplicate detection
# breakpoint()
df_core['seq_hash'] = df_core['prot_seq'].apply(lambda x: hashlib.md5(str(x).encode()).hexdigest())

# Check for sequence ambiguities (e.g., non-standard amino acids)
df_core['has_ambiguity'] = df_core['prot_seq'].str.contains('[^ACDEFGHIKLMNPQRSTVWY]', regex=True, na=False)
if df_core['has_ambiguity'].any():
    print(f"Warning: {df_core['has_ambiguity'].sum()} sequences contain non-standard amino acids.")

# Validate brc_fea_id uniqueness within isolates
dups = df_core[df_core.duplicated(subset=['assembly_id', 'brc_fea_id'], keep=False)]
if not dups.empty:
    raise ValueError(f"Duplicate brc_fea_id found within isolates: \
        {dups[['assembly_id', 'brc_fea_id']]}")
    
# Create pairs
print('\nCreate protein pairs.')
neg_to_pos_ratio = 3
allow_same_func_negatives = True
max_same_func_ratio = 0.5
pairs_df = create_protein_pairs(
    df_core,
    neg_to_pos_ratio=neg_to_pos_ratio,
    allow_same_func_negatives=allow_same_func_negatives,
    max_same_func_ratio=max_same_func_ratio,
    seed=seed
)

# Split dataset
print('\nSplit dataset into train, val, and test sets.')
# hard_partition_isolates = False
hard_partition_isolates = True
hard_partition_duplicates = False
# hard_partition_duplicates = True
train_pairs, val_pairs, test_pairs = split_dataset(
    pairs_df, df_core, hard_partition_isolates, hard_partition_duplicates
)

# Save datasets
print('\nSave datasets.')
train_pairs.to_csv(f"{output_dir}/train_pairs.csv", index=False)
val_pairs.to_csv(f"{output_dir}/val_pairs.csv", index=False)
test_pairs.to_csv(f"{output_dir}/test_pairs.csv", index=False)