"""
Key columns:
- assembly_id: identifies isolates
- canonical_segment: maps each protein to segments: L / M / S 
- prot_seq: protein sequence
- function: protein function
- brc_fea_id: unique feature ID

Split strategies:
- hard_partition_isolates: all proteins from an isolate (assembly_id) are assigned
    to a single train/val/test set to prevent leakage
- hard_partition_duplicates: identical protein sequences (prot_seq) across different
    isolates are assigned to the same set
"""

import hashlib
import os
import random
from itertools import combinations
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Config
SEED = 42
TASK_NAME = 'segmatch'

filepath = Path(__file__).resolve().parent
main_data_dir = filepath / '../../data'
data_version = 'April_2025'
virus_name = 'bunya'
processed_data_dir = main_data_dir / 'processed' / virus_name / data_version
datasets_dir = main_data_dir / 'datasets' / virus_name / data_version / TASK_NAME

output_dir = datasets_dir
os.makedirs(output_dir, exist_ok=True)

print(f'\nmain_data_dir:      {main_data_dir}')
print(f'processed_data_dir: {processed_data_dir}')
print(f'datasets_dir:       {datasets_dir}')

# Settings
use_core_proteins_only = True

def create_protein_pairs(
    df: pd.DataFrame,
    neg_to_pos_ratio: int = 1,
    allow_same_func_negatives: bool = True,
    max_same_func_ratio: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Create positive and negative protein pairs for the binary classifier."""
    np.random.seed(seed)
    random.seed(seed)
    isolates = df.groupby('assembly_id')

    # Create positive pairs (within-isolate, cross-function)
    pos_pairs = []
    isolates_with_few_proteins = []
    for assembly_id, grp in isolates:
        if len(grp) < 2:
            isolates_with_few_proteins.append(assembly_id)
            continue
        pairs = list(combinations(grp.itertuples(), 2))
        for row_a, row_b in pairs:
            if row_a.brc_fea_id != row_b.brc_fea_id and row_a.function != row_b.function:
                dct = {
                    'assembly_id_a': assembly_id, 'assembly_id_b': assembly_id,
                    'brc_a': row_a.brc_fea_id, 'brc_b': row_b.brc_fea_id,
                    'seq_a': row_a.prot_seq, 'seq_b': row_b.prot_seq,
                    'seg_a': row_a.segment, 'seg_b': row_b.segment,
                    'func_a': row_a.function, 'func_b': row_b.function,
                    'seq_hash_a': row_a.seq_hash, 'seq_hash_b': row_b.seq_hash,
                    'label': 1
                }
                pos_pairs.append(dct)
    pos_df = pd.DataFrame(pos_pairs)
    if isolates_with_few_proteins:
        print(f'Warning: {len(isolates_with_few_proteins)} isolates have <2 proteins: {isolates_with_few_proteins}')

    def create_negative_pairs(df: pd.DataFrame, num_negatives: int) -> Tuple[pd.DataFrame, int]:
        """Create negative pairs (cross-isolate)."""
        neg_pairs = []
        seen_pairs = set()
        isolate_groups = {aid: list(grp.itertuples()) for aid, grp in df.groupby('assembly_id')}
        isolate_ids = list(isolate_groups.keys())
        same_func_count = 0
        max_same_func = int(num_negatives * max_same_func_ratio) if allow_same_func_negatives else 0

        while len(neg_pairs) < num_negatives:
            aid1, aid2 = random.sample(isolate_ids, 2)
            row_a = random.choice(isolate_groups[aid1])
            row_b = random.choice(isolate_groups[aid2])
            pair_key = tuple(sorted([row_a.brc_fea_id, row_b.brc_fea_id]))
            if pair_key in seen_pairs:
                continue
            is_same_func = row_a.function == row_b.function
            if is_same_func and (not allow_same_func_negatives or same_func_count >= max_same_func):
                continue
            dct = {
                'assembly_id_a': row_a.assembly_id, 'assembly_id_b': row_b.assembly_id,
                'brc_a': row_a.brc_fea_id, 'brc_b': row_b.brc_fea_id,
                'seq_a': row_a.prot_seq, 'seq_b': row_b.prot_seq,
                'seg_a': row_a.segment, 'seg_b': row_b.segment,
                'func_a': row_a.function, 'func_b': row_b.function,
                'seq_hash_a': row_a.seq_hash, 'seq_hash_b': row_b.seq_hash,
                'label': 0
            }
            neg_pairs.append(dct)
            seen_pairs.add(pair_key)
            if is_same_func:
                same_func_count += 1

        return pd.DataFrame(neg_pairs), same_func_count

    num_negatives = int(len(pos_pairs) * neg_to_pos_ratio)
    neg_df, same_func_count = create_negative_pairs(df, num_negatives)
    print(f'Positive pairs: {len(pos_df)}')
    print(f'Negative pairs: {len(neg_df)}')
    print(f'Same-function negative pairs: {same_func_count} ({same_func_count/len(neg_df)*100:.2f}%)')
    segment_pair_counts = neg_df.groupby(['seg_a', 'seg_b']).size().rename('count').reset_index()
    print(f'Negative pair segment count:\n{segment_pair_counts}')
    pairs_df = pd.concat([pos_df, neg_df], ignore_index=True)
    return pairs_df

def compute_isolate_pair_counts(df: pd.DataFrame) -> Dict:
    """Compute positive pair counts per isolate."""
    isolate_pos_counts = {}
    for aid, grp in df.groupby('assembly_id'):
        n_proteins = len(grp)
        if n_proteins < 2:
            isolate_pos_counts[aid] = 0
        else:
            pairs = list(combinations(grp.itertuples(), 2))
            pos_count = sum(1 for row_a, row_b in pairs if row_a.function != row_b.function)
            isolate_pos_counts[aid] = pos_count
    return isolate_pos_counts

def split_dataset(
    pairs_df: pd.DataFrame,
    df: pd.DataFrame,
    hard_partition_isolates: bool = True,
    hard_partition_duplicates: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, val, and test sets with stratified sampling."""
    np.random.seed(seed)
    random.seed(seed)

    # Compute neg_to_pos_ratio
    jj = pairs_df['label'].value_counts().reset_index()
    neg_count = jj.loc[jj['label'] == 0, 'count'].iloc[0] if 0 in jj['label'].values else 0
    pos_count = jj.loc[jj['label'] == 1, 'count'].iloc[0] if 1 in jj['label'].values else 1
    neg_to_pos_ratio = neg_count / pos_count if pos_count > 0 else 3.0
    target_pos_proportion = 1 / (1 + neg_to_pos_ratio)
    print(f'Negative to positive ratio: {neg_to_pos_ratio:.2f}')
    print(f'Target positive proportion: {target_pos_proportion:.4f}')

    # Compute positive pair counts per isolate
    isolate_pos_counts = compute_isolate_pair_counts(df)

    # Stratify isolates by positive pair counts
    unique_isolates = list(df['assembly_id'].unique())
    pos_count_groups = {}
    for aid in unique_isolates:
        pos_count = isolate_pos_counts[aid]
        if pos_count not in pos_count_groups:
            pos_count_groups[pos_count] = []
        pos_count_groups[pos_count].append(aid)

    # Split each group proportionally
    train_isolates, val_isolates, test_isolates = [], [], []
    for pos_count, isolates in pos_count_groups.items():
        if len(isolates) <= 1:
            # Assign single-isolate groups to train to avoid split errors
            train_isolates.extend(isolates)
            continue
        train, temp = train_test_split(isolates, train_size=train_ratio, random_state=seed)
        val, test = train_test_split(temp, train_size=val_ratio/(1-train_ratio), random_state=seed)
        train_isolates.extend(train)
        val_isolates.extend(val)
        test_isolates.extend(test)

    # Ensure hard partition of isolates
    if hard_partition_isolates:
        val_isolates = [aid for aid in val_isolates if aid not in train_isolates]
        test_isolates = [aid for aid in test_isolates if aid not in train_isolates and aid not in val_isolates]
        unassigned = [aid for aid in unique_isolates if aid not in train_isolates and aid not in val_isolates and aid not in test_isolates]
        for aid in unassigned:
            set_sizes = {'train': len(train_isolates), 'val': len(val_isolates), 'test': len(test_isolates)}
            smallest_set = min(set_sizes, key=set_sizes.get)
            if smallest_set == 'train':
                train_isolates.append(aid)
            elif smallest_set == 'val':
                val_isolates.append(aid)
            else:
                test_isolates.append(aid)

    # Assign pairs
    train_pairs = pairs_df[pairs_df['assembly_id_a'].isin(train_isolates)]
    val_pairs = pairs_df[pairs_df['assembly_id_a'].isin(val_isolates)]
    test_pairs = pairs_df[pairs_df['assembly_id_a'].isin(test_isolates)]

    # Validate assignments
    total_pairs = len(train_pairs) + len(val_pairs) + len(test_pairs)
    if total_pairs != len(pairs_df):
        print(f'Warning: {len(pairs_df) - total_pairs} pairs not assigned.')
    if len(train_pairs) == 0 or len(val_pairs) == 0 or len(test_pairs) == 0:
        raise ValueError('One or more sets are empty.')

    # Check isolate overlap
    if hard_partition_isolates:
        train_set = set(train_isolates)
        val_set = set(val_isolates)
        test_set = set(test_isolates)
        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set
        if train_val_overlap or train_test_overlap or val_test_overlap:
            print(f"Warning: Overlap detected in isolate sets!")
            if train_val_overlap:
                print(f"Train-Val overlap: {train_val_overlap}")
            if train_test_overlap:
                print(f"Train-Test overlap: {train_test_overlap}")
            if val_test_overlap:
                print(f"Val-Test overlap: {val_test_overlap}")
        else:
            print("No overlap in isolates: Train, Val, and Test sets are mutually exclusive.")
        if len(train_set) + len(val_set) + len(test_set) != len(unique_isolates):
            print(f"Warning: {len(unique_isolates) - (len(train_set) + len(val_set) + len(test_set))} isolates not assigned.")

    # Log stats
    total_isolates = len(unique_isolates)
    print(f"Train pairs: {len(train_pairs)} ({len(train_pairs)/total_pairs*100:.2f}%)")
    print(f"Val pairs:   {len(val_pairs)} ({len(val_pairs)/total_pairs*100:.2f}%)")
    print(f"Test pairs:  {len(test_pairs)} ({len(test_pairs)/total_pairs*100:.2f}%)")
    print(f"Train isolates: {len(set(train_isolates))} ({len(set(train_isolates))/total_isolates*100:.2f}%)")
    print(f"Val isolates:   {len(set(val_isolates))} ({len(set(val_isolates))/total_isolates*100:.2f}%)")
    print(f"Test isolates:  {len(set(test_isolates))} ({len(set(test_isolates))/total_isolates*100:.2f}%)")
    print(f"Train positive pairs: {train_pairs['label'].sum()} ({train_pairs['label'].sum()/len(train_pairs)*100:.2f}%)")
    print(f"Val positive pairs:   {val_pairs['label'].sum()} ({val_pairs['label'].sum()/len(val_pairs)*100:.2f}%)")
    print(f"Test positive pairs:  {test_pairs['label'].sum()} ({test_pairs['label'].sum()/len(test_pairs)*100:.2f}%)")
    print(f"Train positive proportion: {train_pairs['label'].sum()/len(train_pairs):.4f}")
    print(f"Val positive proportion:   {val_pairs['label'].sum()/len(val_pairs):.4f}")
    print(f"Test positive proportion:  {test_pairs['label'].sum()/len(test_pairs):.4f}")

    return train_pairs, val_pairs, test_pairs

# Load and preprocess data
print('\nLoad filtered protein data.')
fname = 'protein_filtered.csv'
datapath = processed_data_dir / fname
try:
    prot_df = pd.read_csv(datapath)
except FileNotFoundError:
    raise FileNotFoundError(f"Data file not found at: {datapath}")

# Restrict to core proteins
if use_core_proteins_only:
    core_funcs = {
        'L': 'RNA-dependent RNA polymerase',
        'M': 'Pre-glycoprotein polyprotein GP complex',
        'S': 'Nucleocapsid protein'
    }
    mask = (
        prot_df['canonical_segment'].isin(core_funcs) &
        prot_df.apply(lambda row: row['function'] == core_funcs[row['canonical_segment']], axis=1)
    )
    df_core = prot_df[mask].reset_index(drop=True)

# Preprocess
df_core['prot_seq'] = df_core['prot_seq'].apply(lambda x: x.replace('*', ''))
df_core['segment'] = df_core['canonical_segment'].fillna(
    df_core['replicon_type'].map({
        'Large RNA Segment': 'L', 'Segment One': 'L',
        'Medium RNA Segment': 'M', 'Segment Two': 'M',
        'Small RNA Segment': 'S', 'Segment Three': 'S'
    })
)
if df_core['segment'].isnull().any():
    raise ValueError("Some proteins have undefined segments.")
df_core['seq_hash'] = df_core['prot_seq'].apply(lambda x: hashlib.md5(str(x).encode()).hexdigest())
df_core['has_ambiguity'] = df_core['prot_seq'].str.contains('[^ACDEFGHIKLMNPQRSTVWY]', regex=True, na=False)
if df_core['has_ambiguity'].any():
    print(f"Warning: {df_core['has_ambiguity'].sum()} sequences contain non-standard amino acids.")

# Validate brc_fea_id
dups = df_core[df_core.duplicated(subset=['assembly_id', 'brc_fea_id'], keep=False)]
if not dups.empty:
    raise ValueError(f"Duplicate brc_fea_id found: {dups[['assembly_id', 'brc_fea_id']]}")

# Create pairs
print('\nCreate protein pairs.')
# neg_to_pos_ratio = 3
# neg_to_pos_ratio = 2
neg_to_pos_ratio = 1
pairs_df = create_protein_pairs(
    df_core,
    neg_to_pos_ratio=neg_to_pos_ratio,
    allow_same_func_negatives=True,
    max_same_func_ratio=0.5,
    seed=SEED
)

# Split dataset
print('\nSplit dataset into train, val, and test sets.')
train_pairs, val_pairs, test_pairs = split_dataset(
    pairs_df,
    df_core,
    hard_partition_isolates=True,
    hard_partition_duplicates=False,
)

# Save datasets
print('\nSave datasets.')
breakpoint()
train_pairs.to_csv(f"{output_dir}/train_pairs.csv", index=False)
val_pairs.to_csv(f"{output_dir}/val_pairs.csv", index=False)
test_pairs.to_csv(f"{output_dir}/test_pairs.csv", index=False)

print('\nDone!')