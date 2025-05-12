"""
Key columns:
- assembly_id: identifies isolates
- canonical_segment: maps each protein to a segments: L / M / S 
- prot_seq: protein sequence
- function: protein function
- brc_fea_id: unique feature ID (used to distinguish entries)

Split strategies:
- partition_by_isolate: all proteins from an isolate (assembly_id) are assigned
    to a single train/val/test set to prevent leakage
- partition_by_duplicates: identical protein sequences (prot_seq) across different
    isolates are assigned to the same set, preventing identical sequences from
    appearing across train/val/test sets
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


# ===================================
# Config
# ===================================
SEED = 42
TASK_NAME = 'segmatch'

filepath = Path(__file__).resolve().parent

main_data_dir = filepath / '../../data'

# data_version = 'Feb_2025'
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

    np.random.seed(SEED)
    random.seed(SEED) 
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
        
        # Get all posible protein pairs within an isolate (by definition these are pos pairs)
        pairs = list(combinations(grp.itertuples(), 2)) # [(df_row, df_row), (df_row, df_row), ...]
        for row_a, row_b in pairs:
            # Add pair only if they have different BRC ids and different functions
            if row_a.brc_fea_id != row_b.brc_fea_id and row_a.function != row_b.function:
                dct = {
                    'assembly_id_a': assembly_id, 'assembly_id_b': assembly_id,
                    'brc_a': row_a.brc_fea_id, 'brc_b': row_b.brc_fea_id,
                    'seq_a': row_a.prot_seq, 'seq_b': row_b.prot_seq,
                    'seg_a': row_a.segment, 'seg_b': row_b.segment,
                    'func_a': row_a.function, 'func_b': row_b.function,
                    'seq_hash_a': row_a.seq_hash, 'seq_hash_b': row_b.seq_hash,
                    'label': 1  # Positive pair
                }
                pos_pairs.append(dct)

    pos_df = pd.DataFrame(pos_pairs)
    # print(pos_df[['brc_a', 'brc_b', 'seg_a', 'seg_b', 'func_a', 'func_b']])

    # Log isolates with fewer than 2 proteins
    # If an isolate has only one protein, it won’t generate any positive pairs.
    if isolates_with_few_proteins:
        print(f'Warning: {len(isolates_with_few_proteins)} isolates have <2 \
            proteins: {isolates_with_few_proteins}')

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
                'assembly_id_a': row_a.assembly_id, 'assembly_id_b': row_b.assembly_id,
                'brc_a': row_a.brc_fea_id, 'brc_b': row_b.brc_fea_id,
                'seq_a': row_a.prot_seq, 'seq_b': row_b.prot_seq,
                'seg_a': row_a.segment, 'seg_b': row_b.segment,
                'func_a': row_a.function, 'func_b': row_b.function,
                'seq_hash_a': row_a.seq_hash, 'seq_hash_b': row_b.seq_hash,
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
    """Split dataset into train, val, and test sets, balancing total and positive pair proportions."""

    # breakpoint()
    np.random.seed(seed)
    random.seed(seed)

    # Extract neg_to_pos_ratio
    jj = pairs_df['label'].value_counts().reset_index()
    neg_count = jj.loc[jj['label'] == 0, 'count'].iloc[0] if 0 in jj['label'].values else 0
    pos_count = jj.loc[jj['label'] == 1, 'count'].iloc[0] if 1 in jj['label'].values else 1
    neg_to_pos_ratio = neg_count / pos_count if pos_count > 0 else 3.0
    target_pos_proportion = 1 / (1 + neg_to_pos_ratio)
    print(f'Negative to positive ratio: {neg_to_pos_ratio:.2f}')
    print(f'Target positive proportion: {target_pos_proportion:.4f}')

    # Compute positive and total pair counts per isolate
    isolate_pos_counts = {}  # pos pairs that will originate from this isolate
    isolate_pair_counts = {} # total pairs (pos and neg) that will originate from this isolate
    isolates = df.groupby('assembly_id')
    for aid, grp in isolates:
        n_proteins = len(grp)
        if n_proteins < 2:
            isolate_pos_counts[aid] = 0
            isolate_pair_counts[aid] = 0
        else:
            # Get all posible protein pairs within an isolate (by definition these are pos pairs)
            pairs = list(combinations(grp.itertuples(), 2))
            # Cound protein pairs that have different functions
            pos_count = sum(1 for row_a, row_b in pairs if row_a.function != row_b.function)
            isolate_pos_counts[aid] = pos_count
            isolate_pair_counts[aid] = pos_count * (1 + neg_to_pos_ratio)


    # Split to balance total pairs and positive proportions
    def split_by_pairs(items, pos_weights, total_weights, train_ratio, val_ratio, total_pairs):
        """
        Returns:
            [List, List, List] : Three lists, each containing the unqiue items
                for each set 
        """
        # Initial random split with hard partition
        # TODO. Is it a "hard partition" because we pass items=unique_isolates?
        # TODO. Does it make sense to rename train_items to unq_train_items?
        train_items, temp_items = train_test_split(items, train_size=train_ratio, random_state=seed)
        val_items, test_items = train_test_split(temp_items, train_size=val_ratio/(1-train_ratio), random_state=seed)
        
        # Ensure hard partition of isolates
        if hard_partition_isolates:
            # Remove any overlapping isolates
            val_items = [item for item in val_items if (item not in train_items)]
            test_items = [item for item in test_items if (item not in train_items) and (item not in val_items)]
            # Redistribute any unassigned isolates
            unassigned = [item for item in items if (item not in train_items) and (item not in val_items) and (item not in test_items)]
            # Assign to smallest set
            # TODO. Since we pass items=unique_isolates, I expect unassigned to be empty. Am I missing something?
            while unassigned:
                set_sizes = {
                    'train': len(train_items),
                    'val': len(val_items),
                    'test': len(test_items)
                }
                smallest_set = min(set_sizes, key=set_sizes.get)
                if smallest_set == 'train':
                    train_items.append(unassigned.pop())
                elif smallest_set == 'val':
                    val_items.append(unassigned.pop())
                else:
                    test_items.append(unassigned.pop())
        
        # Compute pair counts
        def get_pos_pair_count(items):
            # E.g., pos_weights can be isolate_pos_counts
            return sum(pos_weights[item] for item in items)
        
        def get_total_pair_count(items):
            # E.g., total_weights can be isolate_pair_counts
            return sum(total_weights[item] for item in items)
        
        # Considering the unique items in each set (T/V/E), compute how many
        # POSITIVE samples should be contributed to the each set
        train_pos   = get_pos_pair_count(train_items)
        val_pos     = get_pos_pair_count(val_items)
        test_pos    = get_pos_pair_count(test_items)
        # Considering the unique items in each set (T/V/E), compute how many
        # TOTAL samples should be contributed to the each set
        train_total = get_total_pair_count(train_items)
        val_total   = get_total_pair_count(val_items)
        test_total  = get_total_pair_count(test_items)

        # Compute the TARGET (desired) numbers of pairs (samples) in each set
        # (T/V/E), based on total sample size (total_pairs) and the required
        # ratios
        train_total_target = total_pairs * train_ratio
        val_total_target   = total_pairs * val_ratio
        test_total_target  = total_pairs * (1 - train_ratio - val_ratio)

        # Iteratively adjust to balance total pairs and positive proportions
        max_iterations = 200
        for _ in range(max_iterations):
            # Compute deviations of the existing from the targeted
            train_total_diff = abs(train_total - train_total_target) / train_total_target
            val_total_diff   = abs(val_total   - val_total_target)   / val_total_target
            test_total_diff  = abs(test_total  - test_total_target)  / test_total_target
            val_test_total_diff = abs(val_total - test_total) / val_total_target
            train_pos_diff = abs((train_pos / train_total if train_total > 0 else 0) - target_pos_proportion) / target_pos_proportion
            val_pos_diff   = abs((val_pos   /     val_total if val_total > 0 else 0) - target_pos_proportion) / target_pos_proportion
            test_pos_diff  = abs((test_pos  /   test_total if test_total > 0 else 0) - target_pos_proportion) / target_pos_proportion

            total_diff = train_total_diff + val_total_diff + test_total_diff + val_test_total_diff
            pos_diff = train_pos_diff + val_pos_diff + test_pos_diff

            if total_diff <= 0.05 and pos_diff <= 0.02:
                break

            # Find item to move
            # TODO. What's going on here?
            all_sets = [
                (train_items, train_total, train_total_target, train_pos, 'train'),
                (val_items,   val_total,   val_total_target,   val_pos,   'val'),
                (test_items,  test_total,  test_total_target,  test_pos,  'test')
            ]

            # Prioritize moving from set with largest combined deviation
            source = max(all_sets, key=lambda x: (
                abs(x[1] - x[2]) / x[2] + 
                2 * abs((x[3] / x[1] if x[1] > 0 else 0) - target_pos_proportion) / target_pos_proportion
            ) if x[1] > x[2] or (x[3] / x[1] if x[1] > 0 else 0) > target_pos_proportion else -float('inf'))

            dest = min(all_sets, key=lambda x: (
                abs(x[1] - x[2]) / x[2] + 
                2 * abs((x[3] / x[1] if x[1] > 0 else 0) - target_pos_proportion) / target_pos_proportion
            ) if x[1] < x[2] or (x[3] / x[1] if x[1] > 0 else 0) < target_pos_proportion else float('inf'))
            
            if source[1] <= source[2] and (source[3] / source[1] if source[1] > 0 else 0) <= target_pos_proportion or \
               dest[1] >= dest[2] and (dest[3] / dest[1] if dest[1] > 0 else 0) >= target_pos_proportion:
                break
                
            source_items, source_total, source_total_target, source_pos, source_name = source
            dest_items, dest_total, dest_total_target, dest_pos, dest_name = dest
            if not source_items:
                break

            # Move item with smallest impact
            impacts = []
            for item in source_items:
                total_weight = total_weights[item]
                pos_weight = pos_weights[item]
                new_source_total = source_total - total_weight
                new_source_pos = source_pos - pos_weight
                new_dest_total = dest_total + total_weight
                new_dest_pos = dest_pos + pos_weight
                new_source_total_diff = abs(new_source_total - source_total_target) / source_total_target
                new_dest_total_diff = abs(new_dest_total - dest_total_target) / dest_total_target
                new_val_test_total_diff = abs((new_dest_total if dest_name == 'val' else val_total) - 
                                             (new_dest_total if dest_name == 'test' else test_total)) / val_total_target
                new_source_pos_diff = abs((new_source_pos / new_source_total if new_source_total > 0 else 0) - target_pos_proportion) / target_pos_proportion
                new_dest_pos_diff = abs((new_dest_pos / new_dest_total if new_dest_total > 0 else 0) - target_pos_proportion) / target_pos_proportion
                impact = new_source_total_diff + new_dest_total_diff + new_val_test_total_diff + 2 * (new_source_pos_diff + new_dest_pos_diff)
                impacts.append((item, impact, total_weight, pos_weight))
            
            if not impacts:
                break
                
            item_to_move, _, total_weight, pos_weight = min(impacts, key=lambda x: x[1])
            source_items.remove(item_to_move)
            dest_items.append(item_to_move)
            source_total -= total_weight
            dest_total += total_weight
            source_pos -= pos_weight
            dest_pos += pos_weight
            
            # Update set assignments
            if source_name == 'train':
                train_items, train_total, train_pos = source_items, source_total, source_pos
            elif source_name == 'val':
                val_items, val_total, val_pos = source_items, source_total, source_pos
            else:
                test_items, test_total, test_pos = source_items, source_total, source_pos
            if dest_name == 'train':
                train_items, train_total, train_pos = dest_items, dest_total, dest_pos
            elif dest_name == 'val':
                val_items, val_total, val_pos = dest_items, dest_total, dest_pos
            else:
                test_items, test_total, test_pos = dest_items, dest_total, dest_pos
        
        # Final hard partition check
        if hard_partition_isolates:
            val_items = [item for item in val_items if item not in train_items]
            test_items = [item for item in test_items if item not in train_items and item not in val_items]
        
        return train_items, val_items, test_items


    unique_isolates = list(df['assembly_id'].unique())
    train_isolates, val_isolates, test_isolates = split_by_pairs(
        items=unique_isolates,
        pos_weights=isolate_pos_counts,
        total_weights=isolate_pair_counts,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        total_pairs=len(pairs_df)
    )

    # Assign pairs based on assembly_id_a
    train_pairs = pairs_df[pairs_df['assembly_id_a'].isin(train_isolates)]
    val_pairs   = pairs_df[pairs_df['assembly_id_a'].isin(val_isolates)]
    test_pairs  = pairs_df[pairs_df['assembly_id_a'].isin(test_isolates)]

    # Validate assignment
    total_pairs = len(train_pairs) + len(val_pairs) + len(test_pairs)
    if total_pairs != len(pairs_df):
        print(f'Warning: {len(pairs_df) - total_pairs} pairs not assigned.')
    if len(train_pairs) == 0 or len(val_pairs) == 0 or len(test_pairs) == 0:
        raise ValueError('One or more sets are empty.')

    # Validate hard partition of isolates
    if hard_partition_isolates:
        train_set = set(train_isolates)
        val_set   = set(val_isolates)
        test_set  = set(test_isolates)
        train_val_overlap  = train_set & val_set
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

    # Mask for core proteins where segment and function match
    mask = (
        prot_df['canonical_segment'].isin(core_funcs) & 
        prot_df.apply(lambda row: row['function'] == core_funcs[row['canonical_segment']], axis=1)
    )
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
# neg_to_pos_ratio = 2
# neg_to_pos_ratio = 1
allow_same_func_negatives = True
max_same_func_ratio = 0.5
pairs_df = create_protein_pairs(
    df_core,
    neg_to_pos_ratio=neg_to_pos_ratio,
    allow_same_func_negatives=allow_same_func_negatives,
    max_same_func_ratio=max_same_func_ratio,
    seed=SEED
)

# Split dataset
print('\nSplit dataset into train, val, and test sets.')
# hard_partition_isolates = False
hard_partition_isolates = True
hard_partition_duplicates = False
# hard_partition_duplicates = True
train_pairs, val_pairs, test_pairs = split_dataset(
    pairs_df,
    df_core,
    hard_partition_isolates,
    hard_partition_duplicates,
)

# Save datasets
breakpoint()
print('\nSave datasets.')
train_pairs.to_csv(f"{output_dir}/train_pairs.csv", index=False)
val_pairs.to_csv(f"{output_dir}/val_pairs.csv", index=False)
test_pairs.to_csv(f"{output_dir}/test_pairs.csv", index=False)

# ----------------------------------------------------------------
# breakpoint()
print('\nDone!')