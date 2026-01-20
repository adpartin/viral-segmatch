"""
Dataset creation for segment pair classification.

Key columns:
- assembly_id: identifies isolates
- canonical_segment: maps each protein to segments: L / M / S 
- prot_seq: protein sequence
- function: protein function
- brc_fea_id: unique feature ID

Split strategies:
- hard_partition_isolates: all proteins from an isolate (assembly_id) are assigned
    to a single train/val/test set to prevent leakage

Duplicate handling (v2 - blocked negatives):
- Identical amino-acid sequences recur across many genomes/isolates
- A pair (seq_a, seq_b) can appear as positive (same isolate) AND negative (different isolates)
- This creates contradictory labels and potential data leakage
- Solution: Block negative pairs where sequences co-occur in ANY isolate
- Split by pair_key (not just isolate) to prevent same pair appearing in train and test
"""

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# print(f'project_root: {project_root}')

from src.utils.timer_utils import Timer
from src.utils.config_hydra import get_virus_config_hydra, print_config_summary
from src.utils.seed_utils import resolve_process_seed, set_deterministic_seeds
from src.utils.path_utils import resolve_run_suffix, build_dataset_paths, load_dataframe
from src.utils.metadata_enrichment import enrich_prot_data_with_metadata

total_timer = Timer()


def canonical_pair_key(seq_hash_a: str, seq_hash_b: str) -> str:
    """Create a canonical pair key from two sequence hashes.

    Ensures consistent ordering so (a, b) and (b, a) produce the same key.
    """
    return "__".join(sorted([seq_hash_a, seq_hash_b]))


def build_cooccurrence_set(df: pd.DataFrame) -> tuple[set, dict]:
    """Build a set of all sequence pairs that co-occur in any isolate.

    Two sequences "co-occur" if they appear together in the same isolate (same assembly_id).
    For example, if PB2 sequence A and PB1 sequence B both appear in isolate X, they co-occur.

    Purpose: Prevent contradictory labels in the dataset.
    - If sequences (A, B) co-occur in isolate X, they form a positive pair (same isolate).
    - If we then use (A, B) from different isolates as a negative pair, we have a contradiction:
      * They co-occurred in isolate X → positive label
      * But we're labeling them as "not from same isolate" → negative label
    - Solution: Block all pairs that co-occur in ANY isolate from being used as negatives.

    This function identifies all such pairs by iterating through each isolate and recording
    all sequence pairs that appear together within that isolate.

    Args:
        df: DataFrame with protein data. Must contain columns:
            - 'assembly_id': Isolate identifier
            - 'seq_hash': Unique hash for each protein sequence
            - 'prot_seq': Protein sequence (optional, for reference)

    Returns:
        Tuple of:
        - cooccur_pairs: Set of canonical pair keys (format: "seq_hash_a__seq_hash_b")
          representing all sequence pairs that co-occur in at least one isolate.
          These pairs should be blocked when creating negative pairs.
        - cooccur_stats: Dict with statistics:
            - 'total_cooccur_pairs': Total number of unique co-occurring pairs
            - 'max_isolates_per_pair': Maximum number of isolates a single pair appears in
            - 'pairs_in_multiple_isolates': Count of pairs that appear in >1 isolate
            - 'isolate_pair_counts': Dict mapping pair_key -> number of isolates it appears in
    """
    cooccur_pairs = set()
    isolate_pair_counts = {}  # Track how many isolates each pair appears in

    for aid, grp in df.groupby('assembly_id'):
        if len(grp) < 2:
            continue

        # Get all unique sequences in this isolate
        seq_hashes = grp['seq_hash'].unique().tolist()

        # All pairs of sequences in this isolate co-occur
        for i in range(len(seq_hashes)):
            for j in range(i + 1, len(seq_hashes)):
                pair_key = canonical_pair_key(seq_hashes[i], seq_hashes[j])
                cooccur_pairs.add(pair_key)

                # Track count for stats
                if pair_key not in isolate_pair_counts:
                    isolate_pair_counts[pair_key] = 0
                isolate_pair_counts[pair_key] += 1

    # Compute statistics
    cooccur_stats = {
        'total_cooccur_pairs': len(cooccur_pairs),
        'max_isolates_per_pair': max(isolate_pair_counts.values()) if isolate_pair_counts else 0,
        'pairs_in_multiple_isolates': sum(1 for c in isolate_pair_counts.values() if c > 1),
        'isolate_pair_counts': isolate_pair_counts  # Full mapping for detailed analysis
    }

    return cooccur_pairs, cooccur_stats


def create_positive_pairs(
    df: pd.DataFrame,
    seed: int = 42,
    ) -> pd.DataFrame:
    """ Create positive protein pairs (within-isolate and cross-function).
    For example, creates pairs within an isolate: RdRp-GPC, RdRp-N, GPC-N

    Symmetric pairs handling (e.g., [seq_a, seq_b] and [seq_b, seq_a]).
    Uses combinations (instead of permutations) to create positive pairs to
    avoid duplicates that stem from symmetric pairs.

    Each pair gets a canonical pair_key based on sequence hashes for:
    1. Deduplication across isolates (same sequences in different isolates)
    2. Preventing data leakage during train/test split
    """
    # np.random.seed(seed)
    # random.seed(seed)
    isolates = df.groupby('assembly_id')

    pos_pairs = []
    isolates_with_few_proteins = []

    for assembly_id, grp in isolates:
        # print(grp[['file', 'assembly_id', 'canonical_segment', 'replicon_type', 'function', 'brc_fea_id']])
        # Ignore isolates with fewer than 2 proteins (these cannot be used to
        # create positive pairs, but these will be used to create negative pairs)
        if len(grp) < 2:
            isolates_with_few_proteins.append(assembly_id)
            continue

        # Get all posible pairs within an isolate
        pairs = list(combinations(grp.itertuples(), 2)) # [(df_row, df_row), (df_row, df_row), ...]
        for row_a, row_b in pairs:
            # Add the pair only if the proteins have different BRC ids and different functions
            if row_a.brc_fea_id != row_b.brc_fea_id and row_a.function != row_b.function:
                # Create canonical pair_key based on sequence hashes
                pair_key = canonical_pair_key(row_a.seq_hash, row_b.seq_hash)
                dct = {
                    'pair_key': pair_key,  # Canonical key for dedup and splitting
                    'assembly_id_a': assembly_id, 'assembly_id_b': assembly_id,
                    'brc_a': row_a.brc_fea_id, 'brc_b': row_b.brc_fea_id,
                    'seq_a': row_a.prot_seq, 'seq_b': row_b.prot_seq,
                    'seg_a': row_a.canonical_segment, 'seg_b': row_b.canonical_segment,
                    'func_a': row_a.function, 'func_b': row_b.function,
                    'seq_hash_a': row_a.seq_hash, 'seq_hash_b': row_b.seq_hash,
                    'label': 1  # By definition a positive pair
                }
                pos_pairs.append(dct)

    pos_df = pd.DataFrame(pos_pairs)
    # print(pos_df[['brc_a', 'brc_b', 'seg_a', 'seg_b', 'func_a', 'func_b']])

    # Log isolates with fewer than 2 proteins
    # If an isolate has only one protein, it won't generate any positive pairs.
    if isolates_with_few_proteins:
        print(f'Warning: {len(isolates_with_few_proteins)} isolates have <2 proteins.')
    return pos_df


def create_negative_pairs(
    df: pd.DataFrame,
    num_negatives: int,
    isolate_ids: list[str],
    cooccur_pairs: set,
    allow_same_func_negatives: bool = True,
    max_same_func_ratio: float = 0.5,
    seed: int = 42,
    max_attempts_multiplier: int = 100,
    ) -> tuple[pd.DataFrame, int, dict]:
    """ Create negative pairs (cross-isolate, with optional control over
    same-function pairs).

    BLOCKED NEGATIVES: Pairs where the two sequences co-occur in ANY isolate
    are blocked to prevent contradictory labels. If sequences (a, b) appear
    together in some isolate (positive), they cannot be used as a negative pair.

    Same-function pairs (e.g., RdRp from A vs. RdRp from B)
    Do we want this in negative pairs? These pairs are important since same-function
    proteins across isolates are more similar (due to functional conservation)
    than cross-function proteins within an isolate. Excluding them could make
    the task too easy, as the model might rely on functional differences alone.
    However, you want to control their ratio and log their prevalence to ensure
    the dataset isn't dominated by same-function negatives.

    Symmetric pairs handling (e.g., [seq_a, seq_b] and [seq_b, seq_a]).
    Tracks seen_pairs when creating negative pairs to prevent symmetric
    duplicates.

    Args:
        df: DataFrame containing the protein data
        num_negatives: Number of negative pairs to create
        isolate_ids: List of isolate IDs to sample from
        cooccur_pairs: Set of canonical pair keys that co-occur in any isolate (blocked)
        allow_same_func_negatives: Whether to allow same-function negative pairs
        max_same_func_ratio: Maximum fraction of same-function negative pairs
        seed: Random seed (not used directly, seeding done upstream)
        max_attempts_multiplier: Max sampling attempts = num_negatives * this value

    Returns:
        Tuple of:
        - DataFrame of negative pairs
        - Number of same-function negative pairs
        - Dict with rejection statistics
    """
    # np.random.seed(seed)
    # random.seed(seed)

    neg_pairs = []
    seen_pairs = set()  # Track unique pairs by brc_fea_id
    seen_seq_pairs = set()  # Track unique pairs by sequence hash

    # Precompute isolate groups to prevent repeated (trial-and-error) sampling.
    # This ensures that 2 different isolates (assembly_id) are sampled during sampling.
    df_subset = df[df['assembly_id'].isin(isolate_ids)].reset_index(drop=True)
    isolate_groups = {aid: list(grp.itertuples()) for aid, grp in df_subset.groupby('assembly_id')}

    # If allows to generate same-function negative pairs, then compute the max
    # fraction of such pairs out of all negative pairs
    same_func_count = 0
    max_same_func = int(num_negatives * max_same_func_ratio) if allow_same_func_negatives else 0

    # Track rejection statistics
    rejection_stats = {
        'blocked_cooccur': 0,      # Rejected because sequences co-occur (would be contradictory)
        'duplicate_brc': 0,         # Rejected because same brc_fea_id pair already seen
        'duplicate_seq': 0,         # Rejected because same sequence pair already seen
        'same_func_limit': 0,       # Rejected due to same-function ratio limit
        'total_attempts': 0,
    }

    max_attempts = num_negatives * max_attempts_multiplier
    attempts = 0

    while len(neg_pairs) < num_negatives and attempts < max_attempts:
        attempts += 1
        aid1, aid2 = random.sample(isolate_ids, 2)  # Sample 2 different isolates
        row_a = random.choice(isolate_groups[aid1]) # Sample a random protein from the 1st isolate
        row_b = random.choice(isolate_groups[aid2]) # Sample a random protein from the 2nd isolate

        # Check if brc_fea_id pair is unique and not symmetric
        brc_pair_key = tuple(sorted([row_a.brc_fea_id, row_b.brc_fea_id]))
        if brc_pair_key in seen_pairs:
            rejection_stats['duplicate_brc'] += 1
            continue

        # Create canonical pair key based on sequence hashes
        seq_pair_key = canonical_pair_key(row_a.seq_hash, row_b.seq_hash)

        # BLOCK CONTRADICTORY PAIRS: Check if these sequences ever co-occur
        if seq_pair_key in cooccur_pairs:
            rejection_stats['blocked_cooccur'] += 1
            continue

        # Check if we've already seen this sequence pair (different brc_ids, same sequences)
        if seq_pair_key in seen_seq_pairs:
            rejection_stats['duplicate_seq'] += 1
            continue

        # Check same-function constraint
        is_same_func = row_a.function == row_b.function
        if is_same_func and (not allow_same_func_negatives or same_func_count >= max_same_func):
            rejection_stats['same_func_limit'] += 1
            continue

        dct = {
            'pair_key': seq_pair_key,  # Canonical key for dedup and splitting
            'assembly_id_a': row_a.assembly_id, 'assembly_id_b': row_b.assembly_id,
            'brc_a': row_a.brc_fea_id, 'brc_b': row_b.brc_fea_id,
            'seq_a': row_a.prot_seq, 'seq_b': row_b.prot_seq,
            'seg_a': row_a.canonical_segment, 'seg_b': row_b.canonical_segment,
            'func_a': row_a.function, 'func_b': row_b.function,
            'seq_hash_a': row_a.seq_hash, 'seq_hash_b': row_b.seq_hash,
            'label': 0  # Negative pair
        }
        neg_pairs.append(dct)
        seen_pairs.add(brc_pair_key)
        seen_seq_pairs.add(seq_pair_key)
        if is_same_func:
            same_func_count += 1

    rejection_stats['total_attempts'] = attempts

    if len(neg_pairs) < num_negatives:
        print(f"⚠️ Warning: Only generated {len(neg_pairs)}/{num_negatives} negative pairs after {attempts} attempts")
        print(f"   This may indicate high sequence overlap across isolates")

    return pd.DataFrame(neg_pairs), same_func_count, rejection_stats
        

def compute_isolate_pair_counts(
    df: pd.DataFrame,
    verbose: bool = False,
    ) -> dict:
    """Compute positive pair counts per isolate for stratified splitting.
    
    Counts how many positive pairs (cross-function pairs within the same isolate)
    can be generated from each isolate. This is used to stratify isolates by
    positive pair count for balanced train/val/test splits.
    
    Note: Data filtering (e.g., by selected_functions) should be done before
    calling this function. This function does not validate data quality.
    
    Args:
        df: DataFrame with protein data, already filtered to desired functions
        verbose: If True, print detailed information per isolate
    
    Returns:
        Dict mapping assembly_id -> number of positive pairs that can be generated
    """
    isolate_pos_counts = {} # pos pairs that will originate from this isolate

    for aid, grp in df.groupby('assembly_id'): # isolates
        n_proteins = len(grp)

        if verbose:
            print(f"assembly_id {aid}: {n_proteins} proteins, Functions: {grp['function'].tolist()}")

        if n_proteins < 2:
            # Need at least 2 proteins to form a pair
            isolate_pos_counts[aid] = 0
        else:
            # Get all possible protein pairs within the isolate (by definition these are pos pairs)
            pairs = list(combinations(grp.itertuples(), 2))
            # Count all possible protein pairs that have different functions
            # (same-function pairs within an isolate are not used as positive pairs)
            pos_count = sum(1 for row_a, row_b in pairs if row_a.function != row_b.function)
            isolate_pos_counts[aid] = pos_count

    return isolate_pos_counts


def split_dataset(
    df: pd.DataFrame,
    neg_to_pos_ratio: float = 3.0,
    allow_same_func_negatives: bool = True,
    max_same_func_ratio: float = 0.5,
    hard_partition_isolates: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """ Split dataset into train, val, and test sets with stratified sampling.

    Key features:
    1. BLOCKED NEGATIVES: Prevents contradictory labels by excluding negative pairs
       where sequences co-occur in any isolate (uses cooccur_pairs set).
    2. PAIR-KEY VALIDATION: After splitting by isolates, validates that no pair_key
       appears across train/val/test splits. If overlap is found, removes those pairs
       from val/test to prevent data leakage.

    Approach:
    - First split ISOLATES into train/val/test (keeps all proteins from an isolate together)
    - Generate positive pairs within each isolate group
    - Generate negative pairs with blocked co-occurring sequences
    - Validate no pair_key leakage across splits (remove if found)

    Returns:
        Tuple of (train_pairs, val_pairs, test_pairs, duplicate_stats) DataFrames/dict
    """
    # Build co-occurrence set FIRST (before any pair generation)
    # This identifies all sequence pairs that appear together in any isolate
    print("\n🔍 Building co-occurrence set (sequences that appear together in any isolate)...")
    cooccur_pairs, cooccur_stats = build_cooccurrence_set(df)
    print(f"   Total co-occurring sequence pairs: {cooccur_stats['total_cooccur_pairs']:,}")
    print(f"   Pairs appearing in multiple isolates: {cooccur_stats['pairs_in_multiple_isolates']:,}")
    print(f"   Max isolates for a single pair: {cooccur_stats['max_isolates_per_pair']}")

    # Compute positive pair counts per isolate
    isolate_pos_counts = compute_isolate_pair_counts(df)

    # Stratify isolates by positive pair counts for balanced train/val/test splits.
    # 
    # Purpose: Ensures that train/val/test splits have similar distributions of
    # isolates with different numbers of positive pairs. This prevents scenarios
    # where, for example, all high-pair-count isolates end up in training, leaving
    # validation/test with only low-pair-count isolates.
    #
    # When does this matter?
    # - With 2 selected_functions: Most isolates generate 1 positive pair (if they
    #   have both functions), but some may have 0 (missing proteins). Stratification
    #   ensures balanced distribution of isolates with 0 vs 1+ pairs.
    # - With 3+ selected_functions: Isolates can generate varying numbers of pairs
    #   (e.g., 3 functions = up to 3 pairs: AB, AC, BC). Some isolates may be missing
    #   one or more functions, resulting in 0, 1, 2, or 3 pairs. Stratification is
    #   especially important here to balance the distribution across splits.
    #
    # Example: If we have 100 isolates with 3 pairs each and 100 isolates with 1 pair
    # each, stratification ensures each split gets roughly 50 of each type, rather
    # than all 3-pair isolates in train and all 1-pair isolates in test.
    unique_isolates = list(df['assembly_id'].unique())
    pos_count_groups = {}
    for aid in unique_isolates:
        pos_count = isolate_pos_counts[aid]
        if pos_count not in pos_count_groups:
            pos_count_groups[pos_count] = []
        pos_count_groups[pos_count].append(aid)

    # Split isolates into train/val/test sets (~80/10/10) based on their positive pair counts.
    train_isolates, val_isolates, test_isolates = [], [], []
    for pos_count, isolates in pos_count_groups.items():
        if len(isolates) <= 1:
            train_isolates.extend(isolates)
            continue
        train, temp = train_test_split(isolates, train_size=train_ratio, random_state=seed)
        val, test = train_test_split(temp, train_size=val_ratio/(1-train_ratio), random_state=seed)
        train_isolates.extend(train)
        val_isolates.extend(val)
        test_isolates.extend(test)

    # Enforce hard partitioning on isolates
    if hard_partition_isolates:
        val_isolates  = [aid for aid in val_isolates if (aid not in train_isolates)]
        test_isolates = [aid for aid in test_isolates if (aid not in train_isolates) and (aid not in val_isolates)]
        unassigned    = [aid for aid in unique_isolates if (aid not in train_isolates) and (aid not in val_isolates) and (aid not in test_isolates)]
        if unassigned:
            print(f"Warning: {len(unassigned)} unassigned isolates.")
            for aid in unassigned:
                set_sizes = {'train': len(train_isolates), 'val': len(val_isolates), 'test': len(test_isolates)}
                smallest_set = min(set_sizes, key=set_sizes.get)
                if smallest_set == 'train':
                    train_isolates.append(aid)
                elif smallest_set == 'val':
                    val_isolates.append(aid)
                else:
                    test_isolates.append(aid)

    # Generate positive pairs for each set (already includes pair_key)
    train_pos = create_positive_pairs(df[df['assembly_id'].isin(train_isolates)], seed=seed)
    val_pos   = create_positive_pairs(df[df['assembly_id'].isin(val_isolates)], seed=seed)
    test_pos  = create_positive_pairs(df[df['assembly_id'].isin(test_isolates)], seed=seed)

    # Generate negative pairs within each set (with BLOCKED contradictory pairs)
    print("\nCreating negative pairs with blocked contradictory pairs...")
    train_neg, train_same_func, train_reject_stats = create_negative_pairs(
        df,
        num_negatives=int(len(train_pos) * neg_to_pos_ratio),
        isolate_ids=train_isolates,
        cooccur_pairs=cooccur_pairs,
        allow_same_func_negatives=allow_same_func_negatives,
        max_same_func_ratio=max_same_func_ratio, seed=seed
    )
    val_neg, val_same_func, val_reject_stats = create_negative_pairs(
        df,
        num_negatives=int(len(val_pos) * neg_to_pos_ratio),
        isolate_ids=val_isolates,
        cooccur_pairs=cooccur_pairs,
        allow_same_func_negatives=allow_same_func_negatives,
        max_same_func_ratio=max_same_func_ratio, seed=seed
    )
    test_neg, test_same_func, test_reject_stats = create_negative_pairs(
        df,
        num_negatives=int(len(test_pos) * neg_to_pos_ratio),
        isolate_ids=test_isolates,
        cooccur_pairs=cooccur_pairs,
        allow_same_func_negatives=allow_same_func_negatives,
        max_same_func_ratio=max_same_func_ratio, seed=seed
    )

    # Log rejection statistics
    total_blocked = (train_reject_stats['blocked_cooccur'] + 
                     val_reject_stats['blocked_cooccur'] + 
                     test_reject_stats['blocked_cooccur'])
    total_attempts = (train_reject_stats['total_attempts'] + 
                      val_reject_stats['total_attempts'] + 
                      test_reject_stats['total_attempts'])
    print(f"\n📊 Negative Pair Rejection Statistics:")
    print(f"   Total sampling attempts: {total_attempts:,}")
    print(f"   Blocked (contradictory co-occur): {total_blocked:,} ({100*total_blocked/max(1,total_attempts):.1f}%)")
    print(f"   Train blocked: {train_reject_stats['blocked_cooccur']:,}")
    print(f"   Val blocked:   {val_reject_stats['blocked_cooccur']:,}")
    print(f"   Test blocked:  {test_reject_stats['blocked_cooccur']:,}")

    # Combine positive and negative pairs
    train_pairs = pd.concat([train_pos, train_neg], ignore_index=True)
    val_pairs   = pd.concat([val_pos, val_neg], ignore_index=True)
    test_pairs  = pd.concat([test_pos, test_neg], ignore_index=True)

    # PAIR-KEY BASED SPLITTING: Ensure no pair_key appears across train/val/test
    # This is a VALIDATION step - our isolate-based split should already prevent this
    # if the same sequence pair doesn't appear in different isolate groups
    print("\nValidating pair_key partitioning...")
    train_pair_keys = set(train_pairs['pair_key'])
    val_pair_keys = set(val_pairs['pair_key'])
    test_pair_keys = set(test_pairs['pair_key'])
    
    # Store original sizes before removal (for percentage calculations)
    val_pairs_before = len(val_pairs)
    test_pairs_before = len(test_pairs)
    
    train_val_key_overlap = train_pair_keys & val_pair_keys
    train_test_key_overlap = train_pair_keys & test_pair_keys
    val_test_key_overlap = val_pair_keys & test_pair_keys
    
    total_key_overlap = len(train_val_key_overlap) + len(train_test_key_overlap) + len(val_test_key_overlap)
    
    if total_key_overlap > 0:
        # Calculate percentages
        train_val_pct = (len(train_val_key_overlap) / val_pairs_before * 100) if val_pairs_before > 0 else 0
        train_test_pct = (len(train_test_key_overlap) / test_pairs_before * 100) if test_pairs_before > 0 else 0
        val_test_pct = (len(val_test_key_overlap) / test_pairs_before * 100) if test_pairs_before > 0 else 0
        
        print(f"   ⚠️ WARNING: Found {total_key_overlap} overlapping pair_keys across splits!")
        print(f"      Train-Val overlap: {len(train_val_key_overlap)} ({train_val_pct:.2f}% of val set)")
        print(f"      Train-Test overlap: {len(train_test_key_overlap)} ({train_test_pct:.2f}% of test set)")
        print(f"      Val-Test overlap: {len(val_test_key_overlap)} ({val_test_pct:.2f}% of test set)")
        print(f"   These represent sequence pairs that appear in isolates assigned to different splits.")
        print(f"   This can cause data leakage. Consider re-splitting or deduplicating by pair_key.")
        
        # Remove overlapping pairs from val and test (keep in train)
        # This is a conservative approach - keep pairs in training, remove from val/test
        print(f"   Removing overlapping pairs from val and test sets...")
        val_pairs = val_pairs[~val_pairs['pair_key'].isin(train_val_key_overlap | val_test_key_overlap)]
        test_pairs = test_pairs[~test_pairs['pair_key'].isin(train_test_key_overlap | val_test_key_overlap)]
        
        # Calculate removal percentages
        val_removed = val_pairs_before - len(val_pairs)
        test_removed = test_pairs_before - len(test_pairs)
        val_removed_pct = (val_removed / val_pairs_before * 100) if val_pairs_before > 0 else 0
        test_removed_pct = (test_removed / test_pairs_before * 100) if test_pairs_before > 0 else 0
        
        print(f"   After removal: Train={len(train_pairs)}, Val={len(val_pairs)} (removed {val_removed}, {val_removed_pct:.2f}%), Test={len(test_pairs)} (removed {test_removed}, {test_removed_pct:.2f}%)")
    else:
        print(f"   ✅ No pair_key overlap detected. Train, Val, Test are mutually exclusive on pair_key.")

    # Shuffle training labels if requested (for sanity tests)
    if SHUFFLE_TRAIN_LABELS:
        shuffle_seed = SHUFFLE_TRAIN_LABELS_SEED if SHUFFLE_TRAIN_LABELS_SEED is not None else RANDOM_SEED
        print(f"\n🔄 Shuffling training labels (seed: {shuffle_seed})...")
        print(f"   Original train label distribution: {train_pairs['label'].value_counts().to_dict()}")
        
        # Use a separate RandomState to avoid affecting global numpy random state
        # This preserves reproducibility of other operations that rely on the global seed
        rng = np.random.RandomState(shuffle_seed)
        shuffled_indices = rng.permutation(len(train_pairs))
        
        # Replace labels in train_pairs using shuffled indices
        train_pairs = train_pairs.copy()
        train_pairs['label'] = train_pairs['label'].iloc[shuffled_indices].values
        
        print(f"   Shuffled train label distribution: {train_pairs['label'].value_counts().to_dict()}")
        print(f"   ⚠️  WARNING: Training labels have been shuffled! Val/Test labels remain unchanged.")
        print(f"   This is a sanity test - model should not generalize if labels are random.")

    # Compute and log dataset stats
    total_pairs = len(train_pairs) + len(val_pairs) + len(test_pairs)
    print(f'\nTotal pairs: {total_pairs}')
    print(f'Positive pairs: {len(train_pos) + len(val_pos) + len(test_pos)}')
    print(f'Negative pairs: {len(train_neg) + len(val_neg) + len(test_neg)}')
    
    train_neg_pct = (train_same_func/len(train_neg)*100) if len(train_neg) > 0 else 0
    val_neg_pct = (val_same_func/len(val_neg)*100) if len(val_neg) > 0 else 0
    test_neg_pct = (test_same_func/len(test_neg)*100) if len(test_neg) > 0 else 0

    print(f'Train same-function negative pairs: {train_same_func} ({train_neg_pct:.2f}%)')
    print(f'Val same-function negative pairs:   {val_same_func} ({val_neg_pct:.2f}%)')
    print(f'Test same-function negative pairs:  {test_same_func} ({test_neg_pct:.2f}%)')
    
    if len(train_neg) > 0 and len(val_neg) > 0 and len(test_neg) > 0:
        segment_pair_counts = pd.concat([train_neg, val_neg, test_neg]).groupby(
            ['seg_a', 'seg_b']).size().rename('count').reset_index()
        print(f'Negative pair segment count:\n{segment_pair_counts}\n')

    # Validate assignments
    if len(train_pairs) == 0 or len(val_pairs) == 0 or len(test_pairs) == 0:
        raise ValueError('One or more sets are empty.')

    # Check isolate overlap in pairs
    if hard_partition_isolates:
        train_set = set(train_pairs['assembly_id_a']).union(set(train_pairs['assembly_id_b']))
        val_set   = set(val_pairs['assembly_id_a']).union(set(val_pairs['assembly_id_b']))
        test_set  = set(test_pairs['assembly_id_a']).union(set(test_pairs['assembly_id_b']))
        train_val_overlap  = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap   = val_set & test_set
        if train_val_overlap or train_test_overlap or val_test_overlap:
            print(f"Error: Overlap detected in isolate sets!")
            if train_val_overlap:
                print(f"Train-Val overlap: {train_val_overlap}")
            if train_test_overlap:
                print(f"Train-Test overlap: {train_test_overlap}")
            if val_test_overlap:
                print(f"Val-Test overlap: {val_test_overlap}")
            raise ValueError("Isolate overlap detected in pair sets.")
        else:
            print("No overlap in isolates: Train, Val, and Test sets are mutually exclusive.")
        if len(set(unique_isolates)) != len(train_set | val_set | test_set):
            print(f"Warning: {len(unique_isolates) - len(train_set | val_set | test_set)} isolates not assigned.")

    # Log statistics
    total_isolates = len(unique_isolates)
    print(f"\nTrain pairs: {len(train_pairs)} ({len(train_pairs)/total_pairs*100:.2f}%)")
    print(f"Val pairs:   {len(val_pairs)} ({len(val_pairs)/total_pairs*100:.2f}%)")
    print(f"Test pairs:  {len(test_pairs)} ({len(test_pairs)/total_pairs*100:.2f}%)")
    print(f"\nTrain isolates: {len(train_set)} ({len(train_set)/total_isolates*100:.2f}%)")
    print(f"Val isolates:   {len(val_set)} ({len(val_set)/total_isolates*100:.2f}%)")
    print(f"Test isolates:  {len(test_set)} ({len(test_set)/total_isolates*100:.2f}%)")
    print(f"\nTrain positive pairs: {train_pairs['label'].sum()} "
          f"({train_pairs['label'].sum()/len(train_pairs)*100:.2f}% of the set)")
    print(f"Val positive pairs:   {val_pairs['label'].sum()} "
          f"({val_pairs['label'].sum()/len(val_pairs)*100:.2f}% of the set)")
    print(f"Test positive pairs:  {test_pairs['label'].sum()} "
          f"({test_pairs['label'].sum()/len(test_pairs)*100:.2f}% of the set)")

    # Compile duplicate/rejection statistics for saving
    # Use stored original sizes for percentage calculations
    duplicate_stats = {
        'cooccur_stats': cooccur_stats,
        'train_reject_stats': train_reject_stats,
        'val_reject_stats': val_reject_stats,
        'test_reject_stats': test_reject_stats,
        'pair_key_overlaps': {
            'train_val': {
                'count': len(train_val_key_overlap),
                'pct_of_val': (len(train_val_key_overlap) / val_pairs_before * 100) if val_pairs_before > 0 else 0,
            },
            'train_test': {
                'count': len(train_test_key_overlap),
                'pct_of_test': (len(train_test_key_overlap) / test_pairs_before * 100) if test_pairs_before > 0 else 0,
            },
            'val_test': {
                'count': len(val_test_key_overlap),
                'pct_of_test': (len(val_test_key_overlap) / test_pairs_before * 100) if test_pairs_before > 0 else 0,
            },
            'total_overlap': total_key_overlap,
            'val_pairs_before_removal': val_pairs_before,
            'test_pairs_before_removal': test_pairs_before,
            'val_pairs_after_removal': len(val_pairs),
            'test_pairs_after_removal': len(test_pairs),
            'val_removed_pct': ((val_pairs_before - len(val_pairs)) / val_pairs_before * 100) if val_pairs_before > 0 else 0,
            'test_removed_pct': ((test_pairs_before - len(test_pairs)) / test_pairs_before * 100) if test_pairs_before > 0 else 0,
        }
    }

    return train_pairs, val_pairs, test_pairs, duplicate_stats


# Parser
parser = argparse.ArgumentParser(description='Create segment pairs dataset')
parser.add_argument(
    '--config_bundle',
    type=str, default=None,
    help='Config bundle to use (e.g., flu_a, bunya).'
)
parser.add_argument(
    '--input_file',
    type=str, default=None,
    help='Path to input CSV file (e.g., protein_final.csv). If not provided, derived from config.'
)
parser.add_argument(
    '--output_dir',
    type=str, default=None,
    help='Path to output directory for datasets. If not provided, derived from config.'
)
parser.add_argument(
    '--run_output_subdir',
    type=str, default=None,
    help='Optional subdirectory name under default output_dir (e.g., experiment/run id).'
)
args = parser.parse_args()

# Load config
config_path = str(project_root / 'conf')  # Pass the config path explicitly
config_bundle = args.config_bundle
if config_bundle is None:
    raise ValueError("❌ Must provide --config_bundle")
config = get_virus_config_hydra(config_bundle, config_path=config_path)
print_config_summary(config)

# Extract config values
VIRUS_NAME = config.virus.virus_name
DATA_VERSION = config.virus.data_version
RANDOM_SEED = resolve_process_seed(config, 'datasets')
USE_SELECTED_ONLY = config.dataset.use_selected_only
# TASK_NAME = config.dataset.task_name
NEG_TO_POS_RATIO = config.dataset.neg_to_pos_ratio
ALLOW_SAME_FUNC_NEGATIVES = config.dataset.allow_same_func_negatives
MAX_SAME_FUNC_RATIO = config.dataset.max_same_func_ratio
TRAIN_RATIO = config.dataset.train_ratio
VAL_RATIO = config.dataset.val_ratio
HARD_PARTITION_ISOLATES = config.dataset.hard_partition_isolates
MAX_ISOLATES_TO_PROCESS = getattr(config.dataset, 'max_isolates_to_process', None)
SHUFFLE_TRAIN_LABELS = getattr(config.dataset, 'shuffle_train_labels', False)
SHUFFLE_TRAIN_LABELS_SEED = getattr(config.dataset, 'shuffle_train_labels_seed', None)

print(f"\n{'='*40}")
print(f"Virus: {VIRUS_NAME}")
print(f"Config bundle: {config_bundle}")
print(f"{'='*40}")

# Resolve run suffix (manual override in config or auto-generate from sampling params)
RUN_SUFFIX = resolve_run_suffix(
    config=config,
    max_isolates=MAX_ISOLATES_TO_PROCESS,  # Dataset-specific isolate sampling
    seed=RANDOM_SEED,
    auto_timestamp=True
)

# Set deterministic seeds for reproducible dataset creation
if RANDOM_SEED is not None:
    set_deterministic_seeds(RANDOM_SEED, cuda_deterministic=False) # No CUDA for dataset creation
    print(f'Set deterministic seeds for dataset creation (seed: {RANDOM_SEED})')
else:
    print('No seed set - dataset creation will be non-deterministic')

# Build dataset paths
paths = build_dataset_paths(
    project_root=project_root,
    virus_name=VIRUS_NAME,
    data_version=DATA_VERSION,
    run_suffix=RUN_SUFFIX,
    config=config
)

default_input_file = paths['input_file']
default_output_dir = paths['output_dir']

# Apply CLI overrides if provided
input_file = Path(args.input_file) if args.input_file else default_input_file
if args.output_dir:
    output_dir = Path(args.output_dir)
elif args.run_output_subdir:
    # Always use runs/ subdirectory for consistency
    # Structure: data/datasets/{virus}/{data_version}/runs/{run_id}/
    # run_id includes config_bundle name: dataset_{config_bundle}_{timestamp}
    output_dir = default_output_dir / 'runs' / args.run_output_subdir
else:
    # Fallback: create a run directory with config bundle name
    # This shouldn't happen if shell script is used correctly
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fallback_run_id = f"dataset_{config_bundle}_{timestamp}"
    output_dir = default_output_dir / 'runs' / fallback_run_id
    print(f"⚠️  Warning: No run_output_subdir provided, using fallback: {fallback_run_id}")
output_dir.mkdir(parents=True, exist_ok=True)

print(f'\nConfig bundle:  {config_bundle}')
print(f'Run suffix:     {RUN_SUFFIX if RUN_SUFFIX else "(none)"}')
print(f'Input file:     {input_file}')
print(f'Output dir:     {output_dir}')
print(f'Run ID:         {args.run_output_subdir if args.run_output_subdir else "auto-generated"}')

# Load protein data
print('\nLoad preprocessed protein sequence data.')
try:
    prot_df = load_dataframe(input_file)
    print(f"Loaded {len(prot_df):,} protein records")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Data file not found at: {input_file}")
except Exception as e:
    raise RuntimeError(f"❌ Error loading data from {input_file}: {e}")

# Enrich with metadata (e.g., host, year, hn_subtype)
print('\nEnrich dataframe with metadata.')
prot_df = enrich_prot_data_with_metadata(prot_df, project_root=project_root)
# breakpoint()
# print(prot_df[prot_df['geo_location_clean'].isna()]['scientific_name'].unique())

# Filter isolates if host/year/hn_subtype/geo_location/passage are specified
host_filter = getattr(config.dataset, 'host', None)
year_filter = getattr(config.dataset, 'year', None)
hn_subtype_filter = getattr(config.dataset, 'hn_subtype', None)
geo_location_filter = getattr(config.dataset, 'geo_location', None)
passage_filter = getattr(config.dataset, 'passage', None)
if (host_filter is not None) or (year_filter is not None) or (hn_subtype_filter is not None) or \
   (geo_location_filter is not None) or (passage_filter is not None):
    print('\nMetadata filtering enabled.')
    print(f'Host filter: {host_filter}')
    print(f'Year filter: {year_filter}')
    print(f'HN subtype filter: {hn_subtype_filter}')
    print(f'Geographic location filter: {geo_location_filter}')
    print(f'Passage filter: {passage_filter}')

    # Filter at isolate level: get isolates matching criteria, then keep all
    # proteins from those isolates. This ensures we don't lose proteins due to
    # metadata merge issues
    meta_cols = ['assembly_id', 'host', 'year', 'hn_subtype']
    if 'geo_location_clean' in prot_df.columns:
        meta_cols.append('geo_location_clean')
    if 'passage' in prot_df.columns:
        meta_cols.append('passage')
    aid_meta = prot_df.groupby('assembly_id')[meta_cols].first().reset_index(drop=True)

    # Debug: print available columns and sample values
    print(f"\n   Available metadata columns: {meta_cols}")
    if 'geo_location_clean' in aid_meta.columns:
        unique_locations = aid_meta['geo_location_clean'].dropna().unique()
        print(f"   Unique geo_location_clean values (first 20): {sorted(unique_locations)[:20]}")
        if geo_location_filter:
            matching_locs = [loc for loc in unique_locations if geo_location_filter.lower() in str(loc).lower()]
            print(f"   Locations matching '{geo_location_filter}' (case-insensitive): {matching_locs[:10]}")
    else:
        print(f"   ⚠️  geo_location_clean column NOT found in prot_df!")
        print(f"   Available columns with 'location' in name: {[c for c in prot_df.columns if 'location' in c.lower()]}")

    # Build filter mask for isolates
    aid_mask = pd.Series([True] * len(aid_meta))
    if host_filter is not None:
        before = aid_mask.sum()
        aid_mask = aid_mask & aid_meta['host'].isin([host_filter])
        after = aid_mask.sum()
        print(f"   Host filter '{host_filter}': {before:,} -> {after:,} isolates")

    if year_filter is not None:
        before = aid_mask.sum()
        aid_mask = aid_mask & aid_meta['year'].isin([year_filter])
        after = aid_mask.sum()
        print(f"   Year filter '{year_filter}': {before:,} -> {after:,} isolates")

    if hn_subtype_filter is not None:
        before = aid_mask.sum()
        aid_mask = aid_mask & aid_meta['hn_subtype'].isin([hn_subtype_filter])
        after = aid_mask.sum()
        print(f"   HN subtype filter '{hn_subtype_filter}': {before:,} -> {after:,} isolates")

    if geo_location_filter is not None:
        before = aid_mask.sum()
        aid_mask = aid_mask & aid_meta['geo_location_clean'].isin([geo_location_filter])
        after = aid_mask.sum()
        print(f"   Geographic location filter '{geo_location_filter}': {before:,} -> {after:,} isolates")

    if passage_filter is not None and 'passage' in aid_meta.columns:
        before = aid_mask.sum()
        aid_mask = aid_mask & aid_meta['passage'].isin([passage_filter])
        after = aid_mask.sum()
        print(f"   Passage filter '{passage_filter}': {before:,} -> {after:,} isolates")

    # Get list of isolates that match criteria
    matching_isolates = aid_meta[aid_mask]['assembly_id'].tolist()

    # Filter protein dataframe to keep only proteins from matching isolates
    n_before = len(prot_df)
    prot_df = prot_df[prot_df['assembly_id'].isin(matching_isolates)].reset_index(drop=True)
    n_after = len(prot_df)

    print(f"   Filtered to {len(matching_isolates):,} isolates matching criteria")
    print(f"   Protein records: {n_before:,} -> {n_after:,} ({100*n_after/n_before:.1f}%)")

# Sample a subset of isolates from the dataframe
sampled_isolates_file = output_dir / 'sampled_isolates.txt'
if MAX_ISOLATES_TO_PROCESS:
    unique_isolates = prot_df['assembly_id'].unique()
    total_isolates = len(unique_isolates)
    print(f"\nSample {MAX_ISOLATES_TO_PROCESS} isolates (out of {total_isolates} total unique isolates).")

    # if sampled_isolates_file.exists():
    #     print(f"Load previously sampled isolates from: {sampled_isolates_file}")
    #     with open(sampled_isolates_file, 'r') as f:
    #         sampled_isolates = [line.strip() for line in f if line.strip()]
    #     print(f"Found {len(sampled_isolates)} sampled isolates in the file.")
    # else:
    if MAX_ISOLATES_TO_PROCESS >= total_isolates:
        sampled_isolates = sorted(unique_isolates)
        print("Requested isolates more than available; using all isolates.")
    else:
        print(f"Sampling {MAX_ISOLATES_TO_PROCESS} isolates (seed: {RANDOM_SEED}).")
        sampled_isolates = np.random.choice(
            unique_isolates,
            size=MAX_ISOLATES_TO_PROCESS,
            replace=False,
        )
        sampled_isolates = sorted(sampled_isolates.tolist())
    sampled_isolates_file.parent.mkdir(parents=True, exist_ok=True)
    with open(sampled_isolates_file, 'w') as f:
        for isolate in sampled_isolates:
            f.write(f"{isolate}\n")
    print(f"Wrote list of {len(sampled_isolates)} sampled isolates to {sampled_isolates_file}")

    original_count = len(prot_df)
    prot_df = prot_df[prot_df['assembly_id'].isin(sampled_isolates)].reset_index(drop=True)
    print(f"Filtered {len(prot_df)} protein records from {original_count} after isolate sampling.")
# else:
#     print("\nDataset isolate sampling disabled (max_isolates_to_process=null).")
#     if sampled_isolates_file.exists():
#         print(f"ℹ️ sampled_isolates.txt found at {sampled_isolates_file} but max_isolates_to_process=null; ignoring file.")

# Restrict to selected functions if specified
# TODO: consider adapting implementation from compute_esm2_embeddings.py
# breakpoint()
if USE_SELECTED_ONLY:
    if hasattr(config.virus, 'selected_functions') and config.virus.selected_functions:
        # Both Flu A and Bunya can use selected_functions approach
        # For Flu A: selected_functions = specific protein functions
        # For Bunya: selected_functions = core protein functions (L, M, S)
        if 'function' not in prot_df.columns:
            raise ValueError("❌ 'function' column not found in protein data")
        print(f"Filtering to selected functions: {config.virus.selected_functions}")
        mask = prot_df['function'].isin(config.virus.selected_functions)
        df = prot_df[mask].reset_index(drop=True)
        print(f"Filtered {len(df)} protein records from {len(prot_df)} based on selected functions.")
    else:
        raise ValueError("use_selected_only is True but no selected_functions defined in config")
else:
    df = prot_df.reset_index(drop=True)

# Add sequence hash for duplicate detection
# TODO Do we actually use 'seq_hash' somewhere?
df['seq_hash'] = df['prot_seq'].apply(lambda x: hashlib.md5(str(x).encode()).hexdigest())

# Validate brc_fea_id uniqueness within isolates
dups = df[df.duplicated(subset=['assembly_id', 'brc_fea_id'], keep=False)]
if not dups.empty:
    raise ValueError(f"❌ Duplicate brc_fea_id found within isolates: \
        {dups[['assembly_id', 'brc_fea_id']]}")

# Settings (using config values)
neg_to_pos_ratio = NEG_TO_POS_RATIO
allow_same_func_negatives = ALLOW_SAME_FUNC_NEGATIVES
max_same_func_ratio = MAX_SAME_FUNC_RATIO

# Split dataset and create pairs (with blocked contradictory negatives)
print('\nSplit dataset and create pairs.')
train_pairs, val_pairs, test_pairs, duplicate_stats = split_dataset(
    df=df,
    neg_to_pos_ratio=neg_to_pos_ratio,
    allow_same_func_negatives=allow_same_func_negatives,
    max_same_func_ratio=max_same_func_ratio,
    hard_partition_isolates=HARD_PARTITION_ISOLATES,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    seed=RANDOM_SEED,
)

# Save datasets
print(f'\nSave datasets: {output_dir}')
train_pairs.to_csv(f"{output_dir}/train_pairs.csv", index=False)
val_pairs.to_csv(f"{output_dir}/val_pairs.csv", index=False)
test_pairs.to_csv(f"{output_dir}/test_pairs.csv", index=False)

# Compute dataset statistics
print('\nComputing dataset statistics...')

# Get unique isolates per split
train_isolates = set(train_pairs['assembly_id_a']).union(set(train_pairs['assembly_id_b']))
val_isolates = set(val_pairs['assembly_id_a']).union(set(val_pairs['assembly_id_b']))
test_isolates = set(test_pairs['assembly_id_a']).union(set(test_pairs['assembly_id_b']))

# Helper function to get metadata distributions
def get_metadata_distributions(df, isolate_set):
    """Get metadata value counts for a set of isolates."""
    if len(isolate_set) == 0:
        return {
            'host': {}, 'year': {}, 'hn_subtype': {},
            'geo_location_clean': {}, 'passage': {}
        }

    # Get one row per isolate (metadata is the same for all proteins from an isolate)
    isolate_meta = df[df['assembly_id'].isin(isolate_set)].groupby('assembly_id').first()
    
    distributions = {}
    for col in ['host', 'year', 'hn_subtype', 'geo_location_clean', 'passage']:
        if col in isolate_meta.columns:
            # Convert value_counts to dict, handling NaN
            counts = isolate_meta[col].value_counts(dropna=False)
            # Convert to JSON-serializable format (NaN -> None, int64 -> int)
            distributions[col] = {
                str(k) if pd.notna(k) else 'null': int(v) 
                for k, v in counts.items()
            }
        else:
            distributions[col] = {}
    
    return distributions

# Build dataset statistics
dataset_stats = {
    'split_sizes': {
        'train': {
            'pairs': len(train_pairs),
            'isolates': len(train_isolates),
            'positive_pairs': int(train_pairs['label'].sum()),
            'negative_pairs': int((train_pairs['label'] == 0).sum()),
            'positive_ratio': float(train_pairs['label'].mean()),
        },
        'val': {
            'pairs': len(val_pairs),
            'isolates': len(val_isolates),
            'positive_pairs': int(val_pairs['label'].sum()),
            'negative_pairs': int((val_pairs['label'] == 0).sum()),
            'positive_ratio': float(val_pairs['label'].mean()),
        },
        'test': {
            'pairs': len(test_pairs),
            'isolates': len(test_isolates),
            'positive_pairs': int(test_pairs['label'].sum()),
            'negative_pairs': int((test_pairs['label'] == 0).sum()),
            'positive_ratio': float(test_pairs['label'].mean()),
        },
    },
    'total': {
        'pairs': len(train_pairs) + len(val_pairs) + len(test_pairs),
        'isolates': len(df['assembly_id'].unique()),
    },
    'metadata_distributions': {
        'train': get_metadata_distributions(df, train_isolates),
        'val': get_metadata_distributions(df, val_isolates),
        'test': get_metadata_distributions(df, test_isolates),
    },
    'co_occurrence_blocking': {
        'total_cooccur_pairs': duplicate_stats['cooccur_stats']['total_cooccur_pairs'],
        'pairs_in_multiple_isolates': duplicate_stats['cooccur_stats']['pairs_in_multiple_isolates'],
        'max_isolates_per_pair': duplicate_stats['cooccur_stats']['max_isolates_per_pair'],
        'train_blocked': duplicate_stats['train_reject_stats']['blocked_cooccur'],
        'val_blocked': duplicate_stats['val_reject_stats']['blocked_cooccur'],
        'test_blocked': duplicate_stats['test_reject_stats']['blocked_cooccur'],
        'total_blocked': (duplicate_stats['train_reject_stats']['blocked_cooccur'] + 
                          duplicate_stats['val_reject_stats']['blocked_cooccur'] + 
                          duplicate_stats['test_reject_stats']['blocked_cooccur']),
    },
    # Filters applied at dataset creation time (for plotting + provenance).
    # NOTE/TODO: Today these are single values (exact-match). If we later support
    # multi-valued filters or ranges (e.g., year: [2010, 2019]), keep storing the
    # raw structure here; plotting code should summarize compactly (e.g., show
    # "host: 7 values" or "year: 2010–2019") rather than dumping long lists.
    'filters_applied': {
        'host': host_filter,
        'year': year_filter,
        'hn_subtype': hn_subtype_filter,
        'geo_location': geo_location_filter,
        'passage': passage_filter,
    },
}

# Save dataset statistics
with open(output_dir / 'dataset_stats.json', 'w') as f:
    json.dump(dataset_stats, f, indent=2)
print(f"Saved dataset stats to: {output_dir / 'dataset_stats.json'}")

# Save per-isolate metadata for downstream analyses/plots (e.g., host×subtype heatmap).
# This avoids trying to re-load and re-enrich the original protein_final.csv.
isolate_metadata_cols = ['assembly_id', 'host', 'hn_subtype', 'year']
if 'geo_location_clean' in df.columns:
    isolate_metadata_cols.append('geo_location_clean')
if 'passage' in df.columns:
    isolate_metadata_cols.append('passage')

try:
    isolate_meta = df.groupby('assembly_id')[isolate_metadata_cols].first().reset_index(drop=True)
    isolate_meta.to_csv(output_dir / 'isolate_metadata.csv', index=False)
    print(f"Saved isolate metadata to: {output_dir / 'isolate_metadata.csv'}")
except Exception as e:
    print(f"⚠️  Warning: failed to save isolate_metadata.csv ({type(e).__name__}: {e})")

# Save duplicate/rejection statistics
cooccur_stats = duplicate_stats['cooccur_stats']
# Remove the large isolate_pair_counts dict for JSON serialization
cooccur_stats_summary = {k: v for k, v in cooccur_stats.items() if k != 'isolate_pair_counts'}

duplicate_summary = {
    'cooccurrence': cooccur_stats_summary,
    'negative_pair_rejections': {
        'train': {k: v for k, v in duplicate_stats['train_reject_stats'].items()},
        'val': {k: v for k, v in duplicate_stats['val_reject_stats'].items()},
        'test': {k: v for k, v in duplicate_stats['test_reject_stats'].items()},
    },
    'pair_key_overlaps': duplicate_stats['pair_key_overlaps'],
}
with open(output_dir / 'duplicate_stats.json', 'w') as f:
    json.dump(duplicate_summary, f, indent=2)
print(f"Saved duplicate stats to: {output_dir / 'duplicate_stats.json'}")

# Save co-occurring pairs list (for analysis)
if cooccur_stats['total_cooccur_pairs'] > 0:
    # Get the top pairs by isolate count for detailed analysis
    isolate_pair_counts = cooccur_stats.get('isolate_pair_counts', {})
    if isolate_pair_counts:
        cooccur_df = pd.DataFrame([
            {'pair_key': k, 'num_isolates': v} 
            for k, v in isolate_pair_counts.items()
        ]).sort_values('num_isolates', ascending=False)
        cooccur_df.to_csv(output_dir / 'cooccurring_sequence_pairs.csv', index=False)
        print(f"Saved {len(cooccur_df)} co-occurring sequence pairs to: cooccurring_sequence_pairs.csv")

# Generate dataset visualization plots (optional)
GENERATE_VISUALIZATIONS = getattr(config.dataset, 'generate_visualizations', True)
if GENERATE_VISUALIZATIONS:
    try:
        print(f'\n📊 Generating dataset visualization plots...')
        from src.analysis.visualize_dataset_stats import visualize_dataset_stats
        
        # Determine output directories
        visualize_dataset_stats(
            dataset_stats_path=output_dir / 'dataset_stats.json',
            bundle_name=config_bundle,
            output_dir_dataset=output_dir,
        )
    except Exception as e:
        print(f"⚠️  Warning: Failed to generate visualizations: {e}")
        print(f"   You can generate them later by running:")
        print(f"   python src/analysis/visualize_dataset_stats.py --bundle {config_bundle}")

print(f'\n✅ Finished {Path(__file__).name}!')
total_timer.display_timer()
