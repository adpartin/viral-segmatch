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

import argparse
import hashlib
import random
import sys
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

total_timer = Timer()


def create_positive_pairs(
    df: pd.DataFrame,
    seed: int = 42,
    ) -> pd.DataFrame:
    """ Create positive protein pairs (within-isolate and cross-function).
    For example, creates pairs within an isolate: RdRp-GPC, RdRp-N, GPC-N
    
    Symmetric pairs handling (e.g., [seq_a, seq_b] and [seq_b, seq_a]).
    Uses combinations (instead of permutations) to create positive pairs to
    avoid duplicates that stem from symmetric pairs.
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
                dct = {
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
    allow_same_func_negatives: bool = True,
    max_same_func_ratio: float = 0.5,
    seed: int = 42,
    ) -> tuple[pd.DataFrame, int]:
    """ Create negative pairs (cross-isolate, with optional control over
    same-function pairs).

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
        allow_same_func_negatives: Whether to allow same-function negative pairs
        max_same_func_ratio: Maximum fraction of same-function negative pairs

    Returns:
        Tuple of (DataFrame of negative pairs, number of same-function negative pairs)
    """
    # np.random.seed(seed)
    # random.seed(seed)

    neg_pairs = []
    seen_pairs = set()  # Track unique pairs

    # Precompute isolate groups to prevent repeated (trial-and-error) sampling.
    # This ensures that 2 different isolates (assembly_id) are sampled during sampling.
    # isolate_groups = {aid: list(grp.itertuples()) for aid, grp in df[df['assembly_id'].isin(isolate_ids)].groupby('assembly_id')} # one-liner
    df_subset = df[df['assembly_id'].isin(isolate_ids)].reset_index(drop=True)
    isolate_groups = {aid: list(grp.itertuples()) for aid, grp in df_subset.groupby('assembly_id')}

    # If allows to generate same-function negative pairs, then compute the max
    # fraction of such pairs out of all negative pairs
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
            'seg_a': row_a.canonical_segment, 'seg_b': row_b.canonical_segment,
            'func_a': row_a.function, 'func_b': row_b.function,
            'seq_hash_a': row_a.seq_hash, 'seq_hash_b': row_b.seq_hash,
            'label': 0  # Negative pair
        }
        neg_pairs.append(dct)
        seen_pairs.add(pair_key)
        if is_same_func:
            same_func_count += 1

    return pd.DataFrame(neg_pairs), same_func_count
        

def compute_isolate_pair_counts(
    df: pd.DataFrame,
    use_core_proteins_only: bool = True,
    verbose: bool = False,
    ) -> dict:
    """ Compute positive pair counts per isolate, validating ≤3 core proteins
    if use_core_proteins_only=True.
    """
    isolate_pos_counts = {} # pos pairs that will originate from this isolate

    for aid, grp in df.groupby('assembly_id'): # isolates
        n_proteins = len(grp)

        # Validate: ≤3 core proteins per isolate (L, M, S) if use_core_proteins_only=True
        if use_core_proteins_only and n_proteins > 3:
            files = grp['file'].unique()
            functions = grp['function'].tolist()
            segments = grp['canonical_segment'].tolist()
            print(f"Warning: assembly_id {aid} has {n_proteins} core proteins, expected ≤3.")
            print(f"Files: {files}")
            print(f"Functions: {functions}")
            print(f"Segments: {segments}")
            raise ValueError(f"assembly_id {aid} has >3 core proteins.")

        if verbose:
            print(f"assembly_id {aid}: {n_proteins} {'core' if use_core_proteins_only else 'total'} proteins, Functions: {grp['function'].tolist()}")

        if n_proteins < 2:
            isolate_pos_counts[aid] = 0
        else:
            # Get all possible protein pairs within the isolate (by definition these are pos pairs)
            pairs = list(combinations(grp.itertuples(), 2))
            # Count all possible protein pairs that have different functions
            pos_count = sum(1 for row_a, row_b in pairs if row_a.function != row_b.function)
            if use_core_proteins_only and pos_count > 3:
                print(f"Error: assembly_id {aid} has {pos_count} positive pairs, expected ≤3.")
                print(f"Proteins: {grp[['brc_fea_id', 'function', 'canonical_segment']]}")
                raise ValueError(f"assembly_id {aid} has >3 positive pairs.")            
            isolate_pos_counts[aid] = pos_count

    return isolate_pos_counts


def split_dataset_v2(
    df: pd.DataFrame,
    neg_to_pos_ratio: float = 3.0,
    allow_same_func_negatives: bool = True,
    max_same_func_ratio: float = 0.5,
    hard_partition_isolates: bool = True,
    hard_partition_duplicates: bool = False,
    use_core_proteins_only: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Split dataset into train, val, and test sets with stratified sampling.

    Returns:
        Tuple of (train_pairs, val_pairs, test_pairs) DataFrames
    """
    # breakpoint()
    # np.random.seed(seed)
    # random.seed(seed)

    # Compute positive pair counts per isolate
    isolate_pos_counts = compute_isolate_pair_counts(df, use_core_proteins_only)

    # Stratify isolates by positive pair counts
    unique_isolates = list(df['assembly_id'].unique())
    # pos_count_groups: {num_pairs: [list_of_isolate_ids]}
    # {1: [list of isolates with 1 pair],
    #  3: [list of isolates with 3 pairs]
    #  5: [list of isolates with 5 pairs]}
    pos_count_groups = {}
    for aid in unique_isolates:
        pos_count = isolate_pos_counts[aid]
        if pos_count not in pos_count_groups:
            pos_count_groups[pos_count] = []
        pos_count_groups[pos_count].append(aid)

    # Split isolates into train/val/test sets (~80/10/10) based on their positive pair counts.
    # Isolates are grouped by the number of positive pairs they contribute (from compute_isolate_pair_counts)
    # to ensure a balanced distribution across sets. Splitting isolates first enforces hard partitioning
    # (when hard_partition_isolates=True), ensuring all proteins and pairs from an isolate stay in one set,
    # preventing data leakage and aligning pair counts with the desired split ratios.
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
        # Ensure no overlap on isolates between val and train
        val_isolates  = [aid for aid in val_isolates if (aid not in train_isolates)]
        # Ensure no overlap on isolates between test and val/train
        test_isolates = [aid for aid in test_isolates if (aid not in train_isolates) and (aid not in val_isolates)]
        # Check for unassigned isolates (should be empty, but included for robustness)
        # An edge case when unassigned could be non-empty: If train_test_split produces empty splits for a small
        # group, rounding might leave isolates unassigned (though our len(isolates) <= 1 check mitigates this).
        unassigned    = [aid for aid in unique_isolates if (aid not in train_isolates) and (aid not in val_isolates) and (aid not in test_isolates)]
        if unassigned:
            print(f"Warning: {len(unassigned)} unassigned isolates.")
            for aid in unassigned:
                # Assign to smallest set to balance sizes
                set_sizes = {'train': len(train_isolates), 'val': len(val_isolates), 'test': len(test_isolates)}
                smallest_set = min(set_sizes, key=set_sizes.get)
                if smallest_set == 'train':
                    train_isolates.append(aid)
                elif smallest_set == 'val':
                    val_isolates.append(aid)
                else:
                    test_isolates.append(aid)

    # Generate positive pairs for each set
    train_pos = create_positive_pairs(df[df['assembly_id'].isin(train_isolates)], seed=seed)
    val_pos   = create_positive_pairs(df[df['assembly_id'].isin(val_isolates)], seed=seed)
    test_pos  = create_positive_pairs(df[df['assembly_id'].isin(test_isolates)], seed=seed)

    # Generate negative pairs within each set (train, val, test)
    train_neg, train_same_func = create_negative_pairs(
        df,
        num_negatives=int(len(train_pos) * neg_to_pos_ratio),
        isolate_ids=train_isolates,
        allow_same_func_negatives=allow_same_func_negatives,
        max_same_func_ratio=max_same_func_ratio, seed=seed
    )
    val_neg, val_same_func = create_negative_pairs(
        df,
        num_negatives=int(len(val_pos) * neg_to_pos_ratio),
        isolate_ids=val_isolates,
        allow_same_func_negatives=allow_same_func_negatives,
        max_same_func_ratio=max_same_func_ratio, seed=seed
    )
    test_neg, test_same_func = create_negative_pairs(
        df,
        num_negatives=int(len(test_pos) * neg_to_pos_ratio),
        isolate_ids=test_isolates,
        allow_same_func_negatives=allow_same_func_negatives,
        max_same_func_ratio=max_same_func_ratio, seed=seed
    )

    # Combine positive and negative pairs
    train_pairs = pd.concat([train_pos, train_neg], ignore_index=True)
    val_pairs   = pd.concat([val_pos, val_neg], ignore_index=True)
    test_pairs  = pd.concat([test_pos, test_neg], ignore_index=True)

    # # Shuffle the combined positive and negative pairs
    # train_pairs = train_pairs.sample(frac=1, random_state=seed).reset_index(drop=True)
    # val_pairs   = val_pairs.sample(frac=1, random_state=seed).reset_index(drop=True)
    # test_pairs  = test_pairs.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Compute and log dataset stats
    total_pairs = len(train_pairs) + len(val_pairs) + len(test_pairs)
    print(f'Total pairs: {total_pairs}')
    print(f'Positive pairs: {len(train_pos) + len(val_pos) + len(test_pos)}')
    print(f'Negative pairs: {len(train_neg) + len(val_neg) + len(test_neg)}')
    # Calculate percentages safely (avoid division by zero)
    train_neg_pct = (train_same_func/len(train_neg)*100) if len(train_neg) > 0 else 0
    val_neg_pct = (val_same_func/len(val_neg)*100) if len(val_neg) > 0 else 0
    test_neg_pct = (test_same_func/len(test_neg)*100) if len(test_neg) > 0 else 0

    print(f'Train same-function negative pairs: {train_same_func} ({train_neg_pct:.2f}%)')
    print(f'Val same-function negative pairs:   {val_same_func} ({val_neg_pct:.2f}%)')
    print(f'Test same-function negative pairs:  {test_same_func} ({test_neg_pct:.2f}%)')
    segment_pair_counts = pd.concat([train_neg, val_neg, test_neg]).groupby(
        ['seg_a', 'seg_b']).size().rename('count').reset_index()
    print(f'Negative pair segment count:\n{segment_pair_counts}\n')

    # # Create histogram for segment_pair_counts
    # plt.figure(figsize=(10, 6))
    # sns.barplot(data=segment_pair_counts, x='seg_a', y='count', hue='seg_b')
    # plt.title('Negative Pair Segment Counts')
    # plt.xlabel('Segment A')
    # plt.ylabel('Count')
    # plt.savefig(output_dir / 'segment_pair_histogram.png')
    # plt.close()

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

    return train_pairs, val_pairs, test_pairs


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
MAX_ISOLATES_TO_PROCESS = getattr(config.dataset, 'max_isolates_to_process', None)

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
    # task_name=TASK_NAME, # TODO: is this necessary?
    run_suffix=RUN_SUFFIX,
    config=config
)

default_input_file = paths['input_file']
default_output_dir = paths['output_dir']

# Apply CLI overrides if provided
input_file = Path(args.input_file) if args.input_file else default_input_file
output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
output_dir.mkdir(parents=True, exist_ok=True)

print(f'\nRun directory: {DATA_VERSION}{RUN_SUFFIX}')
print(f'input_file:    {input_file}')
print(f'output_dir:    {output_dir}')

# Load protein data
print('\nLoad preprocessed protein sequence data.')
try:
    prot_df = load_dataframe(input_file)
    print(f"Loaded {len(prot_df):,} protein records")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Data file not found at: {input_file}")
except Exception as e:
    raise RuntimeError(f"❌ Error loading data from {input_file}: {e}")

# Apply isolate-based sampling (dataset stage only)
sampled_isolates_file = output_dir / 'sampled_isolates.txt'
if MAX_ISOLATES_TO_PROCESS:
    unique_isolates = prot_df['assembly_id'].unique()
    total_isolates = len(unique_isolates)
    print(f"\nDataset sampling enabled: sample {MAX_ISOLATES_TO_PROCESS} isolates (out of {total_isolates} total unique isolates).")
    
    if sampled_isolates_file.exists():
        print(f"Load previously sampled isolates from: {sampled_isolates_file}")
        with open(sampled_isolates_file, 'r') as f:
            sampled_isolates = [line.strip() for line in f if line.strip()]
        print(f"Found {len(sampled_isolates)} sampled isolates in the file.")
    else:
        if MAX_ISOLATES_TO_PROCESS >= total_isolates:
            sampled_isolates = sorted(unique_isolates)
            print("Requested isolates more than available; using all isolates (writing the isolates into the file for consistency).")
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
else:
    print("\nDataset isolate sampling disabled (max_isolates_to_process=null).")
    if sampled_isolates_file.exists():
        print(f"ℹ️ sampled_isolates.txt found at {sampled_isolates_file} but max_isolates_to_process=null; ignoring file.")

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

# Note: Protein validation is now handled by the unified selected_functions approach
# No need for validate_core_proteins() which was specific to the old core proteins method

# Validate brc_fea_id uniqueness within isolates
dups = df[df.duplicated(subset=['assembly_id', 'brc_fea_id'], keep=False)]
if not dups.empty:
    raise ValueError(f"❌ Duplicate brc_fea_id found within isolates: \
        {dups[['assembly_id', 'brc_fea_id']]}")

# Settings (using config values)
neg_to_pos_ratio = NEG_TO_POS_RATIO
allow_same_func_negatives = ALLOW_SAME_FUNC_NEGATIVES
max_same_func_ratio = MAX_SAME_FUNC_RATIO
hard_partition_isolates = True  # TODO: Move to config in future
hard_partition_duplicates = False  # TODO: Move to config in future

# Split dataset and create pairs
print('\nSplit dataset and create pairs.')
train_pairs, val_pairs, test_pairs = split_dataset_v2(
    df=df,
    neg_to_pos_ratio=neg_to_pos_ratio,
    allow_same_func_negatives=allow_same_func_negatives,
    max_same_func_ratio=max_same_func_ratio,
    hard_partition_isolates=hard_partition_isolates,
    hard_partition_duplicates=hard_partition_duplicates,
    use_core_proteins_only=USE_SELECTED_ONLY,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    seed=RANDOM_SEED,
)

# Save datasets
print(f'\nSave datasets: {output_dir}')
train_pairs.to_csv(f"{output_dir}/train_pairs.csv", index=False)
val_pairs.to_csv(f"{output_dir}/val_pairs.csv", index=False)
test_pairs.to_csv(f"{output_dir}/test_pairs.csv", index=False)

print(f'\n✅ Finished {Path(__file__).name}!')
total_timer.display_timer()
