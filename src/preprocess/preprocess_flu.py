"""
Preprocess protein and genome data from GTO files for Flu A.

Unified script that parses each GTO file once and extracts both:
- Protein data -> protein_final.csv (identical to preprocess_flu_protein.py output)
- Genome data  -> genome_final.csv (new)

See docs/genome_pipeline_design.md for design decisions.

Example:
```bash
python src/preprocess/preprocess_flu.py --config_bundle flu_debug --max_files_to_preprocess 10
or
python src/preprocess/preprocess_flu.py --config_bundle flu
or
run using ./scripts/stage1_preprocess_flu.sh
```
"""
import argparse
import json
import random
import sys
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from typing import Optional

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.timer_utils import Timer
from src.utils.config_hydra import get_virus_config_hydra, print_config_summary
from src.utils.seed_utils import resolve_process_seed
from src.utils.path_utils import resolve_run_suffix, build_preprocessing_paths, load_dataframe
from src.utils.experiment_utils import save_experiment_metadata, save_experiment_summary
from src.utils.gto_utils import (
    extract_assembly_info,
    handle_assembly_duplicates,
)
from src.utils.protein_utils import (
    analyze_protein_ambiguities,
    summarize_ambiguities,
    prepare_sequences_for_esm2,
    print_replicon_func_count
)
from src.utils.dna_utils import summarize_dna_qc

# Manual configs
PROT_SEQ_COL_NAME = 'prot_seq'
DNA_SEQ_COL_NAME = 'dna_seq'

total_timer = Timer()


# =============================================================================
# Combined GTO extraction
# =============================================================================

def extract_data_from_gto(gto_file_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract both protein and genome data from a single GTO file.

    Opens the GTO JSON once, extracts shared metadata, then builds:
    - protein_df: one row per protein feature (same as get_protein_data_from_gto)
    - genome_df: one row per contig (genomic segment)

    Returns:
        (protein_df, genome_df)
    """
    with open(gto_file_path, 'r') as file:
        gto = json.load(file)

    # --- Shared metadata ---
    assembly_prefix, assembly_id = extract_assembly_info(gto_file_path.name)
    meta = {
        'file': gto_file_path.name,
        'assembly_prefix': assembly_prefix,
        'assembly_id': assembly_id,
        'ncbi_taxonomy_id': gto['ncbi_taxonomy_id'],
        'genetic_code': gto['genetic_code'],
        'scientific_name': gto['scientific_name'],
        'quality': gto['quality']['genome_quality'],
    }
    meta_df = pd.DataFrame([meta])

    # --- Shared segment map (genbank_ctg_id -> replicon_type) ---
    segment_map = {
        contig['id']: contig.get('replicon_type', 'Unassigned')
        for contig in gto.get('contigs', [])
    }

    # --- Protein extraction (from features) ---
    fea_cols = ['id', 'type', 'function', 'protein_translation', 'location', 'feature_quality']
    prot_rows = []
    for fea_dict in gto['features']:
        row = {k: fea_dict.get(k, None) for k in fea_cols}
        row['length'] = len(fea_dict.get('protein_translation', '')) if 'protein_translation' in fea_dict else 0
        location = fea_dict.get('location')
        genbank_ctg_id = (
            location[0][0]
            if (isinstance(location, list)
                and len(location) > 0
                and len(location[0]) > 0)
            else None
        )
        row['genbank_ctg_id'] = genbank_ctg_id
        row['replicon_type'] = segment_map.get(genbank_ctg_id, 'Unassigned')
        family_assignments = fea_dict.get('family_assignments')
        row['family'] = (
            family_assignments[0][1]
            if (isinstance(family_assignments, list)
                and len(family_assignments) > 0
                and len(family_assignments[0]) > 1)
            else pd.NA
        )
        prot_rows.append(row)

    protein_df = pd.DataFrame(prot_rows)
    protein_df = pd.merge(meta_df, protein_df, how='cross')
    protein_df.rename(columns={'protein_translation': 'prot_seq', 'id': 'brc_fea_id'}, inplace=True)

    # --- Genome extraction (from contigs) ---
    genome_rows = []
    for contig in gto.get('contigs', []):
        genome_rows.append({
            'genbank_ctg_id': contig.get('id'),
            'replicon_type': contig.get('replicon_type', 'Unassigned'),
            'contig_quality': contig.get('contig_quality'),
            'dna_seq': contig.get('dna'),
        })

    genome_df = pd.DataFrame(genome_rows)
    genome_df = pd.merge(meta_df[['assembly_id', 'file', 'quality']], genome_df, how='cross')

    return protein_df, genome_df


def aggregate_data_from_gto_files(
    gto_dir: Path,
    max_files: Optional[int] = None,
    random_seed: Optional[int] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate protein and genome data from GTO files.

    Args:
        gto_dir: Directory containing GTO files
        max_files: Maximum number of files to process (None for all)
        random_seed: Random seed for reproducible sampling

    Returns:
        (protein_df, genome_df)
    """
    prot_dfs = []
    genome_dfs = []
    gto_files = sorted(gto_dir.glob('*.gto'))

    # Apply subset sampling if requested
    if max_files is not None and max_files < len(gto_files):
        total_files = len(sorted(gto_dir.glob('*.gto')))
        if random_seed is not None:
            random.seed(random_seed)
            gto_files = random.sample(gto_files, max_files)
            print(f"Processing subset: {len(gto_files)} files (randomly sampled from {total_files} total files)")
        else:
            gto_files = gto_files[:max_files]
            print(f"Processing subset: first {len(gto_files)} files (from the {total_files} files)")
    else:
        print(f"Processing all {len(gto_files)} files")

    for fpath in tqdm(gto_files, desc='Extracting protein + genome data from GTO files'):
        prot, genome = extract_data_from_gto(fpath)
        prot_dfs.append(prot)
        genome_dfs.append(genome)

    protein_df = pd.concat(prot_dfs, axis=0).reset_index(drop=True)
    genome_df = pd.concat(genome_dfs, axis=0).reset_index(drop=True)
    return protein_df, genome_df


# =============================================================================
# Protein pipeline functions (copied from preprocess_flu_protein.py)
# =============================================================================

def validate_protein_counts(
    df: pd.DataFrame,
    core_only: bool = False,
    verbose: bool = False
    ) -> None:
    """Validate protein counts per assembly to ensure biological consistency.
    Ensure each assembly_id has ≤max_core_proteins core proteins if core_only=True.
    TODO: consider moving this somewhere. Maybe need to create a new utils file?

    For Flu A, each assembly cannot have have more than 9 core proteins.
    For Bunya, each assembly cannot have have more than 3 core proteins.

    Args:
        df: DataFrame with protein data containing 'assembly_id' and 'function' columns
        core_only: If True, only validate core proteins; if False, validate all proteins
        verbose: If True, print detailed validation information

    Raises:
        ValueError: If any assembly has more than the expected number of core proteins
    """
    if core_only:
        df = df[df['function'].isin(core_functions)]
        max_expected = len(core_functions)
    else:
        max_expected = None  # No limit for non-core validation

    for aid, grp in df.groupby('assembly_id'):
        n_proteins = len(grp)
        if core_only and n_proteins > max_expected:
            raise ValueError(
                f"Error: assembly_id {aid} has {n_proteins} core proteins, expected ≤{max_expected}.\n"
                f"   Files: {grp['file'].unique()}\n"
                f"   Functions: {grp['function'].tolist()}"
            )
        elif verbose:
            print(f"assembly_id {aid}: {n_proteins} {'core' if core_only else 'total'} "
                  f"proteins, Functions: {grp['function'].tolist()}")

    return True


def analyze_protein_counts_per_file(
    df: pd.DataFrame,
    core_functions: list[str],
    aux_functions: list[str],
    selected_functions: list[str]
    ) -> None:
    """ Analyze protein counts per GTO file.

    Args:
        df: DataFrame with protein data
        core_functions: List of core protein functions
        aux_functions: List of auxiliary protein functions
        selected_functions: List of selected protein functions
    """
    print(f"\n{'='*50}")
    print("PROTEIN COUNTS PER GTO FILE")
    print('='*50)

    # Count proteins per file
    file_counts = df.groupby('file').agg({
        'function': 'count',  # Total proteins
        'brc_fea_id': 'nunique'  # Unique proteins (should be same as count)
    }).rename(columns={'function': 'total_proteins', 'brc_fea_id': 'unique_proteins'})

    # Count core proteins per file
    core_mask = df['function'].isin(core_functions)
    core_counts = df[core_mask].groupby('file')['function'].count().rename('core_proteins')

    # Count auxiliary proteins per file
    aux_mask = df['function'].isin(aux_functions)
    aux_counts = df[aux_mask].groupby('file')['function'].count().rename('aux_proteins')

    # Combine all counts
    all_counts = file_counts.join(core_counts, how='left').join(aux_counts, how='left')
    all_counts = all_counts.fillna(0).astype(int)

    # Add other proteins (neither core nor aux)
    all_counts['other_proteins'] = all_counts['total_proteins'] - all_counts['core_proteins'] - all_counts['aux_proteins']

    # Identify "other" proteins (not in core or aux lists)
    all_known_functions = set(core_functions + aux_functions)
    other_proteins = df[~df['function'].isin(all_known_functions)]['function'].unique()

    # Summary statistics
    print(f"Total GTO files analyzed: {len(all_counts)}")

    # Show unidentified proteins
    if len(other_proteins) > 0:
        print(f"\nUNIDENTIFIED PROTEINS (not in core_functions or aux_functions):")
        print(f"Found {len(other_proteins)} unique protein functions not in config:")
        for i, protein in enumerate(sorted(other_proteins), 1):
            count = df[df['function'] == protein].shape[0]
            print(f"  {i:2d}. {protein} (appears in {count} records)")
    else:
        print(f"\nAll proteins are classified (no unidentified proteins found)")

    # Show distribution of core protein counts
    print(f"\nCore protein count distribution:")
    core_dist = all_counts['core_proteins'].value_counts().sort_index()
    for count, freq in core_dist.items():
        print(f"  {count} core proteins: {freq} files")

    # Show files with unusual protein counts (less than known core proteins)
    expected_core = len(core_functions)  # Should be 9 for Flu A
    low_core = all_counts[all_counts['core_proteins'] < expected_core]
    if len(low_core) > 0:
        print(f"\nFound {len(low_core)} files with < {expected_core} core proteins (expected for Flu A):")
        print(low_core[['total_proteins', 'core_proteins', 'aux_proteins', 'other_proteins']].head(7))
    else:
        print(f"All files have >= {expected_core} core proteins")

    # Show files with more than known core proteins (potential duplicates)
    high_core = all_counts[all_counts['core_proteins'] > expected_core]
    if len(high_core) > 0:
        print(f"\nFound {len(high_core)} files with > {expected_core} core proteins:")
        print(high_core[['total_proteins', 'core_proteins', 'aux_proteins', 'other_proteins']].head(7))
    else:
        print(f"No files have > {expected_core} core proteins")

    # Analyze intra-file protein (function) duplicates in selected proteins across all files
    print(f"\nANALYZING INTRA-FILE DUPLICATES IN 'SELECTED' (PROTEIN) FUNCTIONS ACROSS ALL FILES:")
    print(f"   This detects FUNCTION-based duplicates (same protein function in same file)")
    print(f"   Sequence-based duplicates are detected later in handle_protein_duplicates()\n")
    analyze_intra_file_function_duplicates(df, selected_functions, output_dir)

    return all_counts


def analyze_intra_file_function_duplicates(
    df: pd.DataFrame,
    subset_functions: Optional[list[str]] = None,
    output_dir: Optional[Path] = None
    ) -> None:
    """Analyze intra-file duplicates in protein FUNCTIONS (not sequences) across all files.

    This detects when the same protein function appears multiple times in the same file.
    This is DIFFERENT from sequence-based duplicates.

    Args:
        df: Full protein dataframe
        subset_functions: List of protein functions to analyze. If None, analyzes ALL functions.
        output_dir: Directory to save problematic proteins file. If None, uses current directory.
    """
    # Set default output directory
    if output_dir is None:
        output_dir = Path(".")
        print(f"No output_dir provided, using current directory: {output_dir.absolute()}")

    # Determine which functions to analyze
    if subset_functions is None:
        funcs = df['function'].unique().tolist()
        print(f"Analyzing ALL {len(funcs)} protein functions for intra-file duplicates...")
    else:
        funcs = subset_functions
        print(f"Analyzing {len(funcs)} selected protein functions for intra-file duplicates...")

    # Filter to specified proteins only (keep original df unchanged)
    sub_df = df[df['function'].isin(funcs)]

    print(f"Files to analyze: {sub_df['file'].nunique()}")
    print(f"Protein records: {len(sub_df)}")

    # Count protein occurrences per file
    counts_per_file = sub_df.groupby(['file', 'function']).size().reset_index(name='count')

    # Find proteins that appear multiple times in the same file
    duplicates = counts_per_file[counts_per_file['count'] > 1]

    if len(duplicates) > 0:
        print(f"\nFound {len(duplicates)} protein duplicates:")
        print('='*50)

        # Prepare data for saving to file
        intra_file_dups = []

        for _, row in duplicates.iterrows():
            file_name = row['file']
            function = row['function']
            count = row['count']

            # Get the actual records for this duplicate
            duplicate_records = sub_df[
                (sub_df['file'] == file_name) &
                (sub_df['function'] == function)
            ]

            print(f"\nFile: {file_name}")
            print(f"   Protein function: {function}")
            print(f"   Count: {count} occurrences")
            print(f"   Details:")

            for i, (_, record) in enumerate(duplicate_records.iterrows(), 1):
                print(f"     {i}. Assembly ID: {record['assembly_id']}")
                print(f"         Replicon: {record['replicon_type']}")
                print(f"         BRC Feature ID: {record['brc_fea_id']}")
                if 'prot_seq' in record:
                    seq_preview = record['prot_seq'][:35] + "..." if len(record['prot_seq']) > 35 else record['prot_seq']
                    print(f"         Seq: {seq_preview}")

                # Collect for saving
                intra_file_dups.append({
                    'file': file_name,
                    'function': function,
                    'duplicate_count': count,
                    'assembly_id': record['assembly_id'],
                    'replicon_type': record['replicon_type'],
                    'brc_fea_id': record['brc_fea_id'],
                    'seq_preview': seq_preview if 'prot_seq' in record else None
                })

        # Save intra-file duplicates to file
        if intra_file_dups:
            intra_file_dups_df = pd.DataFrame(intra_file_dups)
            output_file = output_dir / 'intra_file_protein_duplicates.csv'
            intra_file_dups_df.to_csv(output_file, index=False)
            print(f"\nSaved {len(intra_file_dups)} intra-file protein duplicates to: {output_file}")
    else:
        print("No protein duplicates found")

    # Summary statistics
    print(f"\nINTRA-FILE PROTEIN DUPLICATE SUMMARY (considers 'selected_functions' only):")
    print(f"   Files analyzed: {sub_df['file'].nunique()}")
    print(f"   Protein records across files: {len(sub_df)}")
    print(f"   Unique protein functions across files: {sub_df['function'].nunique()}")
    print(f"   Intra-file duplicates on 'function' column: {len(duplicates)}")


def assign_segment_using_core_proteins(
    prot_df: pd.DataFrame,
    ) -> pd.DataFrame:
    """Assign canonical_segment to proteins using core protein mappings.

    Uses conditional_segment_mappings.core_proteins from the virus config,
    which maps (function, replicon_type) pairs to canonical segments.
    A canonical segment is only assigned when both the protein function and
    the replicon_type match known biology — if a GTO record has an
    unexpected combination (e.g., PB2 on Segment 3), the protein is left
    unassigned. See conf/virus/flu.yaml for the full mapping table.
    """
    prot_df = prot_df.copy()

    # Get conditional mappings from config
    core_mappings = config.virus.conditional_segment_mappings.core_proteins
    core_map = pd.DataFrame(core_mappings)

    # Merge (assign temp segments where function-replicon pairs match)
    # how='left': keeps all rows, assigns NaN where mapping doesn't apply
    prot_df = prot_df.merge(core_map, on=['function', 'replicon_type'], how='left')

    # Assign core_segment to canonical_segment
    mask = prot_df['core_segment'].notna()
    prot_df.loc[mask, 'canonical_segment'] = prot_df.loc[mask, 'core_segment']

    # Diagnostics
    print_replicon_func_count(prot_df, functions=core_functions)
    df = print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)

    prot_df = prot_df.drop(columns=['core_segment'])
    return prot_df


def assign_segment_using_aux_proteins(
    prot_df: pd.DataFrame,
    ) -> pd.DataFrame:
    """Assign canonical_segment to proteins using auxiliary protein mappings.

    Uses conditional_segment_mappings.aux_proteins from the virus config.
    Same validation logic as core: only assigns when (function, replicon_type)
    matches known biology. Only fills in proteins not already mapped by
    assign_segment_using_core_proteins (i.e., canonical_segment is still NaN).
    See conf/virus/flu.yaml for the full mapping table.
    """
    prot_df = prot_df.copy()

    # Get conditional mappings from config
    aux_mappings = config.virus.conditional_segment_mappings.aux_proteins
    aux_map = pd.DataFrame(aux_mappings)

    # Merge (assign temp canonical segments where function-replicon pairs match)
    # how='left': keeps all rows, assigns NaN where mapping doesn't apply
    prot_df = prot_df.merge(aux_map, on=['function', 'replicon_type'], how='left')

    # Assign aux_segment to canonical_segment (only for unassigned proteins)
    mask = prot_df['canonical_segment'].isna() & prot_df['aux_segment'].notna()
    prot_df.loc[mask, 'canonical_segment'] = prot_df.loc[mask, 'aux_segment']

    # Diagnostics
    print_replicon_func_count(prot_df, functions=aux_functions)
    df = print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)

    prot_df = prot_df.drop(columns=['aux_segment'])
    return prot_df


def assign_protein_segments(
    prot_df: pd.DataFrame,
    use_core: bool = True,
    use_aux: bool = True
    ) -> pd.DataFrame:
    """Assign canonical_segment to each protein row in prot_df.

    Creates the 'canonical_segment' column (e.g., 'S1'...'S8') by matching
    each protein's (function, replicon_type) pair against known biology from
    the virus config. A segment is only assigned when both fields agree with
    the expected mapping — records with inconsistent combinations are left
    unassigned. Core proteins are mapped first, then auxiliary proteins fill
    in any remaining unassigned rows.

    This is the protein-side counterpart to assign_genome_segments(), which
    uses the simpler replicon_type-only mapping (genomes have one contig per
    segment, so no function-based validation is needed).

    See conf/virus/flu.yaml conditional_segment_mappings for the full tables.

    Args:
        prot_df: DataFrame with protein data (must have 'function' and
            'replicon_type' columns)
        use_core: If True, assign segments using core protein mappings
        use_aux: If True, assign segments using auxiliary protein mappings

    Returns:
        prot_df with 'canonical_segment' column added (NaN for unassigned)
    """
    print(f"\n{'='*50}")
    print(f'Assign canonical segments using core and/or auxiliary proteins.')
    print('='*50)

    prot_df = prot_df.copy()
    prot_df['canonical_segment'] = pd.NA  # Placeholder for canonical_segment

    if use_core:
        prot_df = assign_segment_using_core_proteins(prot_df)

    if use_aux:
        prot_df = assign_segment_using_aux_proteins(prot_df)

    # Diagnostics
    print("\nCore and auxiliary protein segment mappings:")
    all_functions = config.virus.core_functions + config.virus.aux_functions
    df = print_replicon_func_count(
        prot_df,
        functions=all_functions,
        more_cols=['canonical_segment'],
        drop_na=False)
    df.to_csv(output_dir / 'seg_mapped_core_and_aux_eda.csv', sep=',', index=False)

    unmapped = prot_df[prot_df['canonical_segment'].isna()]
    print(f"\nUnmapped segments: {len(unmapped)}")
    print_replicon_func_count(unmapped)

    return prot_df


def apply_protein_basic_filters(prot_df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic filters to protein data."""
    print(f"\n{'='*50}")
    print(f'Apply basic filters to protein data.')
    print('='*50)

    # Drop unassigned canonical segments
    unassigned_df = prot_df[prot_df['canonical_segment'].isna()]
    prot_df = prot_df[prot_df['canonical_segment'].notna()].reset_index(drop=True)
    print(f'\nDropped unassigned canonical_segment: {unassigned_df.shape}')
    print(f'Remaining prot_df:                    {prot_df.shape}')
    unassigned_df.to_csv(output_dir / 'protein_unassigned_segments.csv', sep=',', index=False)

    # Keep CDS only
    non_cds = prot_df[prot_df['type'] != 'CDS']
    prot_df = prot_df[prot_df['type'] == 'CDS'].reset_index(drop=True)
    print(f'\nDropped non-CDS samples: {non_cds.shape}')
    print(f'Remaining prot_df:       {prot_df.shape}')
    non_cds.to_csv(output_dir / 'protein_non_cds.csv', sep=',', index=False)

    # Drop poor quality (genome-level quality)
    # TODO: Also consider filtering by feature_quality (per-protein quality from GTO)
    #       which may catch poor individual proteins in otherwise good assemblies.
    poor_df = prot_df[prot_df['quality'] == 'Poor']
    prot_df = prot_df[prot_df['quality'] != 'Poor'].reset_index(drop=True)
    print(f'\nDropped Poor quality samples: {poor_df.shape}')
    print(f'Remaining prot_df:            {prot_df.shape}')
    poor_df.to_csv(output_dir / 'protein_poor_quality.csv', sep=',', index=False)

    # Drop unassigned replicons
    unk_df = prot_df[prot_df['replicon_type'] == 'Unassigned']
    prot_df = prot_df[prot_df['replicon_type'] != 'Unassigned'].reset_index(drop=True)
    print(f'\nDropped unassigned replicons: {unk_df.shape}')
    print(f'Remaining prot_df:            {prot_df.shape}')
    unk_df.to_csv(output_dir / 'protein_unassigned_replicons.csv', sep=',', index=False)

    return prot_df


def handle_protein_duplicates(
    prot_df: pd.DataFrame,
    print_eda: bool = False
    ) -> pd.DataFrame:
    """Handle protein sequence duplicates.

    NOTE: Sequence-based intra-file duplicates are handled by handle_assembly_duplicates()
    since file -> assembly_id is 1:1 mapping, any ['prot_seq', 'file'] duplicate is
    automatically a ['prot_seq', 'assembly_id'] duplicate.
    """
    print(f"\n{'='*50}")
    print("Handle protein sequence duplicates.")
    print('='*50)

    # EDA start ------------------------------------------------
    if print_eda:
        aa = print_replicon_func_count(prot_df,
            more_cols=['canonical_segment'], drop_na=False)

        # Get all duplicate protein sequences
        print(f'\nprot_df: {prot_df.shape}')
        prot_dups = (
            prot_df[prot_df.duplicated(subset=[PROT_SEQ_COL_NAME], keep=False)]
            .sort_values(PROT_SEQ_COL_NAME)
            .reset_index(drop=True).copy()
        )
        print(f"Duplicates on '{PROT_SEQ_COL_NAME}': {prot_dups.shape}")
        print(prot_dups[:4][['file', 'assembly_id', 'genbank_ctg_id',
            'replicon_type', 'brc_fea_id', 'function', 'canonical_segment']])

        print(f'\nCount how many unique files contain each duplicated sequence:')
        dup_counts = prot_dups.groupby(PROT_SEQ_COL_NAME).agg(num_files=('file', 'nunique')).reset_index()
        print(dup_counts['num_files'].value_counts().reset_index(name='total_cases'))

        def explore_groupby(df, columns):
            df = df.copy()
            grouped = df.groupby(columns)

            for keys, grp in grouped:
                print('\nDuplicate group (rows with this seq):')
                print(grp[:4][[PROT_SEQ_COL_NAME, 'file', 'assembly_id', 'genbank_ctg_id',
                    'replicon_type', 'brc_fea_id', 'function']])
                break  # remove this to loop over more groups
            return grp

        explore_groupby(prot_dups, PROT_SEQ_COL_NAME)

        # Expand dup_counts to include more info
        dup_stats = (
            prot_dups.groupby(PROT_SEQ_COL_NAME).agg(
                num_occurrences=(PROT_SEQ_COL_NAME, 'count'),
                num_files=('file', 'nunique'),
                num_functions=('function', 'nunique'),
                num_replicons=('replicon_type', 'nunique'),
                functions=('function', lambda x: list(set(x))),
                replicons=('replicon_type', lambda x: list(set(x))),
                brc_fea_ids=('brc_fea_id', lambda x: list(set(x))),
                files=('file', lambda x: list(set(x))),
                assembly_ids=('assembly_id', lambda x: list(set(x))),
                num_assemblies=('assembly_id', 'nunique'),
                assembly_prefixes=('assembly_prefix', lambda x: list(set(x))),
                num_assembly_prefixes=('assembly_prefix', 'nunique'),
            )
            .sort_values('num_occurrences', ascending=False)
            .reset_index()
            .copy()
        )
    # EDA end ------------------------------------------------

    # Handle assembly duplicates (replaces enforce_single_file() for Flu A)
    print("\nHandle assembly duplicates.")
    print(f'prot_df before: {prot_df.shape}')
    prot_df, dup_summary = handle_assembly_duplicates(prot_df)
    print(f'prot_df after:  {prot_df.shape}')

    # Save duplicate summary for analysis
    if not dup_summary.empty:
        dup_summary.to_csv(output_dir / 'duplicate_sequences_removed.csv', index=False)
        print(f"Saved {len(dup_summary)} duplicate records to: duplicate_sequences_removed.csv")

    # Validate protein counts
    print("\nValidate protein counts.")
    validate_protein_counts(prot_df, core_only=True)

    return prot_df


def analyze_sequence_duplicates_for_pair_classification(
    prot_df: pd.DataFrame,
    output_dir: Optional[Path] = None
    ) -> None:
    """Analyze sequence duplicates relevant to pair classification problem.

    This function identifies:
    1. Sequences that appear in multiple genomes/isolates (potential for contradictory labels)
    2. Sequence pairs that could appear with both positive and negative labels
    3. Statistics on how widespread sequence duplication is across genomes

    This is critical for understanding data leakage risks: if a sequence pair (a, b) appears
    in training as positive (same isolate) and in test as negative (different isolates),
    the model could memorize the pair, causing data leakage.

    Args:
        prot_df: DataFrame with protein data (must have 'prot_seq', 'assembly_id', 'function')
        output_dir: Directory to save analysis results. If None, uses current directory.
    """
    if output_dir is None:
        output_dir = Path(".")

    print(f"\n{'='*50}")
    print("Analyze Sequence Duplicates for Pair Classification")
    print('='*50)

    # Compute sequence hash for deduplication
    import hashlib
    prot_df = prot_df.copy()
    if 'seq_hash' not in prot_df.columns:
        prot_df['seq_hash'] = prot_df['prot_seq'].apply(
            lambda x: hashlib.md5(str(x).encode()).hexdigest()
        )

    # 1. Analyze individual sequence duplication across genomes
    print("\n1. Individual Sequence Duplication Analysis:")
    print("-" * 50)

    seq_dup_stats = (
        prot_df.groupby('prot_seq').agg(
            num_occurrences=('prot_seq', 'count'),
            num_isolates=('assembly_id', 'nunique'),
            num_functions=('function', 'nunique'),
            isolates=('assembly_id', lambda x: sorted(list(set(x)))),
            functions=('function', lambda x: sorted(list(set(x)))),
            segment=('canonical_segment', lambda x: sorted(list(set(x))) if pd.notna(x).any() else []),
        )
        .sort_values('num_isolates', ascending=False)
        .reset_index()
    )

    duplicated_seqs = seq_dup_stats[seq_dup_stats['num_isolates'] > 1].copy()

    print(f"Total unique sequences: {len(seq_dup_stats)}")
    print(f"Sequences appearing in >1 isolate: {len(duplicated_seqs)} ({len(duplicated_seqs)/len(seq_dup_stats)*100:.2f}%)")

    if len(duplicated_seqs) > 0:
        print(f"\nDistribution of sequences by number of isolates:")
        isolate_dist = duplicated_seqs['num_isolates'].value_counts().sort_index()
        for num_isolates, count in isolate_dist.items():
            print(f"  {num_isolates} isolates: {count} sequences")

        # Show top duplicated sequences
        print(f"\nTop 10 sequences appearing in most isolates:")
        top_dups = duplicated_seqs.head(10)
        for idx, row in top_dups.iterrows():
            seq_preview = row['prot_seq'][:50] + "..." if len(row['prot_seq']) > 50 else row['prot_seq']
            print(f"  {row['num_isolates']} isolates, {row['num_functions']} functions: {seq_preview}")
            print(f"    Functions: {', '.join(row['functions'][:3])}{'...' if len(row['functions']) > 3 else ''}")

    # 2. Analyze potential contradictory sequence pairs
    print("\n2. Potential Contradictory Sequence Pair Analysis:")
    print("-" * 50)

    # Create a mapping of sequence to isolates
    seq_to_isolates = {}
    for _, row in prot_df.iterrows():
        seq = row['prot_seq']
        isolate = row['assembly_id']
        if seq not in seq_to_isolates:
            seq_to_isolates[seq] = set()
        seq_to_isolates[seq].add(isolate)

    # Find all unique sequence pairs within isolates (potential positive pairs)
    print("Computing potential positive pairs (same isolate)...")
    positive_candidates = []
    for isolate, grp in prot_df.groupby('assembly_id'):
        seqs = grp['prot_seq'].unique().tolist()
        if len(seqs) >= 2:
            from itertools import combinations
            for seq_a, seq_b in combinations(seqs, 2):
                # Normalize pair order
                pair_key = tuple(sorted([seq_a, seq_b]))
                positive_candidates.append({
                    'seq_a': pair_key[0],
                    'seq_b': pair_key[1],
                    'isolate': isolate
                })

    pos_pairs_df = pd.DataFrame(positive_candidates)
    if len(pos_pairs_df) > 0:
        pos_pairs_df = pos_pairs_df.drop_duplicates(subset=['seq_a', 'seq_b'])
        print(f"  Unique sequence pairs within isolates: {len(pos_pairs_df)}")

    # Find sequence pairs that could be negative (different isolates)
    print("Identifying pairs that could appear as both positive and negative...")
    contradictory_pairs = []

    if len(pos_pairs_df) > 0:
        for _, row in pos_pairs_df.iterrows():
            seq_a = row['seq_a']
            seq_b = row['seq_b']

            # Get isolates for each sequence
            isolates_a = seq_to_isolates.get(seq_a, set())
            isolates_b = seq_to_isolates.get(seq_b, set())

            # Check if this pair could appear in different isolates (negative pairs)
            common_isolates = isolates_a & isolates_b
            different_isolates = (isolates_a | isolates_b) - common_isolates

            if len(common_isolates) > 0 and len(different_isolates) > 0:
                contradictory_pairs.append({
                    'seq_a': seq_a,
                    'seq_b': seq_b,
                    'num_positive_occurrences': len(common_isolates),
                    'num_negative_occurrences': len(different_isolates),
                    'total_isolates_a': len(isolates_a),
                    'total_isolates_b': len(isolates_b),
                    'common_isolates': sorted(list(common_isolates)),
                    'different_isolates_count': len(different_isolates)
                })

    contradictory_df = pd.DataFrame(contradictory_pairs)

    if len(contradictory_df) > 0:
        print(f"\n  Found {len(contradictory_df)} sequence pairs that could appear with BOTH positive and negative labels!")
        print(f"  This represents {len(contradictory_df)/len(pos_pairs_df)*100:.2f}% of all unique positive pairs")

        print(f"\n  Distribution of contradictory pairs by number of positive occurrences:")
        pos_dist = contradictory_df['num_positive_occurrences'].value_counts().sort_index()
        for num_pos, count in pos_dist.items():
            print(f"    {num_pos} positive occurrence(s): {count} pairs")

        print(f"\n  Distribution of contradictory pairs by number of negative possibilities:")
        neg_dist = contradictory_df['num_negative_occurrences'].value_counts().sort_index()
        for num_neg, count in neg_dist.head(10).items():
            print(f"    {num_neg} negative possibility/ies: {count} pairs")

        # Save contradictory pairs
        contradictory_df.to_csv(output_dir / 'contradictory_sequence_pairs.csv', index=False)
        print(f"\n  Saved {len(contradictory_df)} contradictory pairs to: contradictory_sequence_pairs.csv")
    else:
        print(f"\n  No contradictory pairs found (all pairs appear only in same isolates)")

    # 3. Summary statistics
    print("\n3. Summary Statistics:")
    print("-" * 50)
    print(f"Total unique sequences: {len(seq_dup_stats)}")
    print(f"Sequences in >1 isolate: {len(duplicated_seqs)} ({len(duplicated_seqs)/len(seq_dup_stats)*100:.2f}%)")
    if len(pos_pairs_df) > 0:
        print(f"Unique positive pair candidates: {len(pos_pairs_df)}")
        print(f"Contradictory pairs (potential data leakage): {len(contradictory_df)} ({len(contradictory_df)/len(pos_pairs_df)*100:.2f}%)")

    # Save sequence duplication stats
    seq_dup_stats.to_csv(output_dir / 'sequence_duplication_stats.csv', index=False)
    print(f"\nSaved sequence duplication stats to: sequence_duplication_stats.csv")

    return seq_dup_stats, contradictory_df


# =============================================================================
# Genome pipeline functions (new)
# =============================================================================

def assign_genome_segments(
    genome_df: pd.DataFrame,
    replicon_to_segment: dict
    ) -> pd.DataFrame:
    """Assign canonical_segment to genome rows using direct replicon_type mapping.

    Args:
        genome_df: DataFrame with genome data (must have 'replicon_type' column)
        replicon_to_segment: Dict mapping replicon_type -> canonical_segment
            e.g. {'Segment 1': 'S1', ..., 'Segment 8': 'S8'}

    Returns:
        genome_df with 'canonical_segment' column added
    """
    print(f"\n{'='*50}")
    print("Assign canonical segments to genome data.")
    print('='*50)

    genome_df = genome_df.copy()
    genome_df['canonical_segment'] = genome_df['replicon_type'].map(replicon_to_segment)

    assigned = genome_df['canonical_segment'].notna().sum()
    unassigned = genome_df['canonical_segment'].isna().sum()
    print(f"Assigned: {assigned}, Unassigned: {unassigned}")
    print(f"Segment distribution:\n{genome_df['canonical_segment'].value_counts().sort_index()}")

    return genome_df


def apply_genome_basic_filters(
    genome_df: pd.DataFrame,
    output_dir: Path
    ) -> pd.DataFrame:
    """Apply basic filters to genome data.

    Filters:
    - Drop unassigned canonical_segment
    - Drop Poor quality assemblies
    - Drop unassigned replicon_type
    - Drop missing dna_seq
    """
    print(f"\n{'='*50}")
    print("Apply basic filters to genome data.")
    print('='*50)

    # Drop unassigned canonical segments
    unassigned = genome_df[genome_df['canonical_segment'].isna()]
    genome_df = genome_df[genome_df['canonical_segment'].notna()].reset_index(drop=True)
    print(f'\nDropped unassigned canonical_segment: {unassigned.shape}')
    print(f'Remaining genome_df:                  {genome_df.shape}')
    if not unassigned.empty:
        unassigned.to_csv(output_dir / 'genome_unassigned_segments.csv', sep=',', index=False)

    # Drop poor quality (genome-level quality)
    # TODO: Also consider filtering by contig_quality (per-contig quality from GTO)
    #       which may catch poor individual segments in otherwise good assemblies.
    poor = genome_df[genome_df['quality'] == 'Poor']
    genome_df = genome_df[genome_df['quality'] != 'Poor'].reset_index(drop=True)
    print(f'\nDropped Poor quality: {poor.shape}')
    print(f'Remaining genome_df: {genome_df.shape}')
    if not poor.empty:
        poor.to_csv(output_dir / 'genome_poor_quality.csv', sep=',', index=False)

    # Drop unassigned replicons
    unk = genome_df[genome_df['replicon_type'] == 'Unassigned']
    genome_df = genome_df[genome_df['replicon_type'] != 'Unassigned'].reset_index(drop=True)
    print(f'\nDropped unassigned replicons: {unk.shape}')
    print(f'Remaining genome_df:          {genome_df.shape}')
    if not unk.empty:
        unk.to_csv(output_dir / 'genome_unassigned_replicons.csv', sep=',', index=False)

    # Drop missing dna_seq
    missing_seq = genome_df[genome_df[DNA_SEQ_COL_NAME].isna()]
    genome_df = genome_df[genome_df[DNA_SEQ_COL_NAME].notna()].reset_index(drop=True)
    print(f'\nDropped missing {DNA_SEQ_COL_NAME}: {missing_seq.shape}')
    print(f'Remaining genome_df:       {genome_df.shape}')
    if not missing_seq.empty:
        missing_seq.to_csv(output_dir / 'genome_missing_seqs.csv', sep=',', index=False)

    return genome_df


def handle_genome_duplicates(
    genome_df: pd.DataFrame,
    output_dir: Path
    ) -> pd.DataFrame:
    """Handle genome sequence duplicates.

    Deduplicates on (dna_seq, assembly_id), keeping the first occurrence.
    """
    print(f"\n{'='*50}")
    print("Handle genome sequence duplicates.")
    print('='*50)

    print(f'genome_df before: {genome_df.shape}')

    # Find duplicates on (dna_seq, assembly_id)
    dup_mask = genome_df.duplicated(subset=[DNA_SEQ_COL_NAME, 'assembly_id'], keep='first')
    dups = genome_df[dup_mask]

    if not dups.empty:
        dups.to_csv(output_dir / 'genome_duplicate_sequences_removed.csv', index=False)
        print(f"Found {len(dups)} duplicate genome records (same dna_seq + assembly_id)")
        genome_df = genome_df[~dup_mask].reset_index(drop=True)
    else:
        print("No genome duplicates found")

    print(f'genome_df after:  {genome_df.shape}')
    return genome_df


# =============================================================================
# Module-level pipeline code
# =============================================================================

# Define paths - make configurable via command line or environment
parser = argparse.ArgumentParser(description='Preprocess protein and genome data from GTO files for Flu A')
parser.add_argument(
    '--config_bundle',
    type=str, default=None,
    help='Config bundle to use (e.g., flu). If not provided, raises error.'
)
parser.add_argument(
    '--force-reprocess',
    action='store_true',
    help='Force reprocess of GTO files, bypassing cache.'
)
parser.add_argument(
    '--max_files_to_preprocess',
    type=int, default=None,
    help='Maximum number of GTO files to process (None for all). Each GTO file = one isolate.'
)
args = parser.parse_args()

# Load config
config_path = str(project_root / 'conf') # Pass the config path explicitly
config_bundle = args.config_bundle
if config_bundle is None:
    raise ValueError("Must provide --config_bundle")
config = get_virus_config_hydra(config_bundle, config_path=config_path)
print_config_summary(config)

# Extract config values
VIRUS_NAME = config.virus.virus_name
DATA_VERSION = config.virus.data_version
RANDOM_SEED = resolve_process_seed(config, 'preprocessing')
MAX_FILES_TO_PREPROCESS = args.max_files_to_preprocess
core_functions = config.virus.core_functions
aux_functions = config.virus.aux_functions
selected_functions = config.virus.selected_functions
replicon_to_segment = dict(config.virus.replicon_to_segment)

print(f"\n{'='*40}")
print(f"Virus: {VIRUS_NAME}")
print(f"Config bundle: {config_bundle}")
print(f"{'='*40}")

# Resolve run suffix (manual override in config or auto-generate from sampling params)
RUN_SUFFIX = resolve_run_suffix(
    config=config,
    max_isolates=MAX_FILES_TO_PREPROCESS,
    seed=RANDOM_SEED,
    auto_timestamp=True
)

# Build preprocessing paths
paths = build_preprocessing_paths(
    project_root=project_root,
    virus_name=VIRUS_NAME,
    data_version=DATA_VERSION,
    run_suffix=RUN_SUFFIX,
    config=config
)

raw_data_dir = paths['raw_dir']
output_dir = paths['output_dir']
gto_dir = raw_data_dir
output_dir.mkdir(parents=True, exist_ok=True)

# Display paths
main_data_dir = project_root / 'data'
print(f'\nmain_data_dir:   {main_data_dir}')
print(f'gto_dir:         {gto_dir}')
print(f'output_dir:      {output_dir}')
print(f'run_suffix:      {RUN_SUFFIX if RUN_SUFFIX else "(none - full dataset)"}\n')


# =============================================================================
# GTO aggregation with caching
# =============================================================================

cached_protein_file = output_dir / 'protein_agg_from_GTOs.parquet'
cached_genome_file = output_dir / 'genome_agg_from_GTOs.parquet'

# Both must exist to use cache; else reprocess
cache_valid = (cached_protein_file.exists() and cached_genome_file.exists()
               and not args.force_reprocess)

if cache_valid:
    print(f"\nLoading cached aggregated data:")
    print(f"   Protein: {cached_protein_file}")
    print(f"   Genome:  {cached_genome_file}")
    prot_df = load_dataframe(cached_protein_file)
    genome_df = load_dataframe(cached_genome_file)
    print(f"   Loaded {len(prot_df)} protein records and {len(genome_df)} genome records from cache")
    print(f"   Use --force-reprocess to bypass cache and re-process GTO files")
else:
    if cached_protein_file.exists() and cached_genome_file.exists():
        print(f"\nForce reprocess requested. Re-processing from GTO files...")
    else:
        print(f"\nNo complete cached data found. Processing GTO files...")

    prot_df, genome_df = aggregate_data_from_gto_files(
        gto_dir,
        max_files=MAX_FILES_TO_PREPROCESS,
        random_seed=RANDOM_SEED
    )
    print(f'   Extracted {len(prot_df)} protein records and {len(genome_df)} genome records from GTO files')

    # Handle mixed encoding in 'location' column (protein-specific)
    ss = prot_df['location'].copy()
    ss = ss.map(lambda x: x.decode('utf-8', 'replace') if isinstance(x, (bytes, bytearray)) else x)
    ss = ss.map(lambda x: str(x) if isinstance(x, (int, float)) else x)
    prot_df['location'] = ss.astype('string[pyarrow]')

    # Save to cache
    prot_df.to_parquet(cached_protein_file, index=False)
    genome_df.to_parquet(cached_genome_file, index=False)
    print(f"   Cached protein data to: {cached_protein_file}")
    print(f"   Cached genome data to:  {cached_genome_file}")

print(f'\nprot_df:   {prot_df.shape}')
print(f'genome_df: {genome_df.shape}')

prot_df_no_seq = prot_df[prot_df[PROT_SEQ_COL_NAME].isna()]
print(f'Records with missing protein seq: {prot_df_no_seq.shape if not prot_df_no_seq.empty else 0}')
prot_df_no_seq.to_csv(output_dir / 'protein_agg_from_GTO_missing_seqs.csv', sep=',', index=False)

genome_df_no_seq = genome_df[genome_df[DNA_SEQ_COL_NAME].isna()]
print(f'Records with missing genome seq:  {genome_df_no_seq.shape if not genome_df_no_seq.empty else 0}')

# Check unique BRC feature IDs
print(f"\nTotal brc_fea_id:  {prot_df['brc_fea_id'].count()}")
print(f"Unique brc_fea_id: {prot_df['brc_fea_id'].nunique()}")


# =============================================================================
# PROTEIN PIPELINE (identical to preprocess_flu_protein.py)
# =============================================================================
print(f"\n{'#'*60}")
print(f"# PROTEIN PIPELINE")
print(f"{'#'*60}")

# EDA start ----------------------------------------
print("\nExplore 'replicon_type' counts.")
print(prot_df['replicon_type'].value_counts().sort_index())

print("\nShow all ['replicon_type', 'function'] combo counts.")
print_replicon_func_count(prot_df)

print('\nCloser look at "core" protein functions.')
print_replicon_func_count(prot_df, functions=core_functions)

print('\nCloser look at "auxiliary" protein functions.')
print_replicon_func_count(prot_df, functions=aux_functions)

print('\nCloser look at "selected" protein functions.')
print_replicon_func_count(prot_df, functions=selected_functions)

# Analyze protein counts per GTO file
protein_counts_per_file = analyze_protein_counts_per_file(
    prot_df, core_functions, aux_functions, selected_functions
)
# EDA end ----------------------------------------

# Assign protein segments
prot_df = assign_protein_segments(prot_df, use_core=True, use_aux=True)

# Basic protein filters
prot_df = apply_protein_basic_filters(prot_df)

# Handle protein duplicates
prot_df = handle_protein_duplicates(prot_df, print_eda=False)

# Analyze sequence duplicates for pair classification
# analyze_sequence_duplicates_for_pair_classification(prot_df, output_dir=output_dir)

# Clean protein sequences
print(f"\n{'='*50}")
print("Explore ambiguities in protein sequences.")
print(f'='*50)
org_cols = prot_df.columns
prot_df = analyze_protein_ambiguities(prot_df)
ambig_cols = [c for c in prot_df.columns if c not in org_cols]
ambig_df = prot_df[prot_df['has_ambiguities'] == True]
ambig_df = ambig_df[['assembly_id', 'brc_fea_id', PROT_SEQ_COL_NAME] + ambig_cols]
cols_print = ['assembly_id', 'brc_fea_id'] + ambig_cols
print(f'Protein sequences with ambiguous chars: {ambig_df.shape[0]}')
print(ambig_df[cols_print])

print(f"\n{'='*50}")
print("Summarize of ambiguities in protein sequences.")
print('='*50)
sa = summarize_ambiguities(prot_df)
pprint(sa)
print(f"Total protein seqs with terminal stops: {sa['terminal_stops']} "
      f"({sa['percent_terminal_stops']:.2f}%)")
print(f"Total protein seqs with internal stops: {sa['internal_stops']} "
      f"({sa['percent_internal_stops']:.2f}%)")
print("Non-standard residue distribution (excluding terminal stops):")
for aa, details in sa['non_standard_residue_summary'].items():
    meaning = details['meaning'] if aa != '*' else 'Premature stop or error'
    print(f"  {aa}: occurred in {details['count']} seqs ({meaning})")

# Prepare for ESM-2
print(f"\n{'='*50}")
print(f'Prepare protein sequences for ESM-2.')
print(f'='*50)
max_internal_stops = 0.1
max_x_residues = 0.1
x_imputation = 'G'
strip_terminal_stop = True
prot_df, problematic_seqs_df = prepare_sequences_for_esm2(
    prot_df,
    x_imputation=x_imputation,
    max_internal_stops=max_internal_stops,
    max_x_residues=max_x_residues,
    strip_terminal_stop=strip_terminal_stop
)
if not problematic_seqs_df.empty:
    problematic_seqs_df = problematic_seqs_df[['file', 'brc_fea_id', 'type', 'function', 'quality', 'prot_seq', 'problem']]
    problematic_seqs_df.to_csv(output_dir / 'problematic_protein_seqs.csv', sep=',', index=False)

# Filter out sequences that failed ESM-2 preparation
n_before_filter = len(prot_df)
prot_df = prot_df[prot_df['esm2_ready_seq'].notna()].reset_index(drop=True)
n_filtered = n_before_filter - len(prot_df)
if n_filtered > 0:
    print(f'\nFiltered out {n_filtered} sequences that failed ESM-2 preparation (already saved to problematic_protein_seqs.csv)')
    print(f'Remaining sequences: {len(prot_df)}')

# Final duplicate counts
print(f"\n{'='*50}")
print("Final duplicate counts.")
print('='*50)
prot_dups = (
    prot_df[prot_df.duplicated(subset=[PROT_SEQ_COL_NAME], keep=False)]
    .sort_values(PROT_SEQ_COL_NAME)
    .reset_index(drop=True)
)
dup_counts = prot_dups.groupby(PROT_SEQ_COL_NAME).agg(num_files=('file', 'nunique')).reset_index()
print(dup_counts['num_files'].value_counts().reset_index(name='total_cases'))

# Save final protein data
del prot_dups, dup_counts, problematic_seqs_df, ambig_df
print(f"\n{'='*50}")
print("Save final protein data.")
print('='*50)
prot_df.to_csv(output_dir / 'protein_final.csv', sep=',', index=False)
prot_df.to_parquet(output_dir / 'protein_final.parquet', index=False)
print(f'prot_df final: {prot_df.shape}')
print(f"Unique protein sequences: {prot_df[PROT_SEQ_COL_NAME].nunique()}")
aa = print_replicon_func_count(prot_df, more_cols=['canonical_segment'])
aa.to_csv(output_dir / 'protein_final_segment_mappings_stats.csv', sep=',', index=False)
print(prot_df['canonical_segment'].value_counts())


# =============================================================================
# GENOME PIPELINE
# =============================================================================
print(f"\n{'#'*60}")
print(f"# GENOME PIPELINE")
print(f"{'#'*60}")

# Assign genome segments
genome_df = assign_genome_segments(genome_df, replicon_to_segment)

# Basic filters
genome_df = apply_genome_basic_filters(genome_df, output_dir)

# DNA QC
print(f"\n{'='*50}")
print("DNA sequence QC.")
print('='*50)
genome_df = summarize_dna_qc(genome_df, seq_col=DNA_SEQ_COL_NAME)
print(f"genome_df after QC: {genome_df.shape}")
print(f"\nLength stats:\n{genome_df['length'].describe()}")
print(f"\nGC content stats:\n{genome_df['gc_content'].describe()}")
print(f"\nAmbig fraction stats:\n{genome_df['ambig_frac'].describe()}")

# Handle genome duplicates
genome_df = handle_genome_duplicates(genome_df, output_dir)

# Save final genome data
print(f"\n{'='*50}")
print("Save final genome data.")
print('='*50)
genome_df.to_csv(output_dir / 'genome_final.csv', sep=',', index=False)
genome_df.to_parquet(output_dir / 'genome_final.parquet', index=False)
print(f'genome_df final: {genome_df.shape}')
print(f"Unique genome sequences: {genome_df[DNA_SEQ_COL_NAME].nunique()}")
print(f"Unique assemblies: {genome_df['assembly_id'].nunique()}")
print(genome_df['canonical_segment'].value_counts().sort_index())


# =============================================================================
# Save experiment metadata
# =============================================================================
print(f"\n{'='*50}")
print("Save experiment metadata.")
print('='*50)
save_experiment_metadata(
    output_dir=output_dir,
    config=config,
    stage='preprocessing',
    script_name=Path(__file__).name,
    additional_info={
        'total_proteins_processed': len(prot_df),
        'unique_protein_sequences': prot_df[PROT_SEQ_COL_NAME].nunique(),
        'total_genome_segments_processed': len(genome_df),
        'unique_genome_sequences': genome_df[DNA_SEQ_COL_NAME].nunique(),
        'unique_files': prot_df['file'].nunique(),
        'processing_time_seconds': total_timer.elapsed
    }
)

save_experiment_summary(
    output_dir=output_dir,
    stage='preprocessing',
    summary={
        'Virus': VIRUS_NAME,
        'Config bundle': config_bundle,
        'Data version': DATA_VERSION,
        'Run suffix': RUN_SUFFIX if RUN_SUFFIX else '(none - full dataset)',
        'Master seed': config.master_seed,
        'Preprocessing seed': RANDOM_SEED,
        'Selected functions': selected_functions,
        'Total proteins processed': len(prot_df),
        'Unique protein sequences': prot_df[PROT_SEQ_COL_NAME].nunique(),
        'Total genome segments processed': len(genome_df),
        'Unique genome sequences': genome_df[DNA_SEQ_COL_NAME].nunique(),
        'Unique files processed': prot_df['file'].nunique(),
        'Processing time': total_timer.get_elapsed_string()
    }
)

print(f'\nFinished {Path(__file__).name}!')
