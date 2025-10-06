"""
Preprocess protein data from GTO files for Flu A.
"""
import json
import os
import random
import sys
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from typing import Optional

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# print(f'project_root: {project_root}')

from src.utils.timer_utils import Timer
from src.utils.gto_utils import (
    extract_assembly_info,
    enforce_single_file,
)
from src.utils.protein_utils import (
    analyze_protein_ambiguities,
    summarize_ambiguities,
    prepare_sequences_for_esm2,
    print_replicon_func_count
)
from src.utils.config_hydra import (
    get_virus_config_hydra, 
    print_config_summary
)

total_timer = Timer()

# Load virus-specific configuration using Hydra
VIRUS_NAME = 'flu_a'
# TRAINING_CONFIG = 'base'  # Optional: override bundle's default training config

# Pass the config path explicitly
config_path = str(project_root / 'conf')
config = get_virus_config_hydra(VIRUS_NAME, config_path=config_path)
print_config_summary(config)

# Extract configuration values
DATA_VERSION = config.virus.data_version
MAX_FILES_TO_PROCESS = config.virus.max_files_to_process
RANDOM_SEED = config.virus.random_seed

# Define paths
main_data_dir = project_root / 'data'

# Use symlinked names (Flu_A -> Full_Flu_Annos, Bunya -> Anno_Updates)
# Prefer subset directories when available, fallback to full dataset
base_data_dir = main_data_dir / 'raw' / 'Flu_A' / DATA_VERSION
subset_data_dir = main_data_dir / 'raw' / 'Flu_A' / f'{DATA_VERSION}_subset_5k'

# Prefer subset directory if it exists (for fast development)
if subset_data_dir.exists() and subset_data_dir.is_dir():
    raw_data_dir = subset_data_dir
    output_dir = main_data_dir / 'processed' / VIRUS_NAME / f'{DATA_VERSION}_subset_5k'
    print(f"Using subset dataset: {raw_data_dir}")
else:
    raw_data_dir = base_data_dir
    output_dir = main_data_dir / 'processed' / VIRUS_NAME / DATA_VERSION
    print(f"Using full dataset: {raw_data_dir}")

gto_dir = raw_data_dir
output_dir.mkdir(parents=True, exist_ok=True)

print(f'main_data_dir:   {main_data_dir}')
print(f'raw_data_dir:    {raw_data_dir}')
print(f'gto_dir:         {gto_dir}')
print(f'output_dir:      {output_dir}')

seq_col_name = 'prot_seq'

# Get core and auxiliary functions from config
core_functions = config.virus.core_functions
aux_functions = config.virus.aux_functions
selected_functions = config.virus.selected_functions

def validate_protein_counts(
    df: pd.DataFrame,
    core_only: bool = False,
    verbose: bool = False
    ) -> None:
    """Ensure each assembly_id has ≤max_core_proteins core proteins if core_only=True.

    This func is specific to preprocessing validation, tightly coupled
    to core_functions and assembly IDs (keep it in this script).
    """
    if core_only:
        df = df[df['function'].isin(core_functions)]
        protein_type = "core"
        max_expected = len(core_functions)
    else:
        protein_type = "total"
        max_expected = None
    
    print(f"\nValidating {protein_type} protein counts per assembly...")
    
    for aid, grp in df.groupby('assembly_id'):
        n_proteins = len(grp)
        if core_only and n_proteins > max_expected:
            print(f"⚠️  WARNING: assembly_id {aid} has {n_proteins} core proteins, expected ≤{max_expected}")
            print(f"   Files: {grp['file'].unique()}")
            print(f"   Functions: {grp['function'].tolist()}")
        elif verbose:
            print(f"assembly_id {aid}: {n_proteins} {protein_type} proteins")
            if core_only:
                print(f"   Functions: {grp['function'].tolist()}")
    
    if core_only:
        print(f"✅ Validation complete: All assemblies have ≤{max_expected} core proteins")
    else:
        print(f"✅ Validation complete: All assemblies processed")
    
    return True


def get_protein_data_from_gto(gto_file_path: Path) -> pd.DataFrame:
    """
    Extract protein data and metadata from a GTO file into a DataFrame.
    
    - The 'id' in 'features' contains values like: fig|1316165.15.CDS.3, fig|11588.3927.CDS.1, etc.
    - These are PATRIC (BV-BRC) feature IDs: fig|<genome_id>.<feature_type>.<feature_number>
    - The genome_id (e.g., 1316165.15) is the internal genome identifier.
    - The 'id' can be used to trace the feature to its source genome in PATRIC/GTOs.
    - This is a not GenBank accession number (e.g., NC_038733.1, OL987654.1).

    - The 'location' in 'features' contains: [[ <segment_id>, <start>, <strand>, <end>]]
    - For example: "location": [[ "NC_086346.1", "70", "+", "738" ]]
    - It means: this feature is located on segment NC_086346.1 (positive strand), from
        nucleotide 70 to 738.
    - Note that the first str in 'location' (i.e., NC_086346.1) is a GenBank accession
        used by NCBI for nucleotide entries.    
    """
    with open(gto_file_path, 'r') as file:
        gto = json.load(file)

    # Create genbank_ctg_id-to-replicon_type mapping using 'contigs'.
    # This mapping is used to assign the replicon_type to protein seq.
    # Note there is a link between 'features' items and 'contigs' items.
    # The link is embedded in:
    #   the first str in 'location' on the 'features' side
    #   the 'id' on the 'contigs' side
    # Iterate over 'contigs' items in GTO (if available)
    # (each item is a dict with keys: 'id', 'dna', 'replicon_type', 'replicon_geometry')
    # 'id': genbank_ctg_id (e.g., NC_086346.1)
    # 'replicon_type': segment label (e.g., "Segment [1-8]")
    segment_map = {
        contig['id']: contig.get('replicon_type', 'Unassigned')
        for contig in gto.get('contigs', [])
    }

    # Extract data from 'features'
    fea_cols = ['id', 'type', 'function', 'protein_translation', 'location']
    rows = []
    for fea_dict in gto['features']:
        row = {k: fea_dict.get(k, None) for k in fea_cols}
        row['length'] = len(fea_dict.get('protein_translation', '')) if 'protein_translation' in fea_dict else 0
        # Example of 'location': [["NC_086346.1", "70", "+", "738"]]
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
        # Example of 'family_assignments':
        # [["Phenuiviridae", "Phenuiviridae.L.9.pssm", "RNA-dependent RNA polymerase", "LowVan Annotate"]]
        family_assignments = fea_dict.get('family_assignments')
        row['family'] = (
            family_assignments[0][1]
            if (isinstance(family_assignments, list)
                and len(family_assignments) > 0
                and len(family_assignments[0]) > 1)
            else pd.NA
        )
        rows.append(row)
    
    df = pd.DataFrame(rows)

    # Extract additional metadata
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

    df = pd.merge(meta_df, df, how='cross')
    df.rename(columns={'protein_translation': 'prot_seq', 'id': 'brc_fea_id'}, inplace=True)
    return df


def aggregate_protein_data_from_gto_files(
    gto_dir: Path,
    max_files: Optional[int] = None,
    random_seed: Optional[int] = None
    ) -> pd.DataFrame:
    """Aggregate protein data from GTO files.
    
    Args:
        gto_dir: Directory containing GTO files
        max_files: Maximum number of files to process (None for all)
        random_seed: Random seed for reproducible sampling
    """
    # breakpoint()
    dfs = []
    gto_files = sorted(gto_dir.glob('*.gto'))  # Updated for Flu A files

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

    for fpath in tqdm(gto_files, desc='Aggregating protein data from GTO files'):
        dfs.append(get_protein_data_from_gto(fpath))
    return pd.concat(dfs, axis=0).reset_index(drop=True)


def analyze_protein_counts_per_file(
    df: pd.DataFrame, 
    core_functions: list[str], 
    aux_functions: list[str],
    selected_functions: list[str]
    ) -> None:
    """ Analyze protein counts per GTO file. """
    print("\n" + "=" * 60)
    print("PROTEIN COUNTS PER GTO FILE")
    print("=" * 60)

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
    # print(f"\nProtein count statistics per file:")
    # print(all_counts.describe())

    # Show unidentified proteins
    if len(other_proteins) > 0:
        print(f"\n🔍 UNIDENTIFIED PROTEINS (not in core_functions or aux_functions):")
        print(f"Found {len(other_proteins)} unique protein functions not in config:")
        for i, protein in enumerate(sorted(other_proteins), 1):
            count = df[df['function'] == protein].shape[0]
            print(f"  {i:2d}. {protein} (appears in {count} records)")
    else:
        print(f"\n✅ All proteins are classified (no unidentified proteins found)")

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
        print(low_core[['total_proteins', 'core_proteins', 'aux_proteins', 'other_proteins']].head(10))
    else:
        print(f"All files have ≥ {expected_core} core proteins ✓")

    # Show files with more than known core proteins (potential duplicates)
    high_core = all_counts[all_counts['core_proteins'] > expected_core]
    if len(high_core) > 0:
        print(f"\nFound {len(high_core)} files with > {expected_core} core proteins:")
        print(high_core[['total_proteins', 'core_proteins', 'aux_proteins', 'other_proteins']].head(10))
    else:
        print(f"No files have > {expected_core} core proteins ✓")

    # Analyze intra-file protein (function) duplicates in selected proteins across all files
    print(f"\n🔍 ANALYZING INTRA-FILE DUPLICATES IN 'SELECTED' (PROTEIN) FUNCTIONS ACROSS ALL FILES:")
    analyze_intra_file_protein_duplicates(df, selected_functions, output_dir)

    return all_counts


def analyze_intra_file_protein_duplicates(
    df: pd.DataFrame, 
    subset_functions: Optional[list[str]] = None,
    output_dir: Optional[Path] = None
    ) -> None:
    """Analyze intra-file duplicates in proteins across all files.

    Args:
        df: Full protein dataframe
        subset_functions: List of protein functions to analyze. If None, analyzes ALL functions.
        output_dir: Directory to save problematic proteins file. If None, uses current directory.
    """
    # Set default output directory
    if output_dir is None:
        output_dir = Path(".")
        print(f"⚠️  No output_dir provided, using current directory: {output_dir.absolute()}")

    # If no subset specified, use all unique functions
    if subset_functions is None:
        subset_functions = df['function'].unique().tolist()
        print(f"🔍 Analyzing ALL {len(subset_functions)} protein functions for intra-file duplicates...")
    else:
        print(f"🔍 Analyzing {len(subset_functions)} selected protein functions for intra-file duplicates...")

    # Filter to subset proteins only (keep original df unchanged)
    sub_df = df[df['function'].isin(subset_functions)]

    print(f"Files to analyze: {sub_df['file'].nunique()}")
    print(f"Protein records: {len(sub_df)}")

    # Count subset protein occurrences per file
    subset_counts_per_file = sub_df.groupby(['file', 'function']).size().reset_index(name='count')

    # Find subset proteins that appear multiple times
    subset_duplicates = subset_counts_per_file[subset_counts_per_file['count'] > 1]

    if len(subset_duplicates) > 0:
        print(f"\n🚨 Found {len(subset_duplicates)} subset protein duplicates:")
        print("=" * 40)

        # Prepare data for saving to file
        intra_file_dups = []

        for _, row in subset_duplicates.iterrows():
            file_name = row['file']
            function = row['function']
            count = row['count']

            # Get the actual records for this duplicate
            duplicate_records = sub_df[
                (sub_df['file'] == file_name) & 
                (sub_df['function'] == function)
            ]

            print(f"\n📁 File: {file_name}")
            print(f"   🔍 Protein function: {function}")
            print(f"   📊 Count: {count} occurrences")
            print(f"   📋 Details:")

            for i, (_, record) in enumerate(duplicate_records.iterrows(), 1):
                print(f"     {i}. Assembly ID: {record['assembly_id']}")
                print(f"         Replicon: {record['replicon_type']}")
                print(f"         BRC Feature ID: {record['brc_fea_id']}")
                if 'prot_seq' in record:
                    seq_preview = record['prot_seq'][:40] + "..." if len(record['prot_seq']) > 40 else record['prot_seq']
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
            print(f"\n💾 Saved {len(intra_file_dups)} intra-file protein duplicates to: {output_file}")
    else:
        print("✅ No subset protein duplicates found")

    # Summary statistics
    print(f"\n📊 INTRA-FILE PROTEIN DUPLICATE SUMMARY:")
    print(f"   Files analyzed: {sub_df['file'].nunique()}")
    print(f"   Protein records across files: {len(sub_df)}")
    print(f"   Unique proteins (subset functions) across files: {sub_df['function'].nunique()}")
    print(f"   Intra-file duplicates on 'function' column: {len(subset_duplicates)}")


def assign_segment_using_core_proteins(
    prot_df: pd.DataFrame,
    data_version: str = 'July_2025'
    ) -> pd.DataFrame:
    """Assign canonical segments (1-8) based on core protein functions from config.
    
    Each Flu A genome segment (replicon) encodes distinct proteins:

    1. Segment 1 → PB2 (RNA-dependent RNA polymerase PB2 subunit)
    2. Segment 2 → PB1 (RNA-dependent RNA polymerase catalytic core PB1 subunit)  
    3. Segment 3 → PA (RNA-dependent RNA polymerase PA subunit)
    4. Segment 4 → HA (Hemagglutinin precursor)
    5. Segment 5 → NP (Nucleocapsid protein)
    6. Segment 6 → NA (Neuraminidase protein)
    7. Segment 7 → M1 (Matrix protein 1)
    8. Segment 8 → NS1/NS2 (Non-structural proteins)

    These core proteins are present in nearly all Flu A genomes.

    We define canonical segment labels (1-8) by mapping known functions
        to these segments.

    Mapping is conditional: only apply if function and replicon type match
        expected biological patterns.
    """
    prot_df = prot_df.copy()

    # Get conditional mappings from config
    core_mappings = config.virus.conditional_segment_mappings.core_proteins
    core_map = pd.DataFrame(core_mappings)
    # print(core_map)

    # Merge (assign temp segments where function-replicon pairs match)
    # how='left': keeps all rows, assigns NaN where mapping doesn't apply
    prot_df = prot_df.merge(core_map, on=['function', 'replicon_type'], how='left')
    # print(prot_df.iloc[:,-4:])

    # Assign core_segment to canonical_segment
    mask = prot_df['core_segment'].notna()
    prot_df.loc[mask, 'canonical_segment'] = prot_df.loc[mask, 'core_segment']
    # print(prot_df.iloc[:,-4:])

    # Diagnostics
    print_replicon_func_count(prot_df, functions=core_functions)
    df = print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)
    # df.to_csv(output_dir / 'core_mappings_eda.csv', sep=',', index=False)

    prot_df = prot_df.drop(columns=['core_segment'])
    return prot_df


def assign_segment_using_aux_proteins(
    prot_df: pd.DataFrame,
    data_version: str = 'July_2025'
    ) -> pd.DataFrame:
    """Assign canonical segments (1-8) based on auxiliary protein functions from config.    

    Some additional protein functions are consistently matched with single 
    segment (replicon_type) across all isolates in our dataset.

    These are non-structural proteins but can be mapped confidently:

    - NS1, NS3 proteins → Segment 8 → assign as '8'
    - M2 ion channel, M42 → Segment 7 → assign as '7'
    - PA-X, PA-N155, PA-N182 → Segment 3 → assign as '3'
    - PB1-F2, PB1-N40 → Segment 2 → assign as '2'
    - PB2-S1 → Segment 1 → assign as '1'
    - HA1, HA2 → Segment 4 → assign as '4'

    Approach:
    - Use auxiliary function-to-segment mapping for unassigned proteins.
    - Do NOT assign segments if segment-function mapping is ambiguous.
    """
    prot_df = prot_df.copy()

    # Get conditional mappings from config
    aux_mappings = config.virus.conditional_segment_mappings.aux_proteins
    aux_map = pd.DataFrame(aux_mappings)
    # print(aux_map)
    
    # Merge (assign temp canonical segments where function-replicon pairs match)
    # how='left': keeps all rows, assigns NaN where mapping doesn't apply
    prot_df = prot_df.merge(aux_map, on=['function', 'replicon_type'], how='left')
    # print(prot_df.iloc[:,-4:])

    # Assign aux_segment to canonical_segment (only for unassigned proteins)
    mask = prot_df['canonical_segment'].isna() & prot_df['aux_segment'].notna()
    prot_df.loc[mask, 'canonical_segment'] = prot_df.loc[mask, 'aux_segment']
    # print(prot_df.iloc[:,-4:])

    # Diagnostics
    print_replicon_func_count(prot_df, functions=aux_functions)
    df = print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)
    # df.to_csv(output_dir / 'aux_mappings_eda.csv', sep=',', index=False)

    prot_df = prot_df.drop(columns=['aux_segment'])
    return prot_df


def assign_segments(
    prot_df: pd.DataFrame,
    data_version: str = 'July_2025',
    use_core: bool = True,
    use_aux: bool = True
    ) -> pd.DataFrame:
    """Assign canonical segments using core and/or auxiliary proteins."""
    prot_df = prot_df.copy()
    prot_df['canonical_segment'] = pd.NA  # Placeholder for canonical_segment

    if use_core:
        prot_df = assign_segment_using_core_proteins(prot_df, data_version)

    if use_aux:
        prot_df = assign_segment_using_aux_proteins(prot_df, data_version)

    # Diagnostics
    print("\nCore and auxiliary protein segment mappings:")
    all_functions = config.virus.core_functions + config.virus.aux_functions
    df = print_replicon_func_count(
        prot_df,
        functions=all_functions,
        more_cols=['canonical_segment'],
        drop_na=False)
    df.to_csv(output_dir / 'seg_mapped_core_and_aux_eda.csv', sep=',', index=False)

    # print('\nAvailable [replicon_type, function] combos for "core" and "auxiliary" protein mapping:')
    # print_replicon_func_count(prot_df, functions=all_functions)
    # print('\nAll [replicon_type, function] combos and assigned segments:')
    # df = print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)
    
    unmapped = prot_df[prot_df['canonical_segment'].isna()]
    print(f"\nUnmapped segments: {len(unmapped)}")
    print_replicon_func_count(unmapped)
    breakpoint()
    return prot_df


def apply_basic_filters(prot_df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic filters to protein data."""
    breakpoint()

    # Drop unassigned canonical segments
    # print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)
    unassigned_df = prot_df[prot_df['canonical_segment'].isna()]
    # unassgined_df = prot_df[prot_df['canonical_segment'].isna()].sort_values('canonical_segment', ascending=False)
    prot_df = prot_df[prot_df['canonical_segment'].notna()].reset_index(drop=True)
    print(f'Dropped unassigned canonical_segment: {unassigned_df.shape}')
    print(f'Remaining prot_df:                    {prot_df.shape}')
    unassigned_df.to_csv(output_dir / 'protein_unassigned_segments.csv', sep=',', index=False)
    # print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)

    # Keep CDS only
    # print_replicon_func_count(prot_df, more_cols=['canonical_segment', 'type'], drop_na=False)
    non_cds = prot_df[prot_df['type'] != 'CDS']
    # non_cds = prot_df[prot_df['type'] != 'CDS'].sort_values('type', ascending=False)
    prot_df = prot_df[prot_df['type'] == 'CDS'].reset_index(drop=True)
    print(f'Dropped non-CDS samples: {non_cds.shape}')
    print(f'Remaining prot_df:       {prot_df.shape}')
    non_cds.to_csv(output_dir / 'protein_non_cds.csv', sep=',', index=False)
    # print_replicon_func_count(prot_df, more_cols=['canonical_segment', 'type'], drop_na=False)

    # Drop poor quality
    # print_replicon_func_count(prot_df, more_cols=['canonical_segment', 'type'], drop_na=False)
    poor_df = prot_df[prot_df['quality'] == 'Poor']
    # poor_df = prot_df[prot_df['quality'] == 'Poor'].sort_values('quality', ascending=False)
    prot_df = prot_df[prot_df['quality'] != 'Poor'].reset_index(drop=True)
    print(f'Dropped Poor quality samples: {poor_df.shape}')
    print(f'Remaining prot_df:            {prot_df.shape}')
    poor_df.to_csv(output_dir / 'protein_poor_quality.csv', sep=',', index=False)
    # print_replicon_func_count(prot_df, more_cols=['canonical_segment', 'type'], drop_na=False)

    # Drop unassigned replicons
    # print_replicon_func_count(prot_df, more_cols=['canonical_segment', 'type'], drop_na=False)
    unk_df = prot_df[prot_df['replicon_type'] == 'Unassigned']
    # unk_df = prot_df[prot_df['replicon_type'] == 'Unassigned'].sort_values('replicon_type', ascending=False)
    prot_df = prot_df[prot_df['replicon_type'] != 'Unassigned'].reset_index(drop=True)
    print(f'Dropped unassigned replicons: {unk_df.shape}')
    print(f'Remaining prot_df:            {prot_df.shape}')
    unk_df.to_csv(output_dir / 'protein_unassigned_replicons.csv', sep=',', index=False)
    # print_replicon_func_count(prot_df, more_cols=['canonical_segment', 'type'], drop_na=False)

    return prot_df


def handle_duplicates(
    prot_df: pd.DataFrame,
    print_eda: bool = False
    ) -> pd.DataFrame:
    """Handle protein sequence duplicates."""
    seq_col_name = 'prot_seq'
    breakpoint()

    # EDA start ------------------------------------------------
    if print_eda:
        aa = print_replicon_func_count(prot_df,
            more_cols=['canonical_segment'], drop_na=False)

        # Get all duplicate protein sequences
        print(f'\nprot_df: {prot_df.shape}')
        prot_dups = (
            prot_df[prot_df.duplicated(subset=[seq_col_name], keep=False)]
            .sort_values(seq_col_name)
            .reset_index(drop=True).copy()
        )
        # prot_dups.to_csv(output_dir / 'protein_all_dups.csv', sep=',', index=False)
        print(f"Duplicates on '{seq_col_name}': {prot_dups.shape}")
        print(prot_dups[:4][['file', 'assembly_id', 'genbank_ctg_id',
            'replicon_type', 'brc_fea_id', 'function', 'canonical_segment']])

        print(f'\nCount how many unique files contain each duplicated sequence:')
        dup_counts = prot_dups.groupby(seq_col_name).agg(num_files=('file', 'nunique')).reset_index()
        print(dup_counts['num_files'].value_counts().reset_index(name='total_cases'))

        def explore_groupby(df, columns):
            seq_col_name = 'prot_seq'
            df = df.copy()
            grouped = df.groupby(columns)
            # grouped_iterator = iter(grouped)
            # first_key, first_group = next(grouped_iterator)

            for keys, grp in grouped:
                print('\nDuplicate group (rows with this seq):')
                print(grp[:4][[seq_col_name, 'file', 'assembly_id', 'genbank_ctg_id',
                    'replicon_type', 'brc_fea_id', 'function']])
                break  # remove this to loop over more groups
            return grp

        explore_groupby(prot_dups, seq_col_name)

        # Expand dup_counts to include more info
        # num_occurrences >= num_files -> there are cases where a protein sequence appears multiple
        #   times in the same file.
        # num_functions >= num_replicons -> it's not a 1-1 mapping between replicons (segment)
        #   and functions, because a segment can encode multiple proteins.
        dup_stats = (
            prot_dups.groupby(seq_col_name).agg(
                num_occurrences=(seq_col_name, 'count'), # total times the sequence appears
                num_files=('file', 'nunique'),           # total unique files the sequence appears in
                num_functions=('function', 'nunique'),   # total distinct functions are assigned to the sequence
                num_replicons=('replicon_type', 'nunique'), 
                functions=('function', lambda x: list(set(x))),
                replicons=('replicon_type', lambda x: list(set(x))),
                brc_fea_ids=('brc_fea_id', lambda x: list(set(x))),  # feature ids
                files=('file', lambda x: list(set(x))),  # all files containing the protein duplicate
                assembly_ids=('assembly_id', lambda x: list(set(x))),
                num_assemblies=('assembly_id', 'nunique'), # total unique assemblies
                assembly_prefixes=('assembly_prefix', lambda x: list(set(x))),
                num_assembly_prefixes=('assembly_prefix', 'nunique'),
            )
            .sort_values('num_occurrences', ascending=False)
            .reset_index()
            .copy()
        )
        # dup_stats.to_csv(output_dir / 'protein_dups_stats.csv', sep=',', index=False)
    # EDA end ------------------------------------------------

    # Enforce single file
    print("\nEnforce single file per assembly_id.")
    print(f'prot_df before: {prot_df.shape}')
    prot_df = enforce_single_file(prot_df)
    print(f'prot_df after: {prot_df.shape}')

    # Validate protein counts
    print("\nValidate protein counts.")
    validate_protein_counts(prot_df, core_only=True)   # Check core proteins
    validate_protein_counts(prot_df, core_only=False)  # Check all proteins

    # EDA start ------------------------------------------------
    # if print_eda:
        # print('\n---- FIND dups on [prot_seq, assembly_id, function, replicon_type] ----')
        # dup_cols = [seq_col_name, 'assembly_id', 'function', 'replicon_type']
        # gca_gcf_dups = (
        #     prot_df[prot_df.duplicated(subset=dup_cols, keep=False)]
        #     .sort_values(dup_cols)
        #     .reset_index(drop=True).copy()
        # )
        # gca_gcf_dups.to_csv(output_dir / 'protein_dups_same_isolate.csv', sep=',', index=False)

        # print(f'gca_gcf_dups: {gca_gcf_dups.shape}')
        # print(f"dups unique seqs: {gca_gcf_dups['prot_seq'].nunique()}")
        # print(gca_gcf_dups[:4][dup_cols + ['assembly_prefix']])

        # gca_gcf_dups_eda = gca_gcf_dups.groupby(dup_cols).agg(
        #     num_assembly_prefixes=('assembly_prefix', 'nunique'),
        #     assembly_prefixes=('assembly_prefix', lambda x: list(set(x))),
        #     brc_fea_ids=('brc_fea_id', lambda x: list(set(x))),
        #     ).sort_values(dup_cols).reset_index()
        # gca_gcf_dups_eda.to_csv(output_dir / 'protein_dups_same_isolate_eda.csv', sep=',', index=False)
        # del dup_cols, # gca_gcf_dups, gca_gcf_dups_eda
    # EDA end --------------------------------------------------

    # EDA start --------------------------------------------------
    if print_eda:
        print('\n---- FIND dups within the same GTO file (intra-file dups) ----')
        dup_cols = [seq_col_name, 'file']
        same_file_dups = (
            prot_df[prot_df.duplicated(subset=dup_cols, keep=False)]
            .sort_values(dup_cols + ['brc_fea_id'])
            .reset_index(drop=True).copy()
        )
        same_file_dups.to_csv(output_dir / 'protein_dups_within_file.csv', sep=',', index=False)

        print(f'same_file_dups: {same_file_dups.shape}')
        print(f"dups unique seqs: {same_file_dups[seq_col_name].nunique()}")
        if not same_file_dups.empty:
            print(same_file_dups[:4][dup_cols + ['brc_fea_id']])

        same_file_dups_eda = same_file_dups.groupby(dup_cols).agg(
            num_brc_fea_ids=('brc_fea_id', 'nunique'),
            num_funcs=('function', 'nunique'),
            brc_fea_ids=('brc_fea_id', lambda x: list(set(x))),
            functions=('function', lambda x: list(set(x))),
            ).sort_values(dup_cols).reset_index()
        # same_file_dups_eda.to_csv(output_dir / 'protein_dups_within_file_eda.csv', sep=',', index=False)
        del dup_cols, # same_file_dups, same_file_dups_eda
    # EDA end --------------------------------------------------

    return prot_df


# Aggregate protein data from GTO files
if 'subset_' in str(raw_data_dir):
    # For other subset sizes, count them (should be fast)
    total_files = len(sorted(gto_dir.glob('*.gto')))
    print(f"\nUsing subset: {total_files} GTO files")
else:
    # For full datasets, count them (may be slow)
    total_files = len(sorted(gto_dir.glob('*.gto')))
    print(f"\nUsing full dataset: {total_files} GTO files")

# breakpoint()
prot_df = aggregate_protein_data_from_gto_files(
    gto_dir, max_files=MAX_FILES_TO_PROCESS,
    random_seed=RANDOM_SEED
)
print(f'prot_df: {prot_df.shape}')

# Save initial data
prot_df.to_csv(output_dir / 'protein_agg_from_GTOs.csv', sep=',', index=False)
prot_df_no_seq = prot_df[prot_df[seq_col_name].isna()]
print(f'Records with missing protein sequence: {prot_df_no_seq.shape if not prot_df_no_seq.empty else 0}')
prot_df_no_seq.to_csv(output_dir / 'protein_agg_from_GTO_missing_seqs.csv', sep=',', index=False)

# Check unique BRC feature IDs
print(f"\nTotal brc_fea_id:  {prot_df['brc_fea_id'].count()}")
print(f"Unique brc_fea_id: {prot_df['brc_fea_id'].nunique()}")


# EDA start ----------------------------------------
"""
Intermediate EDA

Explore all 'replicon_type' entires (segments):
Segment 3    366                                                                                                                                            
Segment 2    302                                                                                                                                              
Segment 4    300                                                                                                                                              
Segment 7    298                                                                                                                                              
Segment 8    292                                                                                                                                              
Segment 1    100                                                                                                                                              
Segment 6    100                                                                                                                                              
Segment 5    100

Explore core protein functions:
   replicon_type                                           function  count
0      Segment 2                            Virulence factor PB1-F2    100
1      Segment 2  RNA-dependent RNA polymerase catalytic core PB...    101
2      Segment 1           RNA-dependent RNA polymerase PB2 subunit    100
3      Segment 3            RNA-dependent RNA polymerase PA subunit    100
4      Segment 2                                    PB1-N40 protein    101
5      Segment 3                                    PA-N182 protein    100
6      Segment 3                                    PA-N155 protein    100
7      Segment 5                               Nucleocapsid protein    100
8      Segment 8                             Nuclear export protein     97
9      Segment 8  Non-structural protein 1, interferon antagonis...    100
10     Segment 6                              Neuraminidase protein    100
11     Segment 4  Mature hemagglutinin N-terminal receptor bindi...    100
12     Segment 4  Mature hemagglutinin C-terminal membrane fusio...    100
13     Segment 7                                   Matrix protein 1    100
14     Segment 7                        M42 alternative ion channel     98
15     Segment 7                                     M2 ion channel    100
16     Segment 8           Hypothetical host adaptation protein NS3     95
17     Segment 3                   Host mRNA degrading protein PA-X     66
18     Segment 4                            Hemagglutinin precursor    100
"""
# breakpoint()
print("\nAnalyze protein counts per GTO file.")
protein_counts_per_file = analyze_protein_counts_per_file(
    prot_df, core_functions, aux_functions, selected_functions
)

# breakpoint()
print("\nExplore 'replicon_type' counts.")
print(prot_df['replicon_type'].value_counts())

# breakpoint()
print("\nShow all ['replicon_type', 'function'] combo counts.")
print_replicon_func_count(prot_df)

# breakpoint()
print('\nCloser look at "core" protein functions.')
print_replicon_func_count(prot_df, functions=core_functions)

# breakpoint()
print('\nCloser look at "auxiliary" protein functions.')
print_replicon_func_count(prot_df, functions=aux_functions)

# breakpoint()
print('\nCloser look at "selected" protein functions.')
print_replicon_func_count(prot_df, functions=selected_functions)
# EDA end ----------------------------------------


# Assign segments
# breakpoint()
print("\nAssign canonical segments.")
prot_df = assign_segments(prot_df, data_version=DATA_VERSION, use_core=True, use_aux=True)
prot_df.to_csv(output_dir / 'protein_assigned_segments.csv', sep=',', index=False)

# Apply basic filters
print("\nApply basic filters.")
prot_df = apply_basic_filters(prot_df)
prot_df.to_csv(output_dir / 'protein_filtered_basic.csv', sep=',', index=False)

# Handle duplicates
print("\nHandle protein sequence duplicates.")
prot_df = handle_duplicates(prot_df, print_eda=False)

# Clean protein sequences
breakpoint()
print("\nExplore ambiguities in protein sequences.")
org_cols = prot_df.columns
prot_df = analyze_protein_ambiguities(prot_df)
ambig_cols = [c for c in prot_df.columns if c not in org_cols]
ambig_df = prot_df[prot_df['has_ambiguities'] == True]
ambig_df = ambig_df[['assembly_id', 'brc_fea_id', seq_col_name] + ambig_cols]
cols_print = ['assembly_id', 'brc_fea_id'] + ambig_cols
print(f'Protein sequences with ambiguous chars: {ambig_df.shape[0]}')
print(ambig_df[cols_print])

print("\nSummarize of ambiguities in protein sequences.")
sa = summarize_ambiguities(prot_df)
pprint(sa)
print(f"Total protein seqs with terminal stops: {sa['terminal_stops']} ({sa['percent_terminal_stops']:.2f}%)")
print(f"Total protein seqs with internal stops: {sa['internal_stops']} ({sa['percent_internal_stops']:.2f}%)")
print("Non-standard residue distribution (excluding terminal stops):")
for aa, details in sa['non_standard_residue_summary'].items():
    meaning = details['meaning'] if aa != '*' else 'Premature stop or error'
    print(f"  {aa}: occurred in {details['count']} seqs ({meaning})")

# Prepare for ESM-2
print("\nPrepare protein sequences for ESM-2.")
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
    problematic_seqs_df.to_csv(output_dir / 'problematic_protein_seqs.csv', sep=',', index=False)

# Final duplicate counts
"""
num_files  total_cases
        2          138
        3           17
        5            7
        4            6
        6            4
        9            2
        10           1
        12           1
        141          1
        16           1
        53           1
        30           1
        13           1
        43           1
        15           1
Explanation.
- consider num_files=2 count=138.
    There are 138 distinct protein sequences that are duplicated (appear at
    least 2 times) across two files AND do not appear in any other file.
- consider num_files=141 count=1.
    There is 1 distinct protein sequence that is duplicated (appears at
    least 2 times) across 141 files AND do not appear in any other file.
"""
print("\nFinal duplicate counts.")
prot_dups = (
    prot_df[prot_df.duplicated(subset=[seq_col_name], keep=False)]
    .sort_values(seq_col_name)
    .reset_index(drop=True)
)
dup_counts = prot_dups.groupby(seq_col_name).agg(num_files=('file', 'nunique')).reset_index()
print(dup_counts['num_files'].value_counts().reset_index(name='total_cases'))

# Save final data
print("\nSave final protein data.")
prot_df.to_csv(output_dir / 'protein_final.csv', sep=',', index=False)
print(f'prot_df final: {prot_df.shape}')
print(f"Unique protein sequences: {prot_df[seq_col_name].nunique()}")
aa = print_replicon_func_count(prot_df, more_cols=['canonical_segment'])
aa.to_csv(output_dir / 'protein_final_segment_mappings_stats.csv', sep=',', index=False)
print(prot_df['canonical_segment'].value_counts())

total_timer.display_timer()
print('\nDone!')