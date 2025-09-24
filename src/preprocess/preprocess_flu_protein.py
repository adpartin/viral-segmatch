"""
Preprocess protein data from GTO files for Flu A.
"""
import json
import sys
from pathlib import Path
from pprint import pprint
from tqdm import tqdm

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

total_timer = Timer()

# Config
VIRUS_NAME = 'flu_a'
DATA_VERSION = 'July_2025'

# Define paths
main_data_dir = project_root / 'data'
raw_data_dir = main_data_dir / 'raw' / 'Anno_Updates' / DATA_VERSION
quality_gto_dir = raw_data_dir / 'bunya-from-datasets' / 'Quality_GTOs'
output_dir = main_data_dir / 'processed' / VIRUS_NAME / DATA_VERSION # processed_data_dir
output_dir.mkdir(parents=True, exist_ok=True)

print(f'main_data_dir:   {main_data_dir}')
print(f'raw_data_dir:    {raw_data_dir}')
print(f'quality_gto_dir: {quality_gto_dir}')
print(f'output_dir:      {output_dir}')

seq_col_name = 'prot_seq'

# Define core and auxiliary functions
if DATA_VERSION == 'April_2025':
    core_functions = [
        'RNA-dependent RNA polymerase',
        'Pre-glycoprotein polyprotein GP complex',
        'Nucleocapsid protein'
    ]
    aux_functions = [
        'Bunyavirales mature nonstructural membrane protein (NSm)',
        'Bunyavirales small nonstructural protein NSs',
        'Phenuiviridae mature nonstructural 78-kD protein',
    ]
elif DATA_VERSION == 'Feb_2025':
    core_functions = [
        'RNA-dependent RNA polymerase (L protein)',
        'Pre-glycoprotein polyprotein GP complex (GPC protein)',
        'Nucleocapsid protein (N protein)'
    ]
    aux_functions = [
        'Bunyavirales mature nonstructural membrane protein (NSm protein)',
        'Bunyavirales small nonstructural protein (NSs protein)',
        'Phenuiviridae mature nonstructural 78-kD protein',
        'Small Nonstructural Protein NSs (NSs Protein)',
    ]
else:
    raise ValueError(f'Unknown data_version: {DATA_VERSION}.')


def validate_protein_counts(
    df: pd.DataFrame,
    core_only: bool = False,
    verbose: bool = False
    ) -> None:
    """Ensure each assembly_id has ≤3 core proteins if core_only=True.

    This func is specific to preprocessing validation, tightly coupled
    to core_functions and assembly IDs (keep it in this script).
    """
    if core_only:
        df = df[df['function'].isin(core_functions)]
    for aid, grp in df.groupby('assembly_id'):
        n_proteins = len(grp)
        if core_only and n_proteins > 3:
            raise ValueError(
                f"assembly_id {aid} has {n_proteins} core proteins, expected ≤3.\n"
                f"Files: {grp['file'].unique()}\n"
                f"Functions: {grp['function'].tolist()}"
            )
        if verbose:
            print(f"assembly_id {aid}: {n_proteins} {'core' if core_only else 'total'} "
                  f"proteins, Functions: {grp['function'].tolist()}")
    return True


def get_protein_data_from_gto(gto_file_path: Path) -> pd.DataFrame:
    """
    Extract protein data and metadata from a GTO file into a DataFrame.
    
    - The 'id' in 'features' contains values like: fig|1316165.15.CDS.3, fig|11588.3927.CDS.1, etc.
    - These are PATRIC (BV-BRC) feature IDs: fig|<genome_id>.<feature_type>.<feature_number>
    - The genome_id (e.g., 1316165.15) is the internal genome identifier.
    - This 'id' can be used to trace the feature to its source genome in PATRIC/GTOs.
    - This is a not GenBank accession number (e.g., NC_038733.1, OL987654.1).

    - The 'location' in 'features' contains: [[ <segment_id>, <start>, <strand>, <end>]]
    - For example: "location": [[ "NC_086346.1", "70", "+", "738" ]]
    - It means: this feature is located on segment NC_086346.1 (positive strand), from
        nucleotide 70 to 738.
    - Note that the first str in 'location' (i.e., NC_086346.1) is a GenBank-style accession,
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
    # 'replicon_type': segment label (e.g., "[Large/Medium/Small] RNA Segment")
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


def aggregate_protein_data_from_gto_files(gto_dir: Path) -> pd.DataFrame:
    """Aggregate protein data from all GTO files."""
    dfs = []
    gto_files = sorted(gto_dir.glob('*.qual.gto'))
    for fpath in tqdm(gto_files, desc='Aggregating protein data from GTO files'):
        dfs.append(get_protein_data_from_gto(fpath))
    return pd.concat(dfs, axis=0).reset_index(drop=True)


def assign_segment_using_core_proteins(
    prot_df: pd.DataFrame,
    data_version: str = 'April_2025'
    ) -> pd.DataFrame:
    """Assign canonical segments (L, M, S) based on core protein functions.
    
    Each tripartite Bunyavirales genome segment (replicon) encodes a distinct core protein:

    1. L segment → encodes RNA-dependent RNA polymerase (L protein, or RdRp)
    2. M segment → encodes the Pre-Glycoprotein polyprotein (GPC), cleaved into Gn and Gc
    3. S segment → encodes the Nucleocapsid (N) protein

    Note! In bipartite Bunyavirales (2-segment), The Small RNA Segment can encode
        N and GPC proteins (e.g., Arenaviridae).

    These three proteins (L, GPC, N) are present in nearly all Bunyavirales genomes.

    - The L protein functions as the viral RNA-dependent RNA polymerase (RdRp)
        and responsible for both replication and transcription of the viral RNA.
    - The polyprotein (Pre-glycoprotein polyprotein GP complex) is cleaved into
        two envelope glycoproteins, Gn and Gc, which are located on the surface
        of the virus and are crucial for attachment to and entry into host cells.
    - The nucleocapsid protein N encapsidates the viral RNA genome to form
        ribonucleoprotein complexes (RNPs) and plays a vital role in viral RNA
        replication and transcription.

    We define canonical segment labels (L, M, S) by mapping known functions
        to these segments.

    Mapping is conditional: only apply if function and replicon type match
        expected biological patterns.

    The code show function-to-canonical_segment mappings conditional on 'replicon_type':
    function -> canonical_segment (condition)
    'RNA-dependent RNA polymerase' -> 'L' (replicon_type = ['Large RNA Segment'/'Segment One'])
    'Pre-glycoprotein polyprotein GP complex' -> 'M' (replicon_type = ['Medium RNA Segment''])
    'Nucleocapsid protein' -> 'S' (replicon_type = ['Small RNA Segment'/'Segment Three'])    
    """
    prot_df = prot_df.copy()

    # Core protein function-to-segment mappings by data_version
    # (function, replicon_type) → canonical_segment
    core_function_segment_maps = {
        'April_2025': [
            {'function': 'RNA-dependent RNA polymerase', 'replicon_type': 'Large RNA Segment', 'core_segment': 'L'},
            {'function': 'RNA-dependent RNA polymerase', 'replicon_type': 'Segment One', 'core_segment': 'L'},
            {'function': 'Pre-glycoprotein polyprotein GP complex', 'replicon_type': 'Medium RNA Segment', 'core_segment': 'M'},
            # {'function': 'Pre-glycoprotein polyprotein GP complex', 'replicon_type': 'Segment Two', 'core_segment': 'M'},
            {'function': 'Nucleocapsid protein', 'replicon_type': 'Small RNA Segment', 'core_segment': 'S'},
            {'function': 'Nucleocapsid protein', 'replicon_type': 'Segment Three', 'core_segment': 'S'},
        ],
        'Feb_2025': [
            {'function': 'RNA-dependent RNA polymerase (L protein)', 'replicon_type': 'Large RNA Segment', 'core_segment': 'L'},
            {'function': 'RNA-dependent RNA polymerase (L protein)', 'replicon_type': 'Segment One', 'core_segment': 'L'},
            {'function': 'Pre-glycoprotein polyprotein GP complex (GPC protein)', 'replicon_type': 'Medium RNA Segment', 'core_segment': 'M'},
            # {'function': 'Pre-glycoprotein polyprotein GP complex (GPC protein)', 'replicon_type': 'Segment Two', 'core_segment': 'M'},
            {'function': 'Nucleocapsid protein (N protein)', 'replicon_type': 'Small RNA Segment', 'core_segment': 'S'},
            {'function': 'Nucleocapsid protein (N protein)', 'replicon_type': 'Segment Three', 'core_segment': 'S'},
        ],
        # Add other versions, e.g., 'Month_YYYY': [...]
    }

    assert data_version in core_function_segment_maps, f'Unknown data_version: {data_version}.'
    core_map = pd.DataFrame(core_function_segment_maps.get(data_version))
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
    df = print_replicon_func_count(prot_df, more_cols=['core_segment'], drop_na=False)
    # df.to_csv(output_dir / 'core_mappings_eda.csv', sep=',', index=False)

    prot_df = prot_df.drop(columns=['core_segment'])
    return prot_df


def assign_segment_using_aux_proteins(
    prot_df: pd.DataFrame,
    data_version: str = 'April_2025'
    ) -> pd.DataFrame:
    """Assign canonical segments (L, M, S) based on auxiliary protein functions.    

    Some additional protein functions (e.g., NSs, NSm, 78-kD) are consistently
    matched with single segment (replicon_type) across all isolates in our dataset.

    These are non-structural proteins but can be mapped confidently:

    - NSs proteins → Small RNA Segment → assign as 'S'
    - NSm, 78-kD proteins → Medium RNA Segment → assign as 'M'

    Approach:
    - Use auxiliary function-to-segment mapping for unassigned proteins.
    - Do NOT assign segments if segment-function mapping is ambiguous.

    Take closer look at auxiliary protein functions.
    replicon_type                                               function   count
    Medium RNA Segment  Bunyavirales mature nonstructural membrane pro...    111
    Small RNA Segment   Bunyavirales small nonstructural protein (NSs ...    163
    Medium RNA Segment   Phenuiviridae mature nonstructural 78-kD protein    165
    Small RNA Segment       Small Nonstructural Protein NSs (NSs Protein)    433

    Medium RNA Segment   Phenuiviridae mature nonstructural 78-kD protein    165
    Small RNA Segment        Bunyavirales small nonstructural protein NSs    604
    Medium RNA Segment  Bunyavirales mature nonstructural membrane pro...    111 
    """
    prot_df = prot_df.copy()

    # Auxiliary protein function-to-segment mappings by data_version
    # (function, replicon_type) → canonical_segment
    # This set can be expanded once we validate other functions (Movement protein, Z protein, etc.).
    aux_function_segment_maps = {
       'April_2025': [
            {'function': 'Bunyavirales mature nonstructural membrane protein (NSm)',
             'replicon_type': 'Medium RNA Segment',
             'aux_segment': 'M'
            },
            {'function': 'Bunyavirales small nonstructural protein NSs',
             'replicon_type': 'Small RNA Segment',
             'aux_segment': 'S'
            },
            {'function': 'Phenuiviridae mature nonstructural 78-kD protein',
             'replicon_type': 'Medium RNA Segment',
             'aux_segment': 'M'
            },
       ],       
       'Feb_2025': [
            {'function': 'Bunyavirales mature nonstructural membrane protein (NSm protein)',
             'replicon_type': 'Medium RNA Segment',
             'aux_segment': 'M'
            },
            {'function': 'Bunyavirales small nonstructural protein (NSs protein)',
             'replicon_type': 'Small RNA Segment',
             'aux_segment': 'S'
            },
            {'function': 'Phenuiviridae mature nonstructural 78-kD protein',
             'replicon_type': 'Medium RNA Segment',
             'aux_segment': 'M'
            },
            {'function': 'Small Nonstructural Protein NSs (NSs Protein)',
             'replicon_type': 'Small RNA Segment',
             'aux_segment': 'S'
            },
       ],
    }

    assert data_version in aux_function_segment_maps, f'Unknown data_version: {data_version}.'
    aux_map = pd.DataFrame(aux_function_segment_maps.get(data_version))
    # print(aux_map)
    
    # Merge (assign temp canonical segments where function-replicon pairs match)
    # how='left': keeps all rows, assigns NaN where mapping doesn't apply
    prot_df = prot_df.merge(aux_map, on=['function', 'replicon_type'], how='left')
    # print(prot_df.iloc[:,-4:])

    # Assign aux_segment to canonical_segment
    mask = prot_df['canonical_segment'].isna() & prot_df['aux_segment'].notna()
    prot_df.loc[mask, 'canonical_segment'] = prot_df.loc[mask, 'aux_segment']
    # print(prot_df.iloc[:,-4:])

    # Diagnostics
    print_replicon_func_count(prot_df, functions=aux_functions)
    df = print_replicon_func_count(prot_df, more_cols=['aux_segment'], drop_na=False)
    # df.to_csv(output_dir / 'axu_mappings_eda.csv', sep=',', index=False)

    prot_df = prot_df.drop(columns=['aux_segment'])
    return prot_df


def assign_segments(
    prot_df: pd.DataFrame,
    data_version: str = 'April_2025',
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
    df = print_replicon_func_count(
        prot_df,
        functions=core_functions + aux_functions,
        more_cols=['canonical_segment'],
        drop_na=False)
    df.to_csv(output_dir / 'seg_mapped_core_and_aux_eda.csv', sep=',', index=False)
    
    # print('\nAvailable [replicon_type, function] combos for "core" and "auxiliary" protein mapping:')
    # print_replicon_func_count(prot_df, functions=core_functions + aux_functions)
    # print('\nAll [replicon_type, function] combos and assigned segments:')
    # df = print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)

    unmapped = prot_df[prot_df['canonical_segment'].isna()]
    print(f"\nUnmapped segments: {len(unmapped)}")
    print_replicon_func_count(unmapped)
    
    return prot_df


def apply_basic_filters(prot_df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic filters to protein data."""
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
    print_eda: str = False
    ) -> pd.DataFrame:
    """Handle protein sequence duplicates."""
    seq_col_name = 'prot_seq'

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
print(f"\nAggregate protein data from {len(sorted(quality_gto_dir.glob('*.qual.gto')))} GTO files.")
prot_df = aggregate_protein_data_from_gto_files(quality_gto_dir)
print(f'prot_df: {prot_df.shape}')

# Save initial data
prot_df.to_csv(output_dir / 'protein_agg_from_GTOs.csv', sep=',', index=False)
prot_df_no_seq = prot_df[prot_df[seq_col_name].isna()]
print(f'Records with missing protein sequence: {prot_df_no_seq.shape}')
prot_df_no_seq.to_csv(output_dir / 'protein_agg_from_GTO_missing_seqs.csv', sep=',', index=False)

# Check unique BRC feature IDs
print(f"\nTotal brc_fea_id:  {prot_df['brc_fea_id'].count()}")
print(f"Unique brc_fea_id: {prot_df['brc_fea_id'].nunique()}")


# EDA start ----------------------------------------
"""
Intermediate EDA

Explore all 'replicon_type' entires (segments):
Medium RNA Segment    3905
Small RNA Segment     2593
Large RNA Segment     1798
Unassigned             165
Segment One             61
Segment Two             59
Segment Four            59
Segment Three           45

Explore core protein functions:
replicon_type                                   function  count
Large RNA Segment           RNA-dependent RNA polymerase   1623
Segment One                 RNA-dependent RNA polymerase     61
Small RNA Segment           RNA-dependent RNA polymerase      4 TODO Note 1 below (dropped)
-----------------
Medium RNA Segment Pre-glycoprotein polyprotein GP complex 1344
Small RNA Segment  Pre-glycoprotein polyprotein GP complex  104 TODO Note 2 below (dropped)
Segment Two        Pre-glycoprotein polyprotein GP complex   59 TODO Note 4 below (temp. dropped)
-----------------
Small RNA Segment                   Nucleocapsid protein   1432
Segment Three                       Nucleocapsid protein     45 TODO Note 3 below (kept)

Notes:
1. [Small RNA Segment, RNA-dependent RNA polymerase] - This doesn't make sense. Later dropped on 'quality' (Poor)
2. [Small RNA Segment, Pre-glycoprotein polyprotein] - Probably comes from from bipartite Bunyas (Jim confirmed).
    We drop it since it's ambiguous in our L/M/S labeling framework (it violates the assumption that 'S'
    implies N protein).
3. [Segment Three, Nucleocapsid protein (N protein)] - This must be a tripartite Bunya (Jim confirmed).
    This should be safe to combine with Small RNA Segment (Jim confirmed).
4. [Segment Two, Pre-glycoprotein polyprotein] - This could be M in tripartite genomes (would keep)
    or S in bipartite genomes (would drop). Segment Two is taxonomically ambiguous (not enough info
    to resolve without consulting taxon-specific rules.)
"""
print("\nExplore 'replicon_type' counts.")
print(prot_df['replicon_type'].value_counts())

print("\nShow all ['replicon_type', 'function'] combo counts.")
print_replicon_func_count(prot_df)

print('\nCloser look at "core" protein functions.')
print_replicon_func_count(prot_df, functions=core_functions)

print('\nCloser look at "auxiliary" protein functions.')
print_replicon_func_count(prot_df, functions=aux_functions)
# EDA end ----------------------------------------


# Assign segments
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
print('\nDone!') 