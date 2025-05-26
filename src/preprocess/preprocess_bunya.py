"""
TODO Questions:
1. Can we map pssm_df['family'] to prot_df? Can this allow us to determine if it's 2-segment or 3-segment virus?
2. For modeling, do we really need to know what segment that it (i.e., S, M, L)?
3. We define a column 'canonical_segment'? Should it be just 'segment'?

Plots:
* boxplots of protein length per protein
* histogram of canonical segments in the dataset (show proteins/functions)

protein_filtered.csv
file: GTO file name (each GTO file contains rna/dna and protein sequences, including various metadata for a given viral istolate)
assembly_prefix: GCA (GenBank) or GCF (RefSeq)
assembly_id: assembly version of the genome
brc_fea_id: feature ID for the protein seq (from BV-BRC)
fcuntion: function of the protein (e.g., "RNA-dependent RNA polymerase")
prot_seq: the protein sequence
length: length of the protein sequence
genbank_ctg_id: the GenBank accession number for the contig
replicon_type: type of replicon (segment) the protein is associated with (e.g., "Large RNA Segment")
"""

import os
import json
import re
import sys
from pathlib import Path
from pprint import pprint
from enum import Enum
from typing import Dict, List, Union

# Third-party libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
print(f'project_root: {project_root}')

from src.utils.timer_utils import Timer
from src.utils.protein_utils import (
    analyze_protein_ambiguities,
    prepare_sequences_for_esm2,
    summarize_ambiguities,
    print_replicon_func_count
)

total_timer = Timer()

## Settings
process_pssm = False
process_dna = False
process_protein = True
assign_segment_using_core_proteins = True
assign_segment_using_aux_proteins = True
apply_basic_filters = True
remove_ambig_prot_seqs = True

## Config
# task_name = 'bunya_processed'
main_data_dir = project_root / 'data'
# data_version = 'Feb_2025'
data_version = 'April_2025'
raw_data_dir = main_data_dir / 'raw/Anno_Updates' / data_version
quality_gto_dir = raw_data_dir / 'bunya-from-datasets/Quality_GTOs'

virus_name = 'bunya'
processed_data_dir = main_data_dir / 'processed' / virus_name / data_version / 'old'

# output_dir = main_data_dir / task_name / data_version
output_dir = processed_data_dir
os.makedirs(output_dir, exist_ok=True)

print(f'main_data_dir:   {main_data_dir}')
print(f'raw_data_dir:    {raw_data_dir}')
print(f'quality_gto_dir: {quality_gto_dir}')
print(f'output_dir:      {output_dir}')

# Core protein definitions (used for validation)
core_funcs = {
    'L': 'RNA-dependent RNA polymerase',
    'M': 'Pre-glycoprotein polyprotein GP complex',
    'S': 'Nucleocapsid protein'
}


def extract_assembly_info(file_name):
    """ Extracts assembly prefix and assembly ID from a file name. """
    match = re.match(r'^(GCA|GCF)_(\d+\.\d+)', file_name)
    if match:
        return match.group(1), match.group(2)
    else:
        return None, None


class SequenceType(Enum):
    DNA = 'dna'
    PROTEIN = 'prot_seq'


def classify_dup_groups(
    df: pd.DataFrame,
    seq_col_name: Union[SequenceType, str]
    ) -> pd.DataFrame:
    """
    Classify duplicate sequences based on assembly IDs, record IDs, and prefixes.

    Args:
        df: DataFrame containing sequence data
        seq_col_name: Column name for sequences, must be either 'dna' or 'prot_seq'

    Returns:
        DataFrame summarizing duplicate groups, including classification case.

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

    # Determine the record ID column (source-specific identifier)
    record_col = {
        SequenceType.DNA: 'genbank_ctg_id',
        SequenceType.PROTEIN: 'brc_fea_id'
    }[seq_col_name]

    dup_info = []

    # df.groupby(seq_col_name) retrieves the rows with the same sequence
    # (i.e., sequence duplicates)
    for seq, grp in df.groupby(seq_col):
        files = list(grp['file'])                # e.g., GCA_031497195.1.qual.gto
        assembly_ids = list(grp['assembly_id'])  # e.g., 031497195.1
        record_ids = list(grp[record_col])       # genbank_ctg_id or brc_fea_id
        prefixes = list(grp['assembly_prefix'])  # GCA or GCF

        jj = grp[['file', 'assembly_id', 'assembly_prefix', 'genbank_ctg_id',
            'replicon_type', 'length', seq_col]]

        case = None
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

    df = pd.DataFrame(dup_info).sort_values('case').reset_index(drop=True)
    return df


def enforce_single_file(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """ Keep GCF_* over GCA_* for each assembly_id.
    TODO log those to file
    """
    keep_rows = []
    for aid, grp in df.groupby('assembly_id'):
        files = grp['file'].unique()
        if len(files) == 1:
            keep_rows.append(grp)
            continue
        # Multiple files: prioritize GCF_* if present
        gcf_files = [f for f in files if f.startswith('GCF_')]
        if gcf_files:
            # Keep only the first GCF file
            chosen_fname = gcf_files[0]
            chosen_file_df = grp[grp['file'] == chosen_fname]
        else:
            # No GCF, keep the first file
            chosein_fname = files[0]
            chosen_file_df = grp[grp['file'] == chosein_fname]
        if verbose:
            print(f"assembly_id {aid}: Multiple files {files}, kept {chosen_fname}")
        keep_rows.append(chosen_file_df)
    return pd.concat(keep_rows, ignore_index=True)


def validate_protein_counts(
    df: pd.DataFrame,
    core_only: bool = False,
    verbose: bool = False
    ) -> None:
    """ Ensure each assembly_id has ≤3 core proteins if core_only=True.
    TODO log those to file
    """
    if core_only:
        df = df[df['function'].isin(core_funcs.values())]
    for aid, grp in df.groupby('assembly_id'):
        n_proteins = len(grp)
        if core_only and n_proteins > 3:
            print(f"Warning: assembly_id {aid} has {n_proteins} core proteins, expected ≤3.")
            print(f"Files: {grp['file'].unique()}")
            print(f"Functions: {grp['function'].tolist()}")
            raise ValueError(f"assembly_id {aid} has >3 core proteins.")
        if verbose:
            print(f"assembly_id {aid}: {n_proteins} {'core' if core_only else 'total'} \
                proteins, Functions: {grp['function'].tolist()}")



# =======================================================================
# DNA/RNA sequence data
# =======================================================================

if process_dna:

    def get_dna_data_from_gto(gto_file_path: Path):
        """
        Extract dna/rna data and metadata from a GTO file and return it as a DataFrame.

        The 'id' in 'contigs' contains values like: NC_038733.1, OL774846.1, etc.
        These are GenBank accession numbers, corresponding to individual RNA segments.        
        """
        # file_path = quality_gto_dir / 'GCF_031497195.1.qual.gto'

        ## Load GTO file
        with open(gto_file_path, 'r') as file:
            gto = json.load(file)

        ## Extract data from 'contigs'
        # Each item in 'contigs' is a dict with keys: 'id', 'replicon_type', 'replicon_geometry', 'dna'
        # ctg is a list of dicts, each representing a 'contigs' item
        ctg = []
        for d in gto['contigs']:
            ctg.append({k: d[k] for k in sorted(d.keys())})

        # Aggregate the 'contigs' items into a DataFrame
        df = []
        # contigs_columns contains available data items from 'contigs' key
        contigs_columns = ['id', 'replicon_type', 'replicon_geometry', 'dna']
        for i, item in enumerate(ctg):
            df.append([item[f] if f in item else np.nan for f in contigs_columns])
        df = pd.DataFrame(df, columns=contigs_columns)
        df['length'] = df['dna'].apply(lambda x: len(x) if isinstance(x, str) else 0)

        ## Extract additional metadata from GTO file
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

        df = df.rename(columns={'id': 'genbank_ctg_id'})
        return df


    def aggregate_dna_data_from_gto_files(gto_dir: Path):
        """
        Aggregate dna data from all GTO files in the specified directory.
        """
        dfs = []
        gto_files = sorted(gto_dir.glob('*.qual.gto'))
        for fpath in tqdm(gto_files, desc='Aggregating dna (rna) data from GTO files'):
            df = get_dna_data_from_gto(gto_file_path=fpath)
            dfs.append(df)

        dna_df = pd.concat(dfs, axis=0).reset_index(drop=True)
        return dna_df

  
    # Aggregate DNA data from all GTO files
    print(f"\nAggregate dna (rna) from all {len(sorted(quality_gto_dir.glob('*.qual.gto')))} .qual.gto files.")
    dna_df = aggregate_dna_data_from_gto_files(quality_gto_dir)
    print(f'dna_df all samples {dna_df.shape}')
    print(f'dna_df columns: {dna_df.columns.tolist()}')

    # Save all samples
    dna_df.to_csv(output_dir / 'dna_agg_from_GTOs.csv', sep=',', index=False)

    # Save samples that missing seq data
    dna_df_no_seq = dna_df[dna_df['dna'].isna()]
    print(f'DNA with missing sequence {dna_df_no_seq.shape}')
    dna_df_no_seq.to_csv(output_dir / 'dna_missing_seqs.csv', sep=',', index=False)

    # Check that all GenBank accession numbers are unique.
    print('\nCheck that all GenBank accession numbers are unique.')
    print(f"Total accession ids (col 'genbank_ctg_id'):  {dna_df['genbank_ctg_id'].count()}")
    print(f"Unique accession ids (col 'genbank_ctg_id'): {dna_df['genbank_ctg_id'].nunique()}")

    # Ambiguous DNA Data. Remove sequences with too many unknown bases ('N' in
    # DNA or 'X' in protein).
    def summarize_dna_qc(df, seq_col: str='dna') -> pd.DataFrame:
        """ Perform quality control (QC) on DNA sequences in a DataFrame. """
        df = df.copy()
        ambig_codes = set("NRYWSKMBDHV")  # IUPAC ambiguous nucleotide codes

        qc = []
        for idx, seq in df[seq_col].items():
            if not isinstance(seq, str) or len(seq) == 0:
                qc.append((0, 0.0, 0, 0.0))  # Handle empty or missing sequences
                continue

            seq_length = len(seq)
            ambig_count = sum(1 for base in seq if base in ambig_codes)
            ambig_frac = ambig_count / seq_length
            gc_content = (seq.count('G') + seq.count('C')) / seq_length
            
            qc.append((ambig_count, ambig_frac, seq_length, gc_content))

        columns = ['ambig_count', 'ambig_frac', 'length', 'gc_content']
        qc_df = pd.DataFrame(qc, columns=columns).reset_index(drop=True)
        df_combined = pd.concat([df, qc_df], axis=1)
        return df_combined


    dna_qc = summarize_dna_qc(dna_df, seq_col='dna')

    # There are 5 samples (April 2025) with ambiguous_frac of > 0.1.
    # TODO! How should we filter this?
    print(dna_qc[:3])
    print(dna_qc['ambig_frac'].value_counts().sort_index(ascending=False))

    # Most segments follow the "[Large/Medium/Small] RNA Segment" naming.
    # Some segments follow the "Segment [One/Two/Three/Four]" naming.
    # Some segments have missing values
    # We assign a canonical segment label (implemented in protein data processing).
    print(dna_df['replicon_type'].value_counts(dropna=False))

    """
    TODO. We made a lot more progress in handling duplicates in protein data.
    We can revisit dna duplicates later.

    --------
    Case 1:
    --------
    Sequences (same), Assembly IDs (same), Accession IDs (different)

    This means that the same genomic segment has been assigned different accession numbers
    within the same assembly. These are likely technical duplicates.

    Example:
    file                        assembly_id     Accession IDs    dna
    GCA_002831345.1.qual.gto    002831345.1     NC_038733.1      AAAACAGTAGTGTACCG...
    GCF_002831345.1.qual.gto    002831345.1     OL123456.1       AAAACAGTAGTGTACCG...    

    --------
    Case 2:
    --------
    Sequences (same), Assembly IDs (different), Accession Numbers (different)

    This suggests the same seq appears in multiple genome assemblies, likely from different isolates.

    Example:
    file                        assembly_id     Accession Number    dna
    GCA_002831345.1.qual.gto    002831345.1     NC_038733.1         AAAACAGTAGTGTACCG...
    GCA_018595275.1.qual.gto    018595275.1     OL987654.1          AAAACAGTAGTGTACCG...

    =========================
    Results:
    =========================
    Most duplicates appear in only 2 files.
    We have 3 seqs that appear in 3 files, and 3 seqs that appear in 5 files.
    num_files	total_cases
    2	        1706
    3	        3 
    5	        3
    """
    seq_col_name = 'dna'
    dna_dups = (
        dna_df[dna_df.duplicated(subset=[seq_col_name], keep=False)]
        .sort_values(seq_col_name)
        .reset_index(drop=True).copy()
    ) 
    dna_dups.to_csv(output_dir / 'dna_all_duplicates.csv', sep=',', index=False)

    print(f"Duplicates on 'dna': {dna_dups.shape}")
    print(dna_dups[:4])

    # Count in how many unique files appears each duplicated sequence
    # This helps identify if a sequence appears across multiple files or within the same
    dup_counts = dna_dups.groupby(seq_col_name).agg(num_files=('file', 'nunique')).reset_index()
    print(dup_counts['num_files'].value_counts().reset_index(name='total_cases'))

    dup_summary = classify_dup_groups(dna_dups, SequenceType.DNA)
    case1 = dup_summary[dup_summary['case'] == 'Case 1']
    case2 = dup_summary[dup_summary['case'] == 'Case 2']
    case_other = dup_summary[dup_summary['case'] == 'Other']
    print(f"Case 1: {case1.shape[0]} sequences")
    print(f"Case 2: {case2.shape[0]} sequences")
    print(f"Case Other: {case_other.shape[0]} sequences")

    show_cols = [c for c in dup_summary.columns if c not in
        [SequenceType.DNA.value, SequenceType.PROTEIN.value]]
    print(f'Case 1:\n{case1[:3][show_cols]}')
    print(f'Case 2:\n{case2[:3][show_cols]}')



# =======================================================================
# Protein sequence data
# =======================================================================

if process_protein:

    def get_protein_data_from_gto(gto_file_path: Path) -> pd.DataFrame:
        """
        Extract protein data and metadata from GTO file and return as a DataFrame.

        - The 'id' in 'features' contains values like: fig|1316165.15.CDS.3, fig|11588.3927.CDS.1, etc.
        - These are PATRIC (BV-BRC) feature IDs: fig|<genome_id>.<feature_type>.<feature_number>
        - The genome_id (e.g., 1316165.15) is the internal genome identifier.
        - This 'id' can be used to trace the feature to its source genome in PATRIC/GTOs.
        - This is a not GenBank accession number (e.g., NC_038733.1, OL987654.1).

        - The 'location' in 'features' contains: [[ <segment_id>, <start>, <strand>, <end>]]
        - For example: "location": [[ "NC_086346.1", "70", "+", "738" ]]
        - It means: this feature is located on segment NC_086346.1 (positive strand), from
        - nucleotide 70 to 738.
        - Note that the first str in 'location' (i.e., NC_086346.1) is a GenBank-style accession,
            used by NCBI for nucleotide entries.
        """
        # gto_file_path = quality_gto_dir / 'GCF_031497195.1.qual.gto'

        # Load GTO file
        with open(gto_file_path, 'r') as file:
            gto = json.load(file)

        # Create genbank_ctg_id-to-replicon_type mapping using 'contigs'
        # This mapping is used to assign the replicon_type to protein seq
        # Note there is a link between 'features' items and 'contigs' items.
        # The link is embedded in:
        #   the first str in 'location' on the 'features' side
        #   the 'id' on the 'contigs' side
        # Iterate over 'contigs' items in GTO (if available)
        # (each item is a dict with keys: 'id', 'dna', 'replicon_type', 'replicon_geometry')
        # 'id': genbank_ctg_id (e.g., NC_086346.1)
        # 'replicon_type': segment label (e.g., "[Large/Medium/Small] RNA Segment")
        segment_map: Dict[str, str] = {
            contig['id']: contig.get('replicon_type', 'Unassigned')
            for contig in gto.get('contigs', [])
        }

        ## Extract data from 'features'
        # features_columns contains data items available in 'features' key
        features_columns = ['id', 'type', 'function', 'protein_translation', 'location']
        rows = []

        for fea_dict in gto['features']:
            row = {k: fea_dict.get(k, None) for k in features_columns}
            row['length'] = len(fea_dict['protein_translation']) if 'protein_translation' in fea_dict else 0

            # Segment ID from 'location' field (e.g., [[ "NC_086346.1", "70", "+", "738" ]]
            if isinstance(fea_dict.get('location'), list) and len(fea_dict['location']) > 0:
                genbank_ctg_id = fea_dict['location'][0][0]
            else:
                genbank_ctg_id = None
            row['genbank_ctg_id'] = genbank_ctg_id
            row['replicon_type'] = segment_map.get(genbank_ctg_id, 'Unassigned')

            # Family
            if ('family_assignments' in fea_dict) and (len(fea_dict['family_assignments'][0]) > 2):
                row['family'] = fea_dict['family_assignments'][0][1]
            else:
                row['family'] = pd.NA

            rows.append(row)

        df = pd.DataFrame(rows)

        ## Extract additional metadata from GTO file
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
        """
        Aggregate protein data from all GTO files in the specified directory.
        """
        dfs = []
        gto_files = sorted(gto_dir.glob('*.qual.gto'))
        for fpath in tqdm(gto_files, desc='Aggregating protein data from GTO files'):
            # if 'GCF_013086535.1' in str(fpath):
            #     # TODO analyze isolate that has two GPC proteins (this shouldn't happen)
            #     print(fpath)
            df = get_protein_data_from_gto(gto_file_path=fpath)
            # print(df[['file', 'genbank_ctg_id', 'type', 'brc_fea_id', 'function']])
            dfs.append(df)

        prot_df = pd.concat(dfs, axis=0).reset_index(drop=True)
        return prot_df


    # Aggregate protein data from all GTO files
    print(f"\nAggregate protein from all {len(sorted(quality_gto_dir.glob('*.qual.gto')))} .qual.gto files.")
    prot_df = aggregate_protein_data_from_gto_files(quality_gto_dir)
    print(f'prot_df: {prot_df.shape}')

    # Save initial data
    prot_df.to_csv(output_dir / 'protein_agg_from_GTOs.csv', sep=',', index=False)
    prot_df_no_seq = prot_df[prot_df['prot_seq'].isna()]
    print(f'prot_df with missing sequences {prot_df_no_seq.shape}')
    prot_df_no_seq.to_csv(output_dir / 'protein_missing_seqs.csv', sep=',', index=False)

    # Check unique BRC feature IDs
    print('\nCheck that all BRC feature IDs are unique.')
    print(f"Total brc_fea_id:  {prot_df['brc_fea_id'].count()}")
    print(f"Unique brc_fea_id: {prot_df['brc_fea_id'].nunique()}")

    """
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
        to resolve without consulting taxon-specific rules).
    """
    print("\nExplore 'replicon_type' counts.")
    print(prot_df['replicon_type'].value_counts())

    if data_version == 'Feb_2025':
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

    elif data_version == 'April_2025':
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

    print("\nShow all ['replicon_type', 'function'] combination counts.")
    print_replicon_func_count(prot_df)

    print('\nCloser look at "core" protein functions.')
    print_replicon_func_count(prot_df, functions=core_functions)

    print('\nCloser look at "auxiliary" protein functions.')
    print_replicon_func_count(prot_df, functions=aux_functions)

    # Create a placeholder for canonical_segment
    prot_df['canonical_segment'] = pd.NA


    if assign_segment_using_core_proteins:
        """
        Step 1: Assign canonical segments (L, M, S) based on core protein functions.
        
        Each tripartite bunyavirales genome segment (replicon) encodes a distinct core protein:

        1. L segment → encodes RNA-dependent RNA polymerase (L protein, or RdRp)
        2. M segment → encodes the Pre-Glycoprotein polyprotein (GPC), cleaved into Gn and Gc
        3. S segment → encodes the Nucleocapsid (N) protein

        Note! In bipartite bunyaviruses (2-segment), The Small RNA Segment can encode N and GPC
        proteins (e.g., Arenaviridae).

        These three proteins (L, GPC, N) are present in nearly all Bunyavirales genomes.

        - The L protein functions as the viral RNA-dependent RNA polymerase (RdRp) and
            responsible for both replication and transcription of the viral RNA.
        - The polyprotein (Pre-glycoprotein polyprotein GP complex) is cleaved into two envelope
            glycoproteins, Gn and Gc, which are located on the surface of the virus and are crucial
            for attachment to and entry into host cells.
        - The nucleocapsid protein N encapsidates the viral RNA genome to form ribonucleoprotein
            complexes (RNPs) and plays a vital role in viral RNA replication and transcription.

        We define canonical segment labels (L, M, S) by mapping known functions to these segments.

        Mapping is conditional: only apply if function and replicon type match expected biological patterns.

        The code show function-to-canonical_segment mappings conditional on 'replicon_type':
        function -> canonical_segment (condition)
        'RNA-dependent RNA polymerase' -> 'L' (replicon_type = ['Large RNA Segment'/'Segment One'])
        'Pre-glycoprotein polyprotein GP complex' -> 'M' (replicon_type = ['Medium RNA Segment''])
        'Nucleocapsid protein' -> 'S' (replicon_type = ['Small RNA Segment'/'Segment Three'])
        """
        print('\n==== Step 1: Assign canonical segments (L/M/S) to "core" protein mapping ====')
        # breakpoint()

        # Define function-to-segment mappings for core proteins
        # (function, replicon_type) → canonical_segment
        if data_version == 'Feb_2025':
            core_function_segment_map = pd.DataFrame([
                # Canonical segment 'L'
                {'function': 'RNA-dependent RNA polymerase (L protein)',
                'replicon_type': 'Large RNA Segment',
                'canonical_segment': 'L'
                },
                {'function': 'RNA-dependent RNA polymerase (L protein)',
                'replicon_type': 'Segment One',
                'canonical_segment': 'L'
                },
                # Canonical segment 'M'
                {'function': 'Pre-glycoprotein polyprotein GP complex (GPC protein)',
                'replicon_type': 'Medium RNA Segment',
                'canonical_segment': 'M'
                },
                # {'function': 'Pre-glycoprotein polyprotein GP complex (GPC protein)',
                # 'replicon_type': 'Segment Two',
                # 'canonical_segment': 'M'
                # },
                # Canonical segment 'S'
                {'function': 'Nucleocapsid protein (N protein)',
                'replicon_type': 'Small RNA Segment',
                'canonical_segment': 'S'
                },
                {'function': 'Nucleocapsid protein (N protein)',
                'replicon_type': 'Segment Three',
                'canonical_segment': 'S'
                },
            ])

        elif data_version == 'April_2025':
            core_function_segment_map = pd.DataFrame([
                # Canonical segment 'L'
                {'function': 'RNA-dependent RNA polymerase',
                'replicon_type': 'Large RNA Segment',
                'canonical_segment': 'L'
                },
                {'function': 'RNA-dependent RNA polymerase',
                'replicon_type': 'Segment One',
                'canonical_segment': 'L'
                },
                # Canonical segment 'M'
                {'function': 'Pre-glycoprotein polyprotein GP complex',
                'replicon_type': 'Medium RNA Segment',
                'canonical_segment': 'M'
                },
                # {'function': 'Pre-glycoprotein polyprotein GP complex',
                # 'replicon_type': 'Segment Two',
                # 'canonical_segment': 'M'
                # },
                # Canonical segment 'S'
                {'function': 'Nucleocapsid protein',
                'replicon_type': 'Small RNA Segment',
                'canonical_segment': 'S'
                },
                {'function': 'Nucleocapsid protein',
                'replicon_type': 'Segment Three',
                'canonical_segment': 'S'
                },
            ])

        map_seg_col = 'core_seg_mapped'
        core_function_segment_map = core_function_segment_map.rename(columns={'canonical_segment': map_seg_col})
        print(core_function_segment_map)

        # Merge: Assign temp canonical segments where function-replicon pairs match
        prot_df = prot_df.merge(
            core_function_segment_map,
            on=['function', 'replicon_type'],
            how='left'  # keep all rows, NaN where mapping doesn't apply
        )
        # print(prot_df.iloc[:,-4:])

        # Assign core_seg_mapped to canonical_segment
        mask_core = prot_df['core_seg_mapped'].notna()
        prot_df.loc[mask_core, 'canonical_segment'] = prot_df.loc[mask_core, 'core_seg_mapped']
        # print(prot_df.iloc[:,-4:])

        # Diagnostics
        # breakpoint()
        print('\nAvailable [replicon_type, function] combos for "core" protein mapping:')
        print_replicon_func_count(prot_df, functions=core_functions)

        print('\nAll [replicon_type, function] combos and assigned segments:')
        df = print_replicon_func_count(prot_df, more_cols=[map_seg_col], drop_na=False)
        df.to_csv(output_dir / 'seg_mapped_core_eda.csv', sep=',', index=False)

        # breakpoint()
        unmapped_core = prot_df[prot_df[map_seg_col].isna()]
        print(f'\nUnmapped segments (after "core" protein mapping only): {len(unmapped_core)}')
        print_replicon_func_count(unmapped_core) # consider all
        # print_replicon_func_count(unmapped_core, core_functions) # consider core only


    if assign_segment_using_aux_proteins:
        """
        Step 2: Extend canonical segment assignments for well-supported auxiliary proteins.

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
        print('\n==== Step 2: Assign canonical segments (L/M/S) to "auxiliary" proteins ====')
        # breakpoint()

        # Define function-to-segment mapping auxiliary proteins
        # (function, replicon_type) → canonical_segment
        # This set can be expanded once we validate other functions (Movement protein, Z protein, etc.).
        if data_version == 'Feb_2025':
            aux_function_segment_map = pd.DataFrame([
                {'function': 'Bunyavirales mature nonstructural membrane protein (NSm protein)',
                'replicon_type': 'Medium RNA Segment',
                'canonical_segment': 'M'
                },
                {'function': 'Bunyavirales small nonstructural protein (NSs protein)',
                'replicon_type': 'Small RNA Segment',
                'canonical_segment': 'S'
                },
                {'function': 'Phenuiviridae mature nonstructural 78-kD protein',
                'replicon_type': 'Medium RNA Segment',
                'canonical_segment': 'M'
                },
                {'function': 'Small Nonstructural Protein NSs (NSs Protein)',
                'replicon_type': 'Small RNA Segment',
                'canonical_segment': 'S'
                },
            ])

        elif data_version == 'April_2025':
            aux_function_segment_map = pd.DataFrame([
                {'function': 'Bunyavirales mature nonstructural membrane protein (NSm)',
                'replicon_type': 'Medium RNA Segment',
                'canonical_segment': 'M'
                },
                {'function': 'Bunyavirales small nonstructural protein NSs',
                'replicon_type': 'Small RNA Segment',
                'canonical_segment': 'S'
                },
                {'function': 'Phenuiviridae mature nonstructural 78-kD protein',
                'replicon_type': 'Medium RNA Segment',
                'canonical_segment': 'M'
                },
            ])

        map_seg_col = 'aux_seg_mapped'
        aux_function_segment_map = aux_function_segment_map.rename(
            columns={'canonical_segment': map_seg_col})

        # Merge: Assign temp. canonical segments where function-replicon pairs match
        prot_df = prot_df.merge(
            aux_function_segment_map, on=['function', 'replicon_type'],
            how='left'  # keep all rows, NaN where mapping doesn't apply
        )
        # print(prot_df.iloc[:,-4:])

        # Assign core_seg_mapped to canonical_segment
        mask_aux = prot_df['canonical_segment'].isna() & prot_df['aux_seg_mapped'].notna()
        prot_df.loc[mask_aux, 'canonical_segment'] = prot_df.loc[mask_aux, 'aux_seg_mapped']
        # print(prot_df.iloc[:,-4:])

        # Diagnostics
        print('\nAvailable [replicon_type, function] combos for "auxiliary" protein mapping:')
        print_replicon_func_count(prot_df, functions=aux_functions)

        print('\nAll [replicon_type, function] combos and assigned segments:')
        df = print_replicon_func_count(prot_df, more_cols=[map_seg_col], drop_na=False)
        df.to_csv(output_dir / 'seg_mapped_aux_eda.csv', sep=',', index=False)

        unmapped_aux = prot_df[prot_df[map_seg_col].isna()]
        print(f'\nUnmapped segments (after "aux" protein mapping only): {len(unmapped_aux)}')
        print_replicon_func_count(unmapped_aux) # consider all
        # print_replicon_func_count(unmapped_aux, aux_functions) # consider aux only

    # Diagnostics
    print('\nAvailable [replicon_type, function] combos for "core" and "auxiliary" protein mapping:')
    print_replicon_func_count(prot_df, functions=core_functions + aux_functions)

    print('\nAll [replicon_type, function] combos and assigned segments:')
    df = print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)

    df.to_csv(output_dir / 'seg_mapped_core_and_aux_eda.csv', sep=',', index=False)

    # Drop the helper columns
    prot_df = prot_df.drop(
        columns=[c for c in prot_df.columns if c in ['core_seg_mapped', 'aux_seg_mapped']]
        )


    # breakpoint()
    print('\nSave protein data before applying filtering.')
    prot_df.to_csv(output_dir / 'protein_before_filtering.csv', sep=',', index=False)



    if apply_basic_filters:
        print('\n=============  Start basic filtering  =============')
        
        print('\n----- Drop rows where canonical_segment=NaN -----')
        # breakpoint()
        print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)
        unassgined_df = prot_df[prot_df['canonical_segment'].isna()].sort_values('canonical_segment', ascending=False)
        prot_df = prot_df[prot_df['canonical_segment'].notna()].reset_index(drop=True)
        print(f'Total unassigned (NaN) canonical_segment: {unassgined_df.shape}')
        print(f'Remaining prot_df:                        {prot_df.shape}')
        print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)


        print('\n----- Keep rows where type=CDS (drops mat_peptide) -----')
        # breakpoint()
        print_replicon_func_count(prot_df, more_cols=['canonical_segment', 'type'], drop_na=False)
        non_cds = prot_df[prot_df['type'] != 'CDS'].sort_values('type', ascending=False)
        prot_df = prot_df[prot_df['type'] == 'CDS'].reset_index(drop=True)
        print(f'Total non-CDS samples: {non_cds.shape}')
        print(f'Remaining prot_df:     {prot_df.shape}')
        # print(prot_df['type'].value_counts())
        non_cds.to_csv(output_dir / 'protein_non_cds.csv', sep=',', index=False)
        print_replicon_func_count(prot_df, more_cols=['canonical_segment', 'type'], drop_na=False)


        print('\n----- Drop rows where quality=Poor -----')
        # breakpoint()
        print_replicon_func_count(prot_df, more_cols=['canonical_segment', 'quality'], drop_na=False)
        poor_df = prot_df[prot_df['quality'] == 'Poor'].sort_values('quality', ascending=False)
        prot_df = prot_df[prot_df['quality'] != 'Poor'].reset_index(drop=True)
        print(f'Total Poor quality samples: {poor_df.shape}')
        print(f'Remaining prot_df:          {prot_df.shape}')
        # print(prot_df['quality'].value_counts())
        poor_df.to_csv(output_dir / 'protein_poor_quality.csv', sep=',', index=False)
        print_replicon_func_count(prot_df, more_cols=['canonical_segment', 'quality'], drop_na=False)


        print('\n----- Drop rows where replicon_type=Unassigned -----')
        # breakpoint()
        print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)
        unk_df = prot_df[prot_df['replicon_type'] == 'Unassigned'].sort_values('replicon_type', ascending=False)
        prot_df = prot_df[prot_df['replicon_type'] != 'Unassigned'].reset_index(drop=True)
        print(f'Total `Unassigned` samples: {unk_df.shape}')
        print(f'Remaining prot_df:          {prot_df.shape}')
        # print(prot_df['replicon_type'].value_counts())
        unk_df.to_csv(output_dir / 'protein_unassigned_replicons.csv', sep=',', index=False)
        print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)


        # Save filtered protein data
        prot_df.to_csv(output_dir / 'protein_filtered_basic.csv', sep=',', index=False)


    print('\n=============  Handle protein sequence duplicates  =============')
    if (output_dir / 'protein_filtered_basic.csv').exists():
        print('Load filtered (basic) protein data.')
        prot_df = pd.read_csv(output_dir / 'protein_filtered_basic.csv')

    """
    Explore duplicates in protein data.
    """
    seq_col_name = 'prot_seq'
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
    print(prot_dups[:4][['file', 'assembly_id', 'genbank_ctg_id', 'replicon_type',
        'brc_fea_id', 'function', 'canonical_segment']])

    # Count in how many unique files each duplicated sequence appears
    """
    num_files  total_cases
            2         1266
            1           68
            3           50
            4           26
            5           13
            6            3
            9            2
           10            1
           12            1
            7            1
           30            1
           13            1
           43            1
           16            1
    Explanation.
    - consider num_files=1 count=68.  This means:
        There are 68 distinct protein sequences that are duplicated (appear at
        least 2 times) inside one single file AND do not appear in any other file.
    """
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


    # Enforce one file per assembly_id before duplicate handling
    print('\n---- Enforce one file per assembly_id ----')
    print(f'prot_df before: {prot_df.shape}')
    prot_df = enforce_single_file(prot_df)
    print(f'prot_df after:  {prot_df.shape}')

    # Validate protein counts after filtering
    print('\n---- Validate protein counts ----')
    validate_protein_counts(prot_df, core_only=True)   # Check core proteins
    validate_protein_counts(prot_df, core_only=False)  # Check all proteins


    """
    Find dups with the same [prot_seq, assembly_id, function, replicon_type]
    Same isolate duplicates.
    """
    # breakpoint()
    print('\n---- FIND dups on [prot_seq, assembly_id, function, replicon_type] ----')
    dup_cols = [seq_col_name, 'assembly_id', 'function', 'replicon_type']
    gca_gcf_dups = (
        prot_df[prot_df.duplicated(subset=dup_cols, keep=False)]
        .sort_values(dup_cols)
        .reset_index(drop=True).copy()
    )
    gca_gcf_dups.to_csv(output_dir / 'protein_dups_same_isolate.csv', sep=',', index=False)

    print(f'gca_gcf_dups: {gca_gcf_dups.shape}')
    print(f"dups unique seqs: {gca_gcf_dups['prot_seq'].nunique()}")
    print(gca_gcf_dups[:4][dup_cols + ['assembly_prefix']])

    gca_gcf_dups_eda = gca_gcf_dups.groupby(dup_cols).agg(
        num_assembly_prefixes=('assembly_prefix', 'nunique'),
        assembly_prefixes=('assembly_prefix', lambda x: list(set(x))),
        brc_fea_ids=('brc_fea_id', lambda x: list(set(x))),
        ).sort_values(dup_cols).reset_index()
    gca_gcf_dups_eda.to_csv(output_dir / 'protein_dups_same_isolate_eda.csv', sep=',', index=False)
    del dup_cols, # gca_gcf_dups, gca_gcf_dups_eda


    """
    Drop GCA sample if the same sequence appears also in GCF with the same
    ['prot_seq', 'assembly_id', 'function', 'replicon_type']
    """
    print('\n---- DROP GCA if GCF exists for the same [prot_seq, assembly_id, function, replicon_type] ----')
    dup_cols = [seq_col_name, 'assembly_id', 'function', 'replicon_type']
    
    # 1) Find all groups (on dup_cols) where both GCA and GCF appear
    groups = (
        prot_df
        .groupby(dup_cols)['assembly_prefix']
        .apply(set)
        .reset_index(name='prefixes')
    )

    # 2) Select only those groups that contain both 'GCA' and 'GCF'
    # Note! This is the same as gca_gcf_dups_eda
    dups_to_fix = groups[
        groups['prefixes'].apply(lambda s: {'GCA','GCF'}.issubset(s))
    ][dup_cols].reset_index(drop=True)

    # print(gca_gcf_dups_eda.shape)
    # print(dups_to_fix.shape)
    # print(dups_to_fix.equals(gca_gcf_dups_eda.iloc[:,:4]))

    # 3) Turn that (dup_cols combos) into a set of keys for fast lookup
    to_drop_keys = set( tuple(x) for x in dups_to_fix.values )

    # 4) Build a mask
    mask_with_gca_dups = prot_df['assembly_prefix'].eq('GCA')
    mask_with_drop_keys = prot_df.set_index(dup_cols).index.isin(to_drop_keys)
    mask_drop = mask_with_gca_dups & mask_with_drop_keys

    # 5) Drop those rows
    prot_df_wo_gca_dups = prot_df[~mask_drop].copy()
    print(f"Dropped {mask_drop.sum()} GCA rows. Remaining rows: {len(prot_df_wo_gca_dups)}")

    # 6) **Verify** no group still has both prefixes
    remaining_groups = (
        prot_df_wo_gca_dups
        .groupby(dup_cols)['assembly_prefix']
        .apply(set)
        .reset_index(name='prefixes')
    )
    bad = remaining_groups[
        remaining_groups['prefixes'].apply(lambda s: {'GCA','GCF'}.issubset(s))
    ]
    assert bad.empty, "⚠️ Filtering failed: some dups still have both GCA & GCF!"
    print("✔ All GCA dups removed where a matching GCF existed.")

    prot_df = prot_df_wo_gca_dups.copy()
    aa = print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)


    """
    Find dups within a GTO file (intra-file dups). This is related to num_occurrences >= num_files.
    """
    print('\n---- FIND dups within the same GTO file (intra-file dups) ----')
    dup_cols = [seq_col_name, 'file']
    same_file_dups = (
        prot_df[prot_df.duplicated(subset=dup_cols, keep=False)]
        .sort_values(dup_cols + ['brc_fea_id'])
        .reset_index(drop=True).copy()
    )
    same_file_dups.to_csv(output_dir / 'protein_dups_within_file.csv', sep=',', index=False)

    print(f'same_file_dups: {same_file_dups.shape}')
    print(f"dups unique seqs: {same_file_dups['prot_seq'].nunique()}")
    if not same_file_dups.empty:
        print(same_file_dups[:4][dup_cols + ['brc_fea_id']])

    same_file_dups_eda = same_file_dups.groupby(dup_cols).agg(
        num_brc_fea_ids=('brc_fea_id', 'nunique'),
        num_funcs=('function', 'nunique'),
        brc_fea_ids=('brc_fea_id', lambda x: list(set(x))),
        functions=('function', lambda x: list(set(x))),
        ).sort_values(dup_cols).reset_index()
    same_file_dups_eda.to_csv(output_dir / 'protein_dups_within_file_eda.csv', sep=',', index=False)
    del dup_cols, # same_file_dups, same_file_dups_eda


    """
    Find dups with different function annotations in the same file.
    That's a special case of intra-file duplicates.
    Note! These exacly match the same-file duplicates.
    """
    print('\n---- FIND dups functions annotations ----')
    dup_cols = [seq_col_name, 'file']

    dups_func_conflicts = (
        prot_df[prot_df.duplicated(subset=dup_cols, keep=False)]
        .sort_values(dup_cols + ['brc_fea_id'])
        .reset_index(drop=True).copy()
    )

    print(f'dups_func_conflicts: {dups_func_conflicts.shape}')
    print(f"dups unique seqs: {dups_func_conflicts['prot_seq'].nunique()}")
    print(dups_func_conflicts[:6][dup_cols + ['brc_fea_id']])

    dups_func_conflicts_eda = dups_func_conflicts.groupby(dup_cols).agg(
            num_funcs=('function', 'nunique'),
            num_replicons=('replicon_type', 'nunique'),
            functions=('function', lambda x: list(set(x))),
            replicons=('replicon_type', lambda x: list(set(x))),
        ).sort_values(dup_cols).reset_index()
    del dup_cols, # dups_func_conflicts, dups_func_conflicts_eda
    # print(same_file_dups.equals(dups_func_conflicts))



    print('\n=============  Cleaning protein sequences  =============')
    # Replace '*' with empty string in protein seqs (asterisk '*' denotes a stop codon)
    # print(f'\nRemove asterisks from prot_seq column.')
    # prot_df['prot_seq'] = prot_df['prot_seq'].apply(lambda x: x.replace('*', ''))

    print("\nExplore ambiguities in protein sequences.")
    org_cols = prot_df.columns
    prot_df = analyze_protein_ambiguities(prot_df)
    ambig_cols = [c for c in prot_df.columns if c not in org_cols]
    ambig_df = prot_df[prot_df['has_ambiguities'] == True]
    ambig_df = ambig_df[['assembly_id', 'brc_fea_id', 'prot_seq'] + ambig_cols]
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

    print("\nPrepare protein sequences for ESM-2.")
    max_internal_stops = 0.1
    max_x_residues = 0.1
    x_imputation = 'G'
    strip_terminal_stop = True
    prot_df, prob_seqs_df = prepare_sequences_for_esm2(
        prot_df,
        seq_column='prot_seq',
        output_column='esm2_ready_seq',
        x_imputation=x_imputation,
        max_internal_stops=max_internal_stops,
        max_x_residues=max_x_residues,
        strip_terminal_stop=strip_terminal_stop
    )
    if not prob_seqs_df.empty:
        prob_seqs_df.to_csv(output_dir / 'problematic_protein_seqs.csv', index=False)

    # ---------------------------------------------------------------


    # Final duplicate counts
    print('\n---- Final protein duplicates counts ----')
    """
    num_files  total_cases
            2          138
            3           17
            5            7
            4            6
            6            4
            9            2
           10            1
           12            1
          141            1
           16            1
           53            1
           30            1
           13            1
           43            1
           15            1
    Explanation.
    - consider num_files=2 count=138.
        There are 138 distinct protein sequences that are duplicated (appear at
        least 2 times) across two files AND do not appear in any other file.
    - consider num_files=141 count=1.
        There is 1 distinct protein sequence that is duplicated (appears at
        least 2 times) across 141 files AND do not appear in any other file.
    """
    prot_dups = (
        prot_df[prot_df.duplicated(subset=[seq_col_name], keep=False)]
        .sort_values(seq_col_name)
        .reset_index(drop=True).copy()
    )
    dup_counts = prot_dups.groupby(seq_col_name).agg(num_files=('file', 'nunique')).reset_index()
    print(dup_counts['num_files'].value_counts().reset_index(name='total_cases'))


    # Finally, save filtered protein data
    print('\n----- Save final protein data -----')
    print(f'prot_df: {prot_df.shape}')
    print(f'Unique protein sequences: {prot_df[seq_col_name].nunique()}')
    prot_df.to_csv(output_dir / 'protein_filtered.csv', sep=',', index=False)
    aa = print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)
    aa.to_csv(output_dir / 'protein_filtered_segment_mappings_stats.csv', sep=',', index=False)
    print(prot_df['canonical_segment'].value_counts())

# ----------------------------------------------------------------
# breakpoint()
total_timer.display_timer()
print('\nDone!')
