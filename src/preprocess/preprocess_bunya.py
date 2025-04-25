"""
TODO Questions:
1. Can we map pssm_df['family'] to prot_df? Can this allow us to determine if it's 2-segment or 3-segment virus?
2. For modeling, do we really need to know what segment that it (i.e., S, M, L)?
"""
import os
import json
import re
from collections import Counter
from pathlib import Path
from pprint import pprint
from time import time
from enum import Enum
from typing import Dict, List, Union

# Third-party libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

filepath = Path(__file__).parent # .py
# filepath = Path(os.path.abspath('')) # .ipynb
print(f'filepath: {filepath}')

# Settings
process_dna = False
# process_dna = True
process_protein = True
# assign_segment_using_core_proteins = False
assign_segment_using_core_proteins = True
# assign_segment_using_aux_proteins = False
assign_segment_using_aux_proteins = True

# task_name = 'Bunya-from-datasets'
task_name = 'bunya_processed'
data_dir = filepath / '../../data'
raw_data_dir = data_dir / 'raw'
quality_gto_dir = raw_data_dir / 'Bunya-from-datasets/Quality_GTOs'

output_dir = data_dir / task_name
os.makedirs(output_dir, exist_ok=True)

print(f'raw_data_dir:    {raw_data_dir}')
print(f'quality_gto_dir: {quality_gto_dir}')
print(f'output_dir:      {output_dir}')


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
        if len(set(assembly_ids)) == 1 and len(set(record_ids)) == 1 and set(prefixes) == {'GCA', 'GCF'}:
            case = 'Case 1'  # Same assembly, same record, GCA vs. GCF
        elif len(set(assembly_ids)) == 1 and len(set(record_ids)) > 1:
            case = 'Case 2'  # Same assembly, different records
        elif len(set(assembly_ids)) > 1 and len(set(record_ids)) > 1:
            case = 'Case 3'  # Different assemblies, different records
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


# =======================================================================
# Exploratory analysis of the Viral_PSSM.json.
# =======================================================================

"""
The annotation script reads from Viral_PSSM.json. If a protein has a segment assigned
to it, then it's used in defining the segment and making sure that segment is intact.

The Viral_PSSM.json file serves as a reference schema for the expected structure and
composition of segmented viral genomes, particularly within the order Bunyavirales
and a few other segmented RNA viruses (e.g., Arenaviridae, Phasmaviridae, etc.).

It is designed to inform genome annotation and quality assessment pipelines by describing:
1. What segments (replicons) are expected for each viral family.
2. What features (usually protein-coding genes) are expected on each segment.
3. What the expected lengths are for both segments and features.
4. Which feature belongs to which segment, based on curated domain knowledge and taxonomic conventions.

🧬 Biological Context
In segmented RNA viruses like those in Bunyavirales:
* The viral genome is split into 2 or more RNA segments, each encoding different viral proteins.
* Each segment is typically associated with a particular set of functional genes/proteins.
* Different viral families follow different segment-protein conventions. For example, some plant viruses have an extra "movement" protein.

🔹 Segment Entry
These entries describe the expected segment structure for a given viral family.
📌 Purpose: Establish expectations for which segments should exist and what lengths are reasonable.

🔹 Feature Entry
These entries describe the expected protein features (typically CDS or mat_peptide entries) on a given segment.
📌 Purpose:
* Define which features should be present on which segments.
* Validate whether the annotated proteins fall within reasonable length bounds.
* Flag proteins on the wrong segment, mislabeled, or missing.
"""

# Load the Viral_PSSM.json file
with open(raw_data_dir / 'Bunya-from-datasets/Viral_PSSM.json', 'r') as f:
    viral_pssm = json.load(f)

# Parse the structure into a flat DataFrame
records = []
for family, family_data in viral_pssm.items():
    segments = family_data.get('segments', {})
    features = family_data.get('features', {})
    
    for seg_name, seg_data in segments.items():
        seg_info = {
            'family': family,
            'segment_name': seg_name,
            'segment_min_len': seg_data.get('min_len'),
            'segment_max_len': seg_data.get('max_len'),
            'replicon_geometry': seg_data.get('replicon_geometry'),
        }
        records.append({**seg_info, 'type': 'segment'})

    for feature_name, feature_data in features.items():
        feature_info = {
            'family': family,
            'feature_name': feature_name,
            'feature_type': feature_data.get('feature_type'),
            'anno': feature_data.get('anno'),
            'segment': feature_data.get('segment'),
            'min_len': feature_data.get('min_len'),
            'max_len': feature_data.get('max_len'),
        }
        records.append({**feature_info, 'type': 'feature'})

# Convert to DataFrame
pssm_df = pd.DataFrame(records)
# pssm_df.to_csv(output_dir / 'viral_pssm.tsv', sep='\t', index=False)
pssm_df.to_csv(output_dir / 'viral_pssm.csv', sep=',', index=False)
# print(f'\npssm_df {pssm_df.shape}')
# print(f"\n{pssm_df['family'].value_counts()}")
# print(f"\n{pssm_df['segment'].value_counts()}")
# print(f"\n{pssm_df['segment_name'].value_counts()}")


# =======================================================================
# Exploratory analysis of the file triplet:
# contig_quality, feature_quality, qual.gto
# =======================================================================

ex_file = 'GCF_031497195.1'
# ex_file = 'GCA_000851025.1'

# contig_quality
print(f"\nTotal .contig_quality files: {len(sorted(quality_gto_dir.glob('*.contig_quality')))}")
cq = pd.read_csv(quality_gto_dir / f'{ex_file}.contig_quality', sep='\t')
print(cq)

# feature_quality
print(f"\nTotal .feature_quality files: {len(sorted(quality_gto_dir.glob('*.feature_quality')))}")
fq = pd.read_csv(quality_gto_dir / f'{ex_file}.feature_quality', sep='\t')
print(fq)

# qual.gto
print(f"n\Total .qual.gto files: {len(sorted(quality_gto_dir.glob('*.qual.gto')))}")
file_path = quality_gto_dir / f'{ex_file}.qual.gto'
with open(file_path, 'r') as file:
    gto = json.load(file)

print('\nGTO file keys:')
for k in sorted(gto.keys()):
    print(f'{k}: {type(gto[k])}')

# Explore 'features' in the GTO dict
print(f"\nTotal 'features' items: {len(gto['features'])}") # that's the number rows in feature_quality file
for d in gto['features']:
    print(sorted(d.keys())) # print sorted keys

# create a list of sorted dicts
fea = []
for d in gto['features']:
    fea.append({k: d[k] for k in sorted(d.keys())})

print("\nShow gto['features'][0]:")
pprint(gto['features'][0])

# The 'features' contains some (or all) of the info available in the .feature_quality file.
print("\nShow a few items of a single 'features' item:")
for i, item in enumerate(fea):
    print(f"{item['id']},  {item['type']},  {item['function']}")




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


    def aggregate_data_from_gto_files(gto_dir: Path):
        """
        Aggregate dna data from all GTO files in the specified directory.
        """
        dfs = []
        for i, fpath in enumerate(sorted(gto_dir.glob('*.gto'))):
            df = get_dna_data_from_gto(gto_file_path=fpath)
            dfs.append(df)

        dna_df = pd.concat(dfs, axis=0).reset_index(drop=True)
        return dna_df

  
    # Aggregate DNA data from all GTO files
    print(f"\nTotal .qual.gto files: {len(sorted(quality_gto_dir.glob('*.qual.gto')))}")
    dna_df = aggregate_data_from_gto_files(quality_gto_dir)
    print(f'DNA all samples {dna_df.shape}')
    print(dna_df.columns.tolist())

    # Save all samples
    # dna_df.to_csv(output_dir / 'dna_data_all.tsv', sep='\t', index=False)
    dna_df.to_csv(output_dir / 'dna_data_all.csv', sep=',', index=False)

    # Save samples that missing seq data
    dna_df_no_seq = dna_df[dna_df['dna'].isna()]
    print(f'DNA with missing sequence {dna_df_no_seq.shape}')
    # dna_df_no_seq.to_csv(output_dir / 'dna_data_no_seq.tsv', sep='\t', index=False)
    dna_df_no_seq.to_csv(output_dir / 'dna_data_no_seq.csv', sep=',', index=False)

    """
    Check that all GenBank accession numbers are unique.
    """
    # All segment IDs are unique (no duplicates).
    print('\nCheck that all GenBank accession numbers are unique.')
    print(f"Total accession ids (col 'genbank_ctg_id'):  {dna_df['genbank_ctg_id'].count()}")
    print(f"Unique accession ids (col 'genbank_ctg_id'): {dna_df['genbank_ctg_id'].nunique()}")

    """
    Ambiguous DNA Data.
    Remove sequences with too many unknown bases ('N' in DNA or 'X' in protein).
    """
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

    # There is only one sample with ambiguous_frac of > 0.1.
    # TODO! How should we filter this?
    print(dna_qc[:3])
    print(dna_qc['ambig_frac'].value_counts().sort_index(ascending=False))

    """
    Explore replicon_type.
    """
    # Most segments follow the "[Large/Medium/Small] RNA Segment" naming.
    # Some segments are labeled as "Segment [One/Two/Three/Four]"
    # 311 entries have missing values
    # We address this by assigning a canonical segment label (the unassgined samples are dropped).
    print(dna_df['replicon_type'].value_counts(dropna=False))

    """
    Check how often each duplicated sequence appears across different files.
    This step helps understand whether we should keep and remove certain duplicates.

    ===============
    If 2 duplicates
    ===============

    --------
    Case 1:
    --------
    Sequecnes (same), Assembly IDs (same), Accession Numbers (same), Versions (different)

    This is likely a technical duplicate (e.g., GCA/GCF versions of the same assembly for
    a given viral isolate).
    Some GTO files may contain both draft (GCA) and reference (GCF) assemblies for the same
    viral isolate, leading to identical sequences appearing in different files.
    GCA (Genomic Contig Assembly):  A draft genome assembly, typically submitted first
    GCF (Genomic Complete Assembly):  A higher-quality, curated version of the same assembly

    Example:
    file                        assembly_id     Accession Number    dna
    GCA_002831345.1.qual.gto    002831345.1     NC_038733.1         AAAACAGTAGTGTACCG...
    GCF_002831345.1.qual.gto    002831345.1     NC_038733.1         AAAACAGTAGTGTACCG...

    GenBank Accession Number: NC_038733.1 (same RNA segment, same isolate)
    File names: GCA vs. GCF (same assembly ID, different versions)

    Conclusion:
    Keep the GCF version (it's typically a higher-quality). TODO confirmed with Carla

    --------
    Case 2:
    --------
    Sequences (same), Assembly IDs (same), Accession IDs (different), Versions (ignore)

    This means that the same genomic segment has been assigned different accession numbers
    within the same assembly. This can happen due to:
    Reannotation:  The sequence may have been re-submitted with updated metadata
    Redundant entries:  Multiple GenBank submissions within the same assembly

    Example:
    file                        assembly_id     Accession IDs    dna
    GCA_002831345.1.qual.gto    002831345.1     NC_038733.1      AAAACAGTAGTGTACCG...
    GCA_002831345.1.qual.gto    002831345.1     OL123456.1       AAAACAGTAGTGTACCG...

    Conclusion:
    These are likely technical duplicates.
    Keep only one — preferably the most recent or official accession number. TODO confirm with Jim

    --------
    Case 3:
    --------
    Sequences (same), Assembly IDs (different), Accession Numbers (different), Versions (ignore)

    This suggests the same sequence appears in multiple genome assemblies, likely from different isolates.

    Example:
    file                        assembly_id     Accession Number    dna
    GCA_002831345.1.qual.gto    002831345.1     NC_038733.1         AAAACAGTAGTGTACCG...
    GCA_018595275.1.qual.gto    018595275.1     OL987654.1          AAAACAGTAGTGTACCG...

    Conclusion:
    If the assemblies are from different viral isolates, this reflects true biological
    conservation — keep both.
    If they are somehow from the same isolate, consider keeping only the higher-quality version.

    =========================
    Result (ap):
    =========================
    Most duplicates appear in only 2 files.
    We have 3 seqs that appear in 3 files, and 3 seqs that appear in 5 files.
    num_files	count
    2	        1592
    3	        3 
    5	        3
    """
    dna_dups = dna_df[dna_df.duplicated(subset=['dna'], keep=False)].sort_values(['dna'])
    dna_dups.to_csv(output_dir / 'dna_all_duplicates.csv', sep=',', index=False)

    print(f"Duplicates on 'dna': {dna_dups.shape}")
    print(dna_dups[:4])

    # dna_dups.groupby('dna').agg(file=('file', 'unique')).reset_index().explode('file')

    # Count in how many unique files appears each duplicated sequence
    # This helps identify if a sequence appears across multiple files or within the same
    dup_counts = dna_dups.groupby('dna')['file'].nunique().reset_index(name='unq_files')
    print(dup_counts['unq_files'].value_counts().reset_index(name='total_cases'))

    """
    ✅ Summary of Duplicate Classifications.
    Case	Description	                                        Count	Suggested Action
    Case 1	Same assembly_id, same record_id, GCA/GCF versions  0	    ✅ Nothing to do
    Case 2	Same assembly_id, diff record_id	                1,524	⚠️ Technical dups — keep one
    Case 3	Diff assembly_id and record_id	                    74	    ✅ Likely true biological conservation — keep all
    """
    # breakpoint()
    dup_summary = classify_dup_groups(dna_dups, SequenceType.DNA)

    show_cols = [c for c in dup_summary.columns if c not in
        [SequenceType.DNA.value, SequenceType.PROTEIN.value]]
    case1 = dup_summary[dup_summary['case'] == 'Case 1']
    case2 = dup_summary[dup_summary['case'] == 'Case 2']
    case3 = dup_summary[dup_summary['case'] == 'Case 3']
    case_other = dup_summary[dup_summary['case'] == 'Other']
    print(f"Case 1: {case1.shape[0]} sequences")
    print(f"Case 2: {case2.shape[0]} sequences")
    print(f"Case 3: {case3.shape[0]} sequences")
    print(f"Case Other: {case_other.shape[0]} sequences")

    print(f'Case 2:\n{case2[:3][show_cols]}')
    print(f'Case 3:\n{case3[:3][show_cols]}')

    # Explore duplicate groups with more than 2 files
    dup_counts = dna_dups.groupby('dna')['file'].nunique().reset_index(name='num_files')
    multi_dups = dup_counts[dup_counts['num_files'] > 2] # > 2 dups
    multi_dup_records = dna_dups.merge(multi_dups[['dna']], on='dna') # Merge back to get full rows
    multi_dup_records = multi_dup_records.sort_values(['dna', 'assembly_id', 'assembly_prefix'])
    print(f'\nmulti_dup_records: {multi_dup_records.shape}')
    print(f'{multi_dup_records}')



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
            df = get_protein_data_from_gto(gto_file_path=fpath)
            dfs.append(df)

        prot_df = pd.concat(dfs, axis=0).reset_index(drop=True)
        return prot_df


    # Aggregate protein data from all GTO files
    print(f"\nAggregate protein from all {len(sorted(quality_gto_dir.glob('*.qual.gto')))} .qual.gto files.")
    prot_df = aggregate_protein_data_from_gto_files(quality_gto_dir)
    print(f'prot_df all samples {prot_df.shape}')
    print(f'prot_df columns: {prot_df.columns.tolist()}')

    # Save all samples
    prot_df.to_csv(output_dir / 'protein_data_all.csv', sep=',', index=False)
    
    # Save samples that missing seq data
    prot_df_no_seq = prot_df[prot_df['prot_seq'].isna()]
    print(f'prot_df with missing sequences {prot_df_no_seq.shape}')
    prot_df_no_seq.to_csv(output_dir / 'protein_data_missing_seqs.csv', sep=',', index=False)

    """
    Check that all (BRC) feature numbers are unique.
    """
    # All BRC feature IDs are unique (no duplicates).
    print('\nCheck that all BRC feature IDs are unique.')
    print(f"Total brc_fea_id:  {prot_df['brc_fea_id'].count()}")
    print(f"Unique brc_fea_id: {prot_df['brc_fea_id'].nunique()}")

    """
    Analyze segments (replicon_type)
    ----------------
    Explore all 'replicon_type' entires:
    Medium RNA Segment    3813
    Small RNA Segment     2512
    Large RNA Segment     1699
    Unassigned             152
    Segment One             61
    Segment Two             59
    Segment Four            59
    Segment Three           45

    Explore core protein functions:
    replicon_type                                               function  count
    Large RNA Segment           RNA-dependent RNA polymerase (L protein)   1524
    Segment One                 RNA-dependent RNA polymerase (L protein)     61
    Small RNA Segment           RNA-dependent RNA polymerase (L protein)      4 TODO Note 1 below (dropped)
    -----------------
    Medium RNA Segment Pre-glycoprotein polyprotein GP complex (GPC p...   1296
    Small RNA Segment  Pre-glycoprotein polyprotein GP complex (GPC p...    104 TODO Note 2 below (dropped)
    Segment Two        Pre-glycoprotein polyprotein GP complex (GPC p...     59 TODO Note 4 below (temp. dropped)
    -----------------
    Small RNA Segment                   Nucleocapsid protein (N protein)   1370
    Segment Three                       Nucleocapsid protein (N protein)     45 TODO Note 3 below (kept)
    
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

    Explore replicon_type='Segment [One/Two/Three/Four]':
    replicon_type                                           function  count
    Segment Four                 Fimoviridae uncharacterized protein      6
    Segment Four                                    Movement protein     53
    Segment One             RNA-dependent RNA polymerase (L protein)     61
    Segment Three                   Nucleocapsid protein (N protein)     45
    Segment Two    Pre-glycoprotein polyprotein GP complex (GPC p...     59
    """
    # breakpoint()
    print("\nExplore all 'replicon_type' counts.")
    print(prot_df['replicon_type'].value_counts())

    # breakpoint()
    print("\nExplore ['Segment [One/Two/Three/Four]', 'function'] counts")
    df = (
        prot_df[prot_df['replicon_type']
        .isin(['Segment One', 'Segment Two', 'Segment Three', 'Segment Four'])]
        .groupby(['replicon_type', 'function'])
        .size()
        .reset_index(name='count')
    )
    print(df)

    def print_replicon_func_count(
            df: pd.DataFrame,
            functions: List[str]=None,
            more_cols: List[str]=None,
            drop_na: bool=True,
        ):
        """ 
        Print counts of replicon_type and function combinations.
        Parameters:
            df (pd.DataFrame): DataFrame containing protein data.
            functions (List[str]): List of functions to filter by.
            more_cols (List[str]): Additional columns to include in the grouping.
            drop_na (bool): Whether to drop NA values from the grouping.
        """
        if functions is not None:
            df = df[df['function'].isin(functions)]

        basic_cols = ['replicon_type', 'function']
        col_list = basic_cols if more_cols is None else basic_cols + more_cols

        df = (
            df.groupby(col_list, dropna=drop_na)
            .size()
            .reset_index(name='count')
            .sort_values(['function', 'count'], ascending=False)
            .reset_index(drop=True)
        )
        print(df)

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

    print("\nShow all ['replicon_type', 'function'] combination counts.")
    print_replicon_func_count(prot_df)

    print('\nCloser look at "core" protein functions.')
    print_replicon_func_count(prot_df, functions=core_functions)

    print('\nCloser look at "auxiliary" protein functions.')
    print_replicon_func_count(prot_df, functions=aux_functions)

    # 0) start with a placeholder for canonical_segment
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
        - The polyprotein (Pre-glycoprotein polyprotein GP complex (GPC protein)) is cleaved into
            two envelope glycoproteins, Gn and Gc, which are located on the surface of the virus
            and are crucial for attachment to and entry into host cells.
        - The nucleocapsid protein N encapsidates the viral RNA genome to form ribonucleoprotein
            complexes (RNPs) and plays a vital role in viral RNA replication and transcription.

        We define canonical segment labels (L, M, S) by mapping known functions to these segments.

        Mapping is conditional: only apply if function and replicon type match expected biological patterns.

        The code show function-to-canonical_segment mappings conditional on 'replicon_type':
        function -> canonical_segment (condition)
        'RNA-dependent RNA polymerase (L protein)' -> 'L' (replicon_type = ['Large RNA Segment'/'Segment One'])
        'Pre-glycoprotein polyprotein GP complex (GPC protein)' -> 'M' (replicon_type = ['Medium RNA Segment''])
        'Nucleocapsid protein (N protein)' -> 'S' (replicon_type = ['Small RNA Segment'/'Segment Three'])

        After mapping:
        L      1585
        S      1415
        M      1296
        """
        print('\n==== Step 1: Assign canonical segments (L/M/S) to "core" protein mapping ====')
        # breakpoint()

        # Define function-to-segment mappings for core proteins
        # (function, replicon_type) → canonical_segment
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
        map_seg_col = 'core_seg_mapped'
        core_function_segment_map = core_function_segment_map.rename(
            columns={'canonical_segment': map_seg_col})

        # Merge: Assign temp. canonical segments where function-replicon pairs match
        prot_df = prot_df.merge(
            core_function_segment_map, on=['function', 'replicon_type'],
            how='left'  # keep all rows, NaN where mapping doesn't apply
        )

        # 1) Assign core_seg_mapped to canonical_segment
        mask_core = prot_df['core_seg_mapped'].notna()
        prot_df.loc[mask_core, 'canonical_segment'] = prot_df.loc[mask_core, 'core_seg_mapped']

        # Diagnostics
        # breakpoint()
        print('\nAvailable [replicon_type, function] combos for "core" protein mapping:')
        print_replicon_func_count(prot_df, functions=core_functions)

        print('\nAll [replicon_type, function] combos and assigned segments:')
        print_replicon_func_count(prot_df, more_cols=[map_seg_col], drop_na=False)

        df.to_csv(output_dir / 'prot_df_core_eda.csv', sep=',', index=False)

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
        """
        print('\n==== Step 2: Assign canonical segments (L/M/S) to "auxiliary" proteins ====')
        # breakpoint()

        # Define function-to-segment mapping auxiliary proteins
        # (function, replicon_type) → canonical_segment
        # This set can be expanded once we validate other functions (Movement protein, Z protein, etc.).
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
        map_seg_col = 'aux_seg_mapped'
        aux_function_segment_map = aux_function_segment_map.rename(
            columns={'canonical_segment': map_seg_col})

        # Merge: Assign temp. canonical segments where function-replicon pairs match
        prot_df = prot_df.merge(
            aux_function_segment_map, on=['function', 'replicon_type'],
            how='left'  # keep all rows, NaN where mapping doesn't apply
        )

        # 2) Assign core_seg_mapped to canonical_segment
        mask_aux = prot_df['canonical_segment'].isna() & prot_df['aux_seg_mapped'].notna()
        prot_df.loc[mask_aux, 'canonical_segment'] = prot_df.loc[mask_aux, 'aux_seg_mapped']

        # Diagnostics
        # breakpoint()
        print('\nAvailable [replicon_type, function] combos for "auxiliary" protein mapping:')
        print_replicon_func_count(prot_df, functions=aux_functions)

        print('\nAll [replicon_type, function] combos and assigned segments:')
        print_replicon_func_count(prot_df, more_cols=[map_seg_col], drop_na=False)

        df.to_csv(output_dir / 'tt_prot_df_aux_eda.csv', sep=',', index=False)

        # breakpoint()
        unmapped_aux = prot_df[prot_df[map_seg_col].isna()]
        print(f'\nUnmapped segments (after "aux" protein mapping only): {len(unmapped_aux)}')
        print_replicon_func_count(unmapped_aux) # consider all
        # print_replicon_func_count(unmapped_aux, aux_functions) # consider aux only

    # Diagnostics
    # breakpoint()
    print('\nAvailable [replicon_type, function] combos for "core" and "auxiliary" protein mapping:')
    print_replicon_func_count(prot_df, functions=core_functions + aux_functions)

    print('\nAll [replicon_type, function] combos and assigned segments:')
    print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)

    df.to_csv(output_dir / 'prot_df_core_and_aux_eda.csv', sep=',', index=False)

    # Drop the helper columns
    prot_df = prot_df.drop(
        columns=[c for c in prot_df.columns if c in ['core_seg_mapped', 'aux_seg_mapped']]
        )


    print('\n============= Start filtering =============')
    """
    Keep only CDS in prot_df (drops mat_peptide and RNA)
    """
    breakpoint()
    print('\n----- Keep rows where type=CDS -----')
    non_cds = prot_df[prot_df['type'] != 'CDS'].sort_values('type', ascending=False)
    prot_df = prot_df[prot_df['type'] == 'CDS'].reset_index(drop=True)
    print(f"Total non-CDS samples: {non_cds.shape}")
    # print(prot_df['type'].value_counts())
    non_cds.to_csv(output_dir / 'protein_data_non_cds.csv', sep=',', index=False)

    print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)

    """
    Drop 'Poor' quality samples
    """
    breakpoint()
    print('\n----- Drop rows where quality=Poor -----')
    poor_df = prot_df[prot_df['quality'] == 'Poor'].sort_values('quality', ascending=False)
    prot_df = prot_df[prot_df['quality'] != 'Poor'].reset_index(drop=True)
    print(f'Total Poor quality samples: {poor_df.shape}')
    # print(prot_df['quality'].value_counts())
    poor_df.to_csv(output_dir / 'protein_data_poor_quality.csv', sep=',', index=False)

    print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)

    """
    Drop segments (replicon_type) labeled 'Unassigned'
    """
    # breakpoint()
    print('\n----- Drop rows where replicon_type=Unassigned -----')
    unk_df = prot_df[prot_df['replicon_type'] == 'Unassigned'].sort_values('replicon_type', ascending=False)
    prot_df = prot_df[prot_df['replicon_type'] != 'Unassigned'].reset_index(drop=True)
    print(f'Total `Unassigned` samples: {unk_df.shape}')
    # print(prot_df['replicon_type'].value_counts())
    unk_df.to_csv(output_dir / 'protein_data_unassigned_replicons.csv', sep=',', index=False)

    print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)

    """
    Drop samples canonical_segment=NaN
    """
    breakpoint()
    print('\n----- Drop rows w/o assigned canonical segments -----')
    prot_df = prot_df[prot_df['canonical_segment'].notna()].reset_index(drop=True)

    print_replicon_func_count(prot_df, more_cols=['canonical_segment'], drop_na=False)


    # Save filtered protein data
    prot_df.to_csv(output_dir / 'protein_data_filtered.csv', sep=',', index=False)


    print('\n============= Handle protein sequence duplicates =============')
    prot_df = pd.read_csv(output_dir / 'protein_data_filtered.csv')

    """
    Explore duplicates in protein data.
    """
    breakpoint()
    seq_col_name = 'prot_seq'
    print_replicon_func_count(prot_df)

    # Get all duplicate protein sequences
    print(f'\nprot_df {prot_df.shape}')
    prot_dups = (
        prot_df[prot_df.duplicated(subset=[seq_col_name], keep=False)]
        .sort_values(seq_col_name)
        .reset_index(drop=True).copy()
    )
    prot_dups.to_csv(output_dir / 'protein_all_duplicates.csv', sep=',', index=False)
    print(f"Duplicates on '{seq_col_name}': {prot_dups.shape}")
    print(prot_dups[:4])

    # Count in how many unique files each duplicated sequence appears
    """
    num_files  count
            2   1152
            3     42
            4     21
            1     14
            5     11
            6      3
            9      2
           10      1
           12      1
            7      1
           30      1
           13      1
           43      1
           16      1
    Explanation.
    - consider num_files=1 count=14.  This means:
        There are 14 distinct protein sequences that are duplicated (appear at
        least 2 times) inside one single file AND do not appear in any other file.
    """
    print(f'\nCount how many unique files contain each duplicated sequence')
    # dup_counts = prot_dups.groupby(seq_col_name)['file'].nunique().reset_index(name='num_files')
    dup_counts = prot_dups.groupby(seq_col_name).agg(num_files=('file', 'nunique')).reset_index()
    print(dup_counts['num_files'].value_counts().reset_index(name='total_cases'))

    def explore_groupby(df, columns):
        seq_col_name = 'prot_seq'
        df = df.copy()
        grouped = df.groupby(columns)

        # grouped_iterator = iter(grouped)
        # first_key, first_group = next(grouped_iterator)

        for keys, grp in grouped:
            # print(f"Sequence: {seq[:50]}")  # print a slice of seq
            print('\nGroup (rows with this seq):')
            print(grp[[seq_col_name, 'file', 'assembly_id', 'genbank_ctg_id', 'replicon_type', 'brc_fea_id', 'function']])
            print(grp)
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
    dup_stats.to_csv(output_dir / 'protein_duplicates_stats.csv', sep=',', index=False)


    """
    Duplicate case: duplicate within a GTO file (intra-file dups)
    (related to num_occurrences >= num_files)
    TODO Jim found the problem!
    """
    print('\nDuplicate case -- Explore duplicates within the same GTO file (intra-file dups).')
    # # Method 1:
    # # Pulls out seqs where duplicates are only inside one file
    # # This method misses seqs that are duplicated in that file AND also appear
    # # somewhere else
    # df1 = dup_stats[dup_stats['num_files'] == 1].reset_index(drop=True).copy()
    # df1['files'] = df1['files'].apply(lambda x: x[0] if len(x) == 1 else x) 
    # print(f'df1 {df1.shape}')
    # print(f"df1 unique seqs {df1['prot_seq'].nunique()}")
    # print(df1[['num_occurrences', 'files', 'brc_fea_ids']])

    # # Method 2:
    # # Pulls out seqs that have ≥ 2 different feature IDs in the same file
    # # This method loses the metadata columns (this is solved in Method 3)
    # df2 = prot_dups.groupby(['prot_seq', 'file']).agg(
    #     num_occurrences=(seq_col_name, 'count'),
    #     brc_fea_ids=('brc_fea_id', lambda x: list(set(x))),
    #     num_brc_fea_ids=('brc_fea_id', 'nunique'),
    #     ).sort_values('num_occurrences', ascending=False).reset_index()
    # df2 = df2[df2['num_brc_fea_ids'] >= 2]
    # print(f'df2 {df2.shape}')
    # print(f"df2 unique seqs {df2['prot_seq'].nunique()}")
    # print(df2[['num_occurrences', 'brc_fea_ids']])

    # Method 3:
    # Pulls out every row that belongs to a (prot_seq, file) pair appearing ≥ 2 times
    # This method can include seqs that are duplicated across several files
    dup_cols = [seq_col_name, 'file']

    same_file_dups = (
        prot_df[prot_df.duplicated(subset=dup_cols, keep=False)]
        .sort_values(dup_cols + ['brc_fea_id'])
        .reset_index(drop=True).copy()
    )
    same_file_dups.to_csv(output_dir / 'protein_duplicates_within_file.csv', sep=',', index=False)
    print(f'dups: {same_file_dups.shape}')
    print(f"dups unique seqs: {same_file_dups['prot_seq'].nunique()}")
    print(same_file_dups[dup_cols + ['brc_fea_id']])
    
    same_file_dups_eda = same_file_dups.groupby(dup_cols).agg(
        num_brc_fea_ids=('brc_fea_id', 'nunique'),
        num_funcs=('function', 'nunique'),
        brc_fea_ids=('brc_fea_id', lambda x: list(set(x))),
        functions=('function', lambda x: list(set(x))),
        ).sort_values(dup_cols).reset_index()
    same_file_dups_eda.to_csv(output_dir / 'protein_duplicates_within_file_eda.csv', sep=',', index=False)

    breakpoint()
    del dup_cols, same_file_dups, same_file_dups_eda
    

    """
    Duplicate case: duplicates with the same assembly_id (i.e. same isolate), function, replicon_type
    combined case1 and case2 (see func classify_dup_groups())
    Jim confirmed we can drop GCAs.
    """
    print('\nDuplicate case -- Explore duplicates with the same [assembly_id, function, replicon_type] combos.')
    # dup_tech = dup_stats[dup_stats['num_assemblies'] == 1].reset_index(drop=True).copy()
    # dup_tech.to_csv(output_dir / 'protein_duplicates_same_assembly.csv', sep=',', index=False)
    # print(dup_tech['num_assembly_prefixes'].value_counts().reset_index(name='count'))

    # # GCA/GCF duplicate pairs
    # gca_gcf_dups = dup_stats[(dup_stats['num_assemblies'] == 1) & (dup_stats['num_assembly_prefixes'] == 2)].sort_values(seq_col_name).reset_index(drop=True)
    # gca_gcf_dups['assembly_id'] = gca_gcf_dups['assembly_ids'].apply(lambda x: x[0])
    # print(gca_gcf_dups[['prot_seq', 'assembly_id', 'brc_fea_ids', 'assembly_prefixes']])
    # print(f'Total GCA-GCF pairs: {gca_gcf_dups.shape}')
    # gca_gcf_dups.to_csv(output_dir / 'protein_duplicates_gca_gcf_pairs.csv', sep=',', index=False)
    
    dup_cols = [seq_col_name, 'assembly_id', 'function', 'replicon_type']

    gca_gcf_dups = (
        prot_df[prot_df.duplicated(subset=dup_cols, keep=False)]
        .sort_values(dup_cols)
        .reset_index(drop=True).copy()
    )
    gca_gcf_dups.to_csv(output_dir / 'protein_duplicates_same_isolate.csv', sep=',', index=False)
    print(f'dups {gca_gcf_dups.shape}')
    print(f"dups unique seqs {gca_gcf_dups['prot_seq'].nunique()}")
    print(gca_gcf_dups[dup_cols + ['assembly_prefix']])

    gca_gcf_dups_eda = dups.groupby(dup_cols).agg(
        num_assembly_prefixes=('assembly_prefix', 'nunique'),
        assembly_prefixes=('assembly_prefix', lambda x: list(set(x))),
        brc_fea_ids=('brc_fea_id', lambda x: list(set(x))),
        ).sort_values(dup_cols).reset_index()
    gca_gcf_dups_eda.to_csv(output_dir / 'protein_duplicates_same_isolate_eda.csv', sep=',', index=False)
    
    breakpoint()
    del dup_cols, gca_gcf_dups, gca_gcf_dups_eda
    

    """
    Duplicate case: duplicates that with different function annotations in the same file
    TODO Note! These exacly match the same-file duplicates
    """
    print('\nDuplicate case -- Explore duplicates (functions annotations)')

    dup_cols = [seq_col_name, 'file']

    dups_func_conflicts = (
        prot_df[prot_df.duplicated(subset=dup_cols, keep=False)]
        .sort_values(dup_cols + ['brc_fea_id'])
        .reset_index(drop=True).copy()
    )
    print(f'dups {dups_func_conflicts.shape}')
    print(f"dups unique seqs {dups_func_conflicts['prot_seq'].nunique()}")
    print(dups_func_conflicts[dup_cols + ['brc_fea_id']])

    dups_func_conflicts_eda = dups_func_conflicts.groupby(dup_cols).agg(
            num_funcs=('function', 'nunique'),
            num_replicons=('replicon_type', 'nunique'),
            functions=('function', lambda x: list(set(x))),
            replicons=('replicon_type', lambda x: list(set(x))),
        ).sort_values(dup_cols).reset_index()

    # dups_func_conflicts = (
    #     prot_dups.groupby(dup_cols).agg(
    #         num_funcs=('function', 'nunique'),
    #         num_replicons=('replicon_type', 'nunique'),
    #         functions=('function', lambda x: list(set(x))),
    #         replicons=('replicon_type', lambda x: list(set(x))),
    #     )
    #     .reset_index()
    # )
    # # Same prot_seq, same file, different functions
    # dups_func_conflicts = dups_func_conflicts[dups_func_conflicts['num_funcs'] > 1].reset_index(drop=True)
    # dups_func_conflicts.to_csv(output_dir / 'protein_duplicates_function_conflicts.csv', sep=',', index=False)
    # breakpoint()
    
    breakpoint()
    del dup_cols, dups_func_conflicts, dups_func_conflicts_eda






    # Find sequences that exist in both GCA and GCF for same (seq, assembly, function, segment)
    to_filter = gca_gcf_eda[gca_gcf_eda['num_assembly_prefixes'] == 2]

    # Create a set of tuples you want to remove (GCA only)
    drop_keys = set(
        tuple(row[col] for col in dup_cols)
        for _, row in to_filter.iterrows()
    )
    drop_keys = list(drop_keys)

    # Filter GCA entries where GCF exists
    mask = (
        prot_df['assembly_prefix'] == 'GCA'
        & prot_df[dup_cols].apply(tuple, axis=1).isin(drop_keys) # TODO explain this!
    )
    filtered_df = prot_df[~mask].copy()



    """
    Step 1: Drop GCA sample if the same sequence appears also in GCF with the same assembly_id
    🔹 [Essential] Filter technical duplicates within the same assembly (Case 1 analog).
    If the same protein sequence appears in both a GCA and GCF assembly with the same assembly_id,
    drop the GCA.
    """
    print('\n----- Step 1: Drop GCA duplicates when GCF exists for the same protein + assembly_id -----')

    breakpoint()
    # Step 1.1: Identify duplicated protein sequences
    dups = prot_df[prot_df.duplicated(subset=['prot_seq'], keep=False)].copy()

    # Step 1.2: Identify [prot_seq, assembly_id] combinations where both GCA and GCF exist
    gca_gcf_pairs = dups.groupby(['prot_seq', 'assembly_id'])['assembly_prefix'].apply(set).reset_index()

    # Filter to rows where both GCA and GCF versions exist
    gca_gcf_pairs['has_both'] = gca_gcf_pairs['assembly_prefix'].apply(lambda x: 'GCA' in x and 'GCF' in x)
    gca_gcf_with_both = gca_gcf_pairs[gca_gcf_pairs['has_both']].reset_index()
    print(f'Total GCA-GCF pairs with both: {gca_gcf_with_both.shape}')

    # Step 1.3: Create a set of (prot_seq, assembly_id) to drop if GCA
    to_drop = set(zip(gca_gcf_with_both['prot_seq'], gca_gcf_with_both['assembly_id']))

    # Step 1.4: Vectorized filtering — drop only the GCA rows for those pairs
    drop_mask = (prot_df['assembly_prefix'] == 'GCA') & prot_df[['prot_seq', 'assembly_id']].apply(tuple, axis=1).isin(to_drop)

    # Step 1.5: Apply the mask to filter the dataframe
    prot_df_filtered = prot_df[~drop_mask].copy()

    # Show how many rows were dropped
    num_dropped = drop_mask.sum()
    print(prot_df_filtered.shape)
    print(num_dropped)

    # breakpoint()
    df = prot_df_filtered.copy()
    print(df['replicon_type'].value_counts())
    print(df['canonical_segment'].value_counts())
    print(df['function'].value_counts())
    dups = df[df.duplicated(subset=['prot_seq'], keep=False)].sort_values(['prot_seq'])
    dup_counts = dups.groupby('prot_seq')['file'].nunique().reset_index(name='file_count')
    print(dup_counts['file_count'].value_counts().reset_index(name='total_cases'))


    """
    Step 2: Remove duplicate protein sequences within the same file
    """
    # Goal: For any file, if the same protein sequence appears more than once, keep only one instance

    # Drop duplicates based on (file, protein_translation)
    protein_df_dedup = protein_df_filtered.drop_duplicates(subset=['file', 'protein_translation'])

    # Show how many were removed
    num_removed = len(protein_df_filtered) - len(protein_df_dedup)
    protein_df_dedup.shape, num_removed





