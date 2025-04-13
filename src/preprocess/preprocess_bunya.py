"""
TODO Questions:
1. Can we map pssm_df['family'] to prot_df? Can this allow us to determine if it's 2-segment or 3-segment virus?
2. For modeling, do we really need to know what segment that it (i.e., S, M, L)?
3. Should we filters GCA/GCF duplicates?
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
# process_dna = False
process_dna = True
process_protein = True
# protein_core_filter = False
protein_core_filter = True
# protein_aux_filter = False
protein_aux_filter = True

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
            'replicon_type', 'length', 'dna']]

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

    df = pd.DataFrame(dup_info).sort_values('case')
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
pssm_df.to_csv(output_dir / 'viral_pssm.tsv', sep='\t', index=False)
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
    dna_data_fname = 'dna_data_all.tsv'
    dna_df.to_csv(output_dir / 'dna_data_all.tsv', sep='\t', index=False)
    dna_df.to_csv(output_dir / 'dna_data_all.csv', sep=',', index=False)

    # Save samples that missing seq data
    dna_df_no_seq = dna_df[dna_df['dna'].isna()]
    print(f'DNA with missing sequence {dna_df_no_seq.shape}')
    dna_df_no_seq.to_csv(output_dir / 'dna_data_no_seq.tsv', sep='\t', index=False)
    dna_df_no_seq.to_csv(output_dir / 'dna_data_no_seq.csv', sep=',', index=False)

    """
    Check that all accession numbers are unique.
    """
    # All segment IDs are unique (no duplicates).
    print('\nCheck that all accession numbers are unique.')
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
    # Most segments follow the "[Large/Medium/Small] RNA Segment" naming convention.
    # Some segments are labeled as "Segment [One/Two/Three/Four]"
    # 311 entries have missing values
    # TODO! 1) How should we standardize the "Segment [One/...]" labels? 2) Should we drop NaN samples?
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
    If more than 2 duplicates
    =========================
    Case 1:
    TODO.

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

    dup_counts = dna_dups.groupby('dna')['file'].nunique().reset_index(name='unq_files')
    print(dup_counts['unq_files'].value_counts().reset_index())

    """
    ✅ Summary of Duplicate Classifications.
    Case	Description	                                        Count	Suggested Action
    Case 1	Same assembly_id, same record_id, GCA/GCF versions  0	    ✅ Nothing to do
    Case 2	Same assembly_id, diff record_id	                1,524	⚠️ Technical dups — keep one
    Case 3	Diff assembly_id and record_id	                    74	    ✅ Likely true biological conservation — keep all
    """
    breakpoint()
    dup_summary = classify_dup_groups(dna_dups, SequenceType.DNA)

    show_cols = [c for c in case2.columns if c not in
        [SequenceType.DNA.value, SequenceType.PROTEIN.value]]
    case1 = dup_summary[dup_summary['case'] == 'Case 1']
    case2 = dup_summary[dup_summary['case'] == 'Case 2']
    case3 = dup_summary[dup_summary['case'] == 'Case 3']
    case_other = dup_summary[dup_summary['case'] == 'Other']
    print(f"Case 1: {case1.shape[0]} sequences")
    print(f"Case 2: {case2.shape[0]} sequences")
    print(f"Case 3: {case3.shape[0]} sequences")
    print(f"Case Other: {case_other.shape[0]} sequences")

    # print(f'Case 2:\n{case2[:3]}')
    print(f'Case 2:\n{case2[:3][show_cols]}')
    print(f'Case 3:\n{case3[:3][show_cols]}')

    # Explore duplicate groups with more than 2 files
    dna_dups = dna_df[dna_df.duplicated(subset=['dna'], keep=False)].sort_values(['dna'])
    dup_counts = dna_dups.groupby('dna')['file'].nunique().reset_index(name='num_files')
    multi_dups = dup_counts[dup_counts['num_files'] > 2] # > 2 dups
    multi_dup_records = dna_dups.merge(multi_dups[['dna']], on='dna') # Merge back to get full rows for inspection
    multi_dup_records = multi_dup_records.sort_values('dna')
    print(f'\nmulti_dup_records: {multi_dup_records.shape}')
    print(multi_dup_records.shape)



# =======================================================================
# Protein sequence data
# =======================================================================

if process_protein:

    def get_protein_data_from_gto(gto_file_path: Path) -> pd.DataFrame:
        """
        Extract protein data and metadata from GTO file and return as a DataFrame.

        The 'id' in 'features' contains values like: fig|1316165.15.CDS.3, fig|11588.3927.CDS.1, etc.
        These are PATRIC (BV-BRC) feature IDs: fig|<genome_id>.<feature_type>.<feature_number>
        The genome_id (e.g., 1316165.15) is the internal genome identifier.
        This 'id' can be used to trace the feature to its source genome in PATRIC/GTOs.
        This is a not GenBank accession number (e.g., NC_038733.1, OL987654.1).

        The 'location' in 'features' contains: [[ <segment_id>, <start>, <strand>, <end>]]
        For example: "location": [[ "NC_086346.1", "70", "+", "738" ]]
        It means, this feature is located on segment NC_086346.1 (positive strand), from
        nucleotide 70 to 738.
        The NC_086346.1 is a GenBank-style accession, used by NCBI for nucleotide entries.
        """
        # gto_file_path = quality_gto_dir / 'GCF_031497195.1.qual.gto'

        # Load GTO file
        with open(gto_file_path, 'r') as file:
            gto = json.load(file)

        # TODO this was copied from dna processing! check this!
        # Build segment_id → replicon_type mapping
        # Iterates over 'contigs' key in GTO (if available)
        # Each item in 'contigs' is a dict with keys: 'id', 'dna', 'replicon_type', 'replicon_geometry'
        # 'id': segment id (e.g., NC_086346.1)
        # 'replicon_type': segment type/name (e.g., L, M, S)
        segment_map: Dict[str, str] = {
            contig['id']: contig.get('replicon_type', 'Unknown')
            for contig in gto.get('contigs', [])
        }

        ## Extract data from 'features'
        # features_columns contains data items available in 'features' key
        features_columns = ['id', 'type', 'function', 'protein_translation', 'location']
        rows = []
        for fea_dict in gto['features']: # d is a 'features' dict
            row = {k: fea_dict.get(k, None) for k in features_columns}
            row['length'] = len(fea_dict['protein_translation']) if 'protein_translation' in fea_dict else 0
            # Segment ID from location field
            if isinstance(fea_dict.get('location'), list) and len(fea_dict['location']) > 0:
                segment_id = fea_dict['location'][0][0]
            else:
                segment_id = None
            row['segment_id'] = segment_id # TODO consider renaming segment_accession
            row['replicon_type'] = segment_map.get(segment_id, 'Unknown')
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


    breakpoint()
    # Aggregate protein data from all GTO files
    print(f"\nAggregate protein from all {len(sorted(quality_gto_dir.glob('*.qual.gto')))} .qual.gto files.")
    prot_df = aggregate_protein_data_from_gto_files(quality_gto_dir)
    print(f'Protein all samples {prot_df.shape}')
    print(f'prot_df columns: {prot_df.columns.tolist()}')

    # Save all samples
    prot_df.to_csv(output_dir / 'protein_data_all.tsv', sep='\t', index=False)
    prot_df.to_csv(output_dir / 'protein_data_all.csv', sep=',', index=False)
    
    # Save samples that missing seq data
    prot_df_no_seq = prot_df[prot_df['prot_seq'].isna()]
    print(f'Protein with missing sequence {prot_df_no_seq.shape}')
    prot_df_no_seq.to_csv(output_dir / 'protein_data_no_seq.tsv', sep='\t', index=False)
    prot_df_no_seq.to_csv(output_dir / 'protein_data_no_seq.csv', sep=',', index=False)

    """
    Check that all feature numbers are unique.
    """
    # All segment IDs are unique (no duplicates).
    print('\nCheck that all feature numbers are unique.')
    print(f"Total feature ids   {prot_df['feature_id'].count()}")
    print(f"Unique feature ids: {prot_df['feature_id'].nunique()}")

    """
    Analyze segments (replicon_type)
    ----------------
    Explore all 'replicon_type' entires:
    Medium RNA Segment    3813
    Small RNA Segment     2512
    Large RNA Segment     1699
    Unknown                152
    Segment One             61
    Segment Two             59
    Segment Four            59
    Segment Three           45

    Explore core protein functions:
    replicon_type                                               function  count
    Large RNA Segment           RNA-dependent RNA polymerase (L protein)   1524
    Segment One                 RNA-dependent RNA polymerase (L protein)     61
    Small RNA Segment           RNA-dependent RNA polymerase (L protein)      4 TODO note below (dropped)
    Medium RNA Segment Pre-glycoprotein polyprotein GP complex (GPC p...   1296
    Small RNA Segment  Pre-glycoprotein polyprotein GP complex (GPC p...    104 TODO note below (dropped)
    Segment Two        Pre-glycoprotein polyprotein GP complex (GPC p...     59 TODO note below (dropped temporarily)
    Small RNA Segment                   Nucleocapsid protein (N protein)   1370
    Segment Three                       Nucleocapsid protein (N protein)     45 TODO note below (kept)
    Notes:
        - [Small RNA Segment, RNA-dependent RNA polymerase] - What happens here? Later dropped on 'quality' (poor)
        - [Small RNA Segment, Pre-glycoprotein polyprotein] - Probably comes from from bipartite Bunyas (confirm). We drop it since it's ambiguous in our L/M/S labeling framework (it violates the assumption that 'S' implies N protein).
        - [Segment Two, Pre-glycoprotein polyprotein] - This could be M in tripartite genomes (would keep) or S in bipartite genomes (would drop). Segment Two is taxonomically ambiguous (not enough info to resolve without consulting taxon-specific rules.)
        - [Segment Three, Nucleocapsid protein (N protein)] - This must be a 3-segment Bunya (confirm). This should be safe to combine with Small RNA Segment (confirm).

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

    # breakpoint()
    print("\nShow all ['replicon_type', 'function'] combination counts.")
    seg_func = (
        prot_df.groupby(['replicon_type', 'function'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
    )
    print(seg_func)

    # breakpoint()
    print('\nCloser look at "core" protein functions.')
    core_functions = ['RNA-dependent RNA polymerase (L protein)',
                      'Pre-glycoprotein polyprotein GP complex (GPC protein)',
                      'Nucleocapsid protein (N protein)'
    ]
    def print_replicon_func_count(df, functions=None):
        if functions is not None:
            df = df[df['function'].isin(functions)]
        df = (
            df.groupby(['replicon_type', 'function']).size()
            .reset_index(name='count')
            .sort_values(['function', 'count'], ascending=False)
        )
        print(df)

    print_replicon_func_count(prot_df, core_functions)
    

    # breakpoint()
    if protein_core_filter:
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
        """
        print('\n==== Step 1: Assign core canonical segments (L/M/S) ====')

        # Define mapping of (function, replicon_type) → canonical_segment
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

        # Merge: Assign canonical segments where function-replicon pairs match
        prot_df = prot_df.merge(
            core_function_segment_map, on=['function', 'replicon_type'],
            how='left'  # keep all rows, NaN where mapping doesn't apply
        )

        # Diagnostics
        # breakpoint()
        print('\nAssigned canonical segments.')
        print(prot_df['canonical_segment'].value_counts(dropna=False))

        unmapped = prot_df[prot_df['canonical_segment'].isna()]
        print(f'\nUnmapped after Step 1: {len(unmapped)}')
        print(unmapped['function'].value_counts())


    # breakpoint()
    if protein_aux_filter:
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
        print('\n==== Step 2: Assign auxiliary canonical segments (NSs, NSm, etc.) ====')

        # breakpoint()
        print('\nTake closer look at "auxiliary" protein functions.')
        aux_functions = ['Bunyavirales mature nonstructural membrane protein (NSm protein)',
                         'Bunyavirales small nonstructural protein (NSs protein)',
                         'Phenuiviridae mature nonstructural 78-kD protein',
                         'Small Nonstructural Protein NSs (NSs Protein)',
        ]
        df = (
            prot_df[prot_df['function'].isin(aux_functions)]
            .groupby(['replicon_type', 'function'])
            .size()
            .reset_index(name='count')
            .sort_values('function')
        )
        print(df)

        # Define auxiliary function-to-segment mapping
        # This set can be expanded once we validate other functions (e.g., Movement protein, Z protein, etc.).
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
        # Prepare for merge (rename avoids collision with existing column)
        aux_function_segment_map = aux_function_segment_map.rename(columns={'canonical_segment': 'inferred_segment'})

        # Extract rows still missing canonical_segment
        unmapped = prot_df[prot_df['canonical_segment'].isna()]
        unmapped = unmapped.drop(columns=['canonical_segment'])

        # Merge and align
        aux_annotated = unmapped.merge(aux_function_segment_map, on=['function', 'replicon_type'], how='left')

        # Assign inferred values to prot_df (note that aux_annotated is reindexed)
        prot_df.loc[unmapped.index, 'inferred_segment'] = aux_annotated['inferred_segment']

        # Consolidate: Fill canonical_segment only if missing
        prot_df['combined_segment'] = [
        i if (pd.isna(c) and pd.notna(i)) else c 
            for c, i in zip(prot_df['canonical_segment'], prot_df['inferred_segment'])
        ]

        # Finally, keep only 'canonical_segment'
        print('\nInspect assgined segments')
        print(prot_df[['canonical_segment', 'inferred_segment', 'combined_segment']][:20])
        prot_df['canonical_segment'] = prot_df['combined_segment']
        prot_df = prot_df.drop(columns=['combined_segment', 'inferred_segment'])
        
        # Diagnostics
        # breakpoint()
        print('\nRemaining unmapped canonical segments.')
        remain_unmapped = prot_df[prot_df['canonical_segment'].isna()]
        print(f'Unmapped after Step 2: {len(remain_unmapped)}')
        print(remain_unmapped['function'].value_counts())


    print('\n============= Start filtering =============')
    """
    Keep only CDS in prot_df (drops mat_peptide and RNA)
    """
    # breakpoint()
    print('\n----- Keep protein data where type=CDS -----')
    non_cds = prot_df[prot_df['type'] != 'CDS'].sort_values('type', ascending=False)
    prot_df = prot_df[prot_df['type'] == 'CDS'].reset_index(drop=True)
    print(f"Total non-CDS samples: {non_cds.shape}")
    print(prot_df['type'].value_counts())
    non_cds.to_csv(output_dir / 'protein_data_non_cds.csv', sep=',', index=False)

    print_replicon_func_count(prot_df, core_functions)


    """
    Drop 'Poor' quality samples
    """
    # breakpoint()
    print('\n----- Drop samples where quality=Poor -----')
    poor_df = prot_df[prot_df['quality'] == 'Poor'].sort_values('quality', ascending=False)
    prot_df = prot_df[prot_df['quality'] != 'Poor'].reset_index(drop=True)
    print(f'Total Poor quality samples: {poor_df.shape}')
    print(prot_df['quality'].value_counts())
    poor_df.to_csv(output_dir / 'protein_data_poor_quality.csv', sep=',', index=False)

    print_replicon_func_count(prot_df, core_functions)


    """
    Drop segments (replicon_type) labeled 'Unknown'
    """
    # breakpoint()
    print('\n----- Drop segments (replicon_type) labeled `Unknown` -----')
    unk_df = prot_df[prot_df['replicon_type'] == 'Unknown'].sort_values('replicon_type', ascending=False)
    prot_df = prot_df[prot_df['replicon_type'] != 'Unknown'].reset_index(drop=True)
    print(f'Total `Unknown` samples: {unk_df.shape}')
    print(prot_df['replicon_type'].value_counts())
    unk_df.to_csv(output_dir / 'protein_data_unknown_replicons.csv', sep=',', index=False)

    print_replicon_func_count(prot_df, core_functions)


    """
    Drop segments (replicon_type) labeled 'Unknown'
    """
    # breakpoint()
    print('\n----- Drop entries that w/o assigned canonical segments -----')
    prot_df = prot_df[prot_df['canonical_segment'].notna()].reset_index(drop=True)

    print_replicon_func_count(prot_df, core_functions)


    print('\n============= Handle protein sequence duplicates =============')
    """
    Explore duplicates in protein data.
    """
    # Count exact duplicate protein sequences
    breakpoint()
    seq_col_name = 'prot_seq'
    print_replicon_func_count(prot_df)
    print(f'prot_df {prot_df.shape}')
    prot_dups = prot_df[prot_df.duplicated(subset=[seq_col_name], keep=False)].sort_values([seq_col_name])
    prot_dups.to_csv(output_dir / 'protein_all_duplicates.csv', sep=',', index=False)
    print(f"Duplicates on {seq_col_name}: {prot_dups.shape}")
    print(prot_dups[:4])

    # Count how many unique files contain each duplicated protein sequence
    # This helps identify if a protein seq appears across multiple files or within the same
    print(f'\nCount how many unique files contain each duplicated protein sequence')
    dup_counts = prot_dups.groupby(seq_col_name)['file'].nunique().reset_index(name='file_count')
    print(dup_counts['file_count'].value_counts().reset_index(name='total_cases'))

    # df.groupby('dna') retrieves the rows with the same sequence (i.e., sequence duplicates)
    grouped = prot_dups.groupby(seq_col_name)

    # Explore the first group
    for seq, grp in grouped:
        print(f"Sequence: {seq[:50]}")  # print a slice of seq
        print("Group (rows with this seq):")
        print(grp)
        print(grp[[seq_col_name, 'file', 'assembly_id', 'accession_id', 'function', 'replicon_type']])
        break  # remove this to loop over more groups

    # Count how many unique protein sequences are duplicated
    dup_stats = prot_dups.groupby(seq_col_name).agg(
            num_occurrences=(seq_col_name, 'count'),
            num_files=('file', 'nunique'),
            num_functions=('function', 'nunique'),
            num_replicons=('replicon_type', 'nunique'),
            functions=('function', lambda x: list(set(x))),      # the function (protein)
            replicons=('replicon_type', lambda x: list(set(x))), # the segment
        ).sort_values('num_occurrences', ascending=False).reset_index()
    dup_stats.to_csv(output_dir / 'protein_duplicates_stats.csv', sep=',', index=False)


    # Same duplicates analysis as we did with dna
    breakpoint()
    dup_summary = classify_dup_groups(prot_dups, SequenceType.PROTEIN)

    case1 = dup_summary[dup_summary['case'] == 'Case 1']
    case2 = dup_summary[dup_summary['case'] == 'Case 2']
    case3 = dup_summary[dup_summary['case'] == 'Case 3']
    print(f'Case 1: {case1.shape[0]} sequences')
    print(f'Case 2: {case2.shape[0]} sequences')
    print(f'Case 3: {case3.shape[0]} sequences')

    print(f'Case 2:\n{case2[:3]}')


    # Note 1:  num_occurrences >= num_files
    # It means there are cases where a protein sequence appears multiple times in the same file.
    # TODO Why this happens?? What should we do about this??
    # Carla: might be an error in the assembly (??) error in annotation or error in the assembly
    # For example:
    # file                      assembly_prefix  assembly_id  id                    type    function
    # GCA_003087875.1.qual.gto  GCA              003087875.1  fig|1316165.15.CDS.3  CDS     Nucleocapsid protein (N protein)
    # GCA_003087875.1.qual.gto  GCA              003087875.1  fig|1316165.15.CDS.4  CDS     Small Nonstructural Protein NSs (NSs Protein)
    #
    # Note 2:  num_functions >= num_replicons
    # This means that there is not a 1-1 mapping between replicons (segment) and functions (encoded proteins). Is that correct? Does it make sense?
    # file                      assembly_prefix assembly_id  id                    type replicon_type      function
    # GCA_003087875.1.qual.gto  GCA             003087875.1  fig|1316165.15.CDS.3  CDS  Small RNA Segment  Nucleocapsid protein (N protein)
    # GCA_003087875.1.qual.gto  GCA             003087875.1  fig|1316165.15.CDS.4  CDS  Small RNA Segment  Small Nonstructural Protein NSs (NSs Protein)
    # GCA_003087915.1.qual.gto  GCA             003087915.1  fig|1316166.15.CDS.3  CDS  Small RNA Segment  Nucleocapsid protein (N protein)
    print(f'dup_stats {dup_stats.shape}')
    print(dup_stats[['prot_seq', 'num_occurrences', 'num_files', 'num_functions', 'num_replicons', 'functions', 'replicons']][:3])
    print(f"Cases where num_occurrences >= num_files: {sum(dup_stats['num_occurrences'] >= dup_stats['num_files'])} out of {dup_stats.shape[0]}")
    print(f"Cases where num_functions >= num_replicons: {sum(dup_stats['num_functions'] >= dup_stats['num_replicons'])} out of {dup_stats.shape[0]}")

    # Explore specific protein seq that has most duplicates
    seq = dup_stats.iloc[0, :]['prot_seq']
    df = prot_df[prot_df['prot_seq'].isin([seq])]
    print(f'\nExplore specific protein seq that has most duplicates: {seq}')
    print(f'Total rows: {df.shape}')
    print(df[['file', 'assembly_prefix', 'assembly_id', 'feature_id', 'type', 'replicon_type', 'function']][:3])


    """
    Step 1: Drop GCA entries if the same protein sequence exists in GCF for the same assembly_id
    """
    print('\n----- Step 1: Drop GCA duplicates when GCF exists for the same protein + assembly_id -----')

    breakpoint()
    # Step 1.1: Identify duplicated protein sequences
    dups = prot_df[prot_df.duplicated(subset=['prot_seq'], keep=False)].copy()

    # Step 1.2: Identify [prot_seq, assembly_id] co975Gmbinations where both GCA and GCF exist
    gca_gcf_pairs = dups.groupby(['prot_seq', 'assembly_id'])['assembly_prefix'].apply(set).reset_index()

    # Filter to rows where both GCA and GCF versions exist
    gca_gcf_pairs['has_both'] = gca_gcf_pairs['assembly_prefix'].apply(lambda x: 'GCA' in x and 'GCF' in x)
    gca_gcf_with_both = gca_gcf_pairs[gca_gcf_pairs['has_both']]
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

    breakpoint()
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





