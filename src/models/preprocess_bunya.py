import os
import json
import re
from collections import Counter
from pathlib import Path
from pprint import pprint
from time import time
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

filepath = Path(__file__).parent # .py
# filepath = Path(os.path.abspath('')) # .ipynb
print(f'filepath: {filepath}')

process_dna = False
process_protein = True

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

    def get_dna_data_from_gto(gto_file_path):
        """
        Extract dna/rna data and metadata from a GTO file and return it as a DataFrame.
        """
        # file_path = quality_gto_dir / 'GCF_031497195.1.qual.gto'

        ## Load GTO file
        with open(gto_file_path, 'r') as file:
            gto = json.load(file)

        ## Extract data from 'contigs'
        # Each item in 'contigs' is a dict with keys: 'id', 'replicon_type', 'replicon_geometry', 'dna'
        # ctg is a list of dicts, each representing a 'contigs' item
        cng = []
        for d in gto['contigs']:
            cng.append({k: d[k] for k in sorted(d.keys())})

        # Aggregate the 'contigs' items into a DataFrame
        df = []
        # contigs_columns contains available data items from 'contigs' key
        contigs_columns = ['id', 'replicon_type', 'replicon_geometry', 'dna'] # 'id' is an Accession Number
        for i, item in enumerate(cng):
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
        return df


    def aggregate_data_from_gto_files(gto_dir):
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
    print(f"Total accession ids:        {dna_df['id'].count()}")
    print(f"Total unique accession ids: {dna_df['id'].nunique()}")

    """
    Ambiguous DNA Data.
    Remove sequences with too many unknown bases ('N' in DNA or 'X' in protein).
    """
    def summarize_dna_qc(df, seq_col='dna'):
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
    # TODO! How should we filter this??
    print(dna_qc[:3])
    print(dna_qc['ambig_frac'].value_counts().sort_index(ascending=False))

    """
    Check replicon types.
    """
    # Most segments follow the "[Large/Medium/Small] RNA Segment" naming convention.
    # Some segments are labeled as "Segment One", "Segment Two", ...
    # 311 entries have missing values
    # TODO! 1) How can we standardize the "Segment [One/Two/...]" labels? 2) How can we handle the missing values?
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

    This is likely a technical duplicate (e.g., GCA/GCF versions of the same assembly for a given viral isolate). Some GTO files may contain both draft (GCA) and reference (GCF) assemblies for the same viral isolate, leading to identical sequences appearing in different files.
    GCA (Genomic Contig Assembly): A draft genome assembly, typically submitted first.
    GCF (Genomic Complete Assembly): A higher-quality, curated version of the same assembly.

    Example:
    file                        assembly_id     Accession Number    dna
    GCA_002831345.1.qual.gto    002831345.1     NC_038733.1         AAAACAGTAGTGTACCG...
    GCF_002831345.1.qual.gto    002831345.1     NC_038733.1         AAAACAGTAGTGTACCG...

    Accession Number: NC_038733.1 (same RNA segment, same isolate)
    File names: GCA vs. GCF (same assembly ID, different versions)

    Conclusion:
    Keep the GCF version (it's typically a higher-quality). TODO confirm with Jim

    --------
    Case 2:
    --------
    Sequences (same), Assembly IDs (same), Accession Numbers (different), Versions (ignore)

    This means that the same genomic segment has been assigned different accession numbers within the same assembly. This can happen due to:
    Reannotation: The sequence may have been re-submitted with updated metadata.
    Redundant entries: Multiple GenBank submissions within the same assembly.

    Example:
    file                        assembly_id     Accession Number    dna
    GCA_002831345.1.qual.gto    002831345.1     NC_038733.1         AAAACAGTAGTGTACCG...
    GCA_002831345.1.qual.gto    002831345.1     OL123456.1          AAAACAGTAGTGTACCG...

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
    If the assemblies are from different viral isolates, this reflects true biological conservation — keep both.
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


    def classify_dup_groups(df):
        """
        Classifies duplicate groups based on the number of unique assembly IDs, accession numbers, and prefixes.
        """
        dup_info = []

        # df.groupby('dna') retrieves the rows with the same dna sequence (i.e., dna sequence duplicates)
        for seq, group in df.groupby('dna'):
            files = list(group['file'])               # GCA_018595275.1.qual.gto
            assembly_ids = list(group['assembly_id']) # 18595275.1
            accession_ids = list(group['id'])         # MW039262.1
            prefixes = list(group['assembly_prefix']) # GCA
            
            case = None
            if len(set(assembly_ids)) == 1 and len(set(accession_ids)) == 1 and set(prefixes) == {'GCA', 'GCF'}:
                # Sequecnes (same), Assembly IDs (same), Accession Numbers (same), Versions (different)
                case = 'Case 1'
            elif len(set(assembly_ids)) == 1 and len(set(accession_ids)) > 1:
                # Sequences (same), Assembly IDs (same), Accession Numbers (different), Versions (ignore)
                case = 'Case 2'
            elif len(set(assembly_ids)) > 1 and len(set(accession_ids)) > 1:
                # Sequences (same), Assembly IDs (different), Accession Numbers (different), Versions (ignore)
                case = 'Case 3'
            else:
                case = 'Other'

            dup_info.append({
                'dna': seq,
                'num_dups': len(group),
                'files': files,
                'assembly_ids': list(set(assembly_ids)),
                'accession_ids': list(set(accession_ids)),
                'prefixes': list(set(prefixes)),
                'case': case
            })

        return pd.DataFrame(dup_info)

    """
    ✅ Summary of Duplicate Classifications.
    Case	Description	                                            Count	Suggested Action
    Case 1	Same assembly_id, same accession_id, GCA & GCF versions	0	    ✅ Nothing to do
    Case 2	Same assembly_id, different accession numbers	        1,524	⚠️ Technical duplicates — keep only one
    Case 3	Different assembly_ids and accession_ids	            74	    ✅ Likely true biological conservation — keep all
    """
    dna_dups = dna_df[dna_df.duplicated(subset=['dna'], keep=False)].sort_values(['dna'])
    dup_summary = classify_dup_groups(dna_dups)

    case1 = dup_summary[dup_summary['case'] == 'Case 1']
    case2 = dup_summary[dup_summary['case'] == 'Case 2']
    case3 = dup_summary[dup_summary['case'] == 'Case 3']
    print(f"Case 1: {case1.shape[0]} sequences")
    print(f"Case 2: {case2.shape[0]} sequences")
    print(f"Case 3: {case3.shape[0]} sequences")

    print(f'Case 2:\n{case2[:3]}')

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
        """
        # gto_file_path = quality_gto_dir / 'GCF_031497195.1.qual.gto'

        # Load GTO file
        with open(gto_file_path, 'r') as file:
            gto = json.load(file)

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
            row['segment_id'] = segment_id
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
        df.rename(columns={'protein_translation': 'prot_seq'}, inplace=True)

        return df


    def aggregate_protein_data_from_gto_files(gto_dir) -> pd.DataFrame:
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
    print(f"\nTotal .qual.gto files: {len(sorted(quality_gto_dir.glob('*.qual.gto')))}")
    prot_df = aggregate_protein_data_from_gto_files(quality_gto_dir)
    print(f'Protein all samples {prot_df.shape}')
    print(prot_df.columns.tolist())

    # Save all samples
    prot_df.to_csv(output_dir / 'protein_data_all.tsv', sep='\t', index=False)
    prot_df.to_csv(output_dir / 'protein_data_all.csv', sep=',', index=False)
    
    # Save samples that missing seq data
    prot_df_no_seq = prot_df[prot_df['prot_seq'].isna()]
    print(f'Protein with missing sequence {prot_df_no_seq.shape}')
    prot_df_no_seq.to_csv(output_dir / 'protein_data_no_seq.tsv', sep='\t', index=False)
    prot_df_no_seq.to_csv(output_dir / 'protein_data_no_seq.csv', sep=',', index=False)

    """
    Check that all accession numbers are unique.
    """
    # All segment IDs are unique (no duplicates).
    print('\nCheck that all accession numbers are unique.')
    print(f"Total accession ids:        {prot_df['id'].count()}")
    print(f"Total unique accession ids: {prot_df['id'].nunique()}")

    """
    Explore function-replicon combinations.
    TODO! Can it be that several segment types have the same function (i.e., encode the same protein)?
    E.g., 'RNA-dependent RNA polymerase (L protein)' is associated with: 'Large RNA Segment', 'Segment One', 'Small RNA Segment'
    """
    func_rep = prot_df.groupby(['function', 'replicon_type']).agg(replicons=('replicon_type', 'count')).sort_values('function', ascending=False).reset_index()
    func_rep.to_csv(output_dir / 'function_replicon_count.csv', sep=',', index=False)

    # Check specific function: 'RNA-dependent RNA polymerase (L protein)'
    df = prot_df[prot_df['function'].isin(['RNA-dependent RNA polymerase (L protein)'])]
    print(df['replicon_type'].value_counts())

    """
    Explore duplicates in protein data.
    """
    # Count exact duplicate protein sequences
    prot_dups = prot_df[prot_df.duplicated(subset=['prot_seq'], keep=False)].sort_values(['prot_seq'])
    prot_dups.to_csv(output_dir / 'protein_all_duplicates.csv', sep=',', index=False)
    print(f"Duplicates on 'prot_seq': {prot_dups.shape}")
    print(prot_dups[:4])

    # Count how many unique files contain each duplicated protein sequence
    # This helps identify if a protein seq appears across multiple files or within the same
    print(f'\nCount how many unique files contain each duplicated protein sequence')
    dup_counts = prot_dups.groupby('prot_seq')['file'].nunique().reset_index(name='file_count')
    print(dup_counts['file_count'].value_counts().reset_index(name='total_cases'))

    # df.groupby('dna') retrieves the rows with the same dna sequence (i.e., dna sequence duplicates)
    grouped = prot_dups.groupby('prot_seq')

    # Explore the first group
    for seq, grp in grouped:
        print(f"Sequence: {seq[:50]}")  # print a slice of seq
        print("Group (rows with this seq):")
        print(grp)
        print(grp[['prot_seq', 'file', 'function', 'replicon_type']])
        break  # remove this to loop over more groups

    # Count how many unique protein sequences are duplicated
    dup_stats = prot_dups.groupby('prot_seq').agg(
            num_occurrences=('prot_seq', 'count'),
            num_files=('file', 'nunique'),
            num_functions=('function', 'nunique'),
            num_replicons=('replicon_type', 'nunique'),
            functions=('function', lambda x: list(set(x))),      # the function (protein)
            replicons=('replicon_type', lambda x: list(set(x))), # the segment
        ).sort_values('num_occurrences', ascending=False).reset_index()
    dup_stats.to_csv(output_dir / 'protein_duplicates_stats.csv', sep=',', index=False)

    # Note 1:  num_occurrences >= num_files
    # It means that there are cases where a protein sequence appears multiple times in the same file.
    # TODO Ask Jim. Why this may happen?? What should we do about this??
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
    print(df[['file', 'assembly_prefix', 'assembly_id', 'id', 'type', 'replicon_type', 'function']][:3])


    """
    Step 1: Drop GCA entries if the same protein sequence exists in GCF for the same assembly_id
    """
    # Step 1.1: Identify all duplicated protein sequences
    dups = prot_df[prot_df.duplicated(subset=['prot_seq'], keep=False)].copy()

    # Step 1.2: Keep only rows where GCA and GCF exist for the same protein and same assembly_id
    gca_gcf_pairs = dups.groupby(['prot_seq', 'assembly_id'])['assembly_prefix'].apply(set).reset_index()

    # Filter to rows where both GCA and GCF versions exist
    gca_gcf_pairs['has_both'] = gca_gcf_pairs['assembly_prefix'].apply(lambda x: 'GCA' in x and 'GCF' in x)
    gca_gcf_with_both = gca_gcf_pairs[gca_gcf_pairs['has_both']]
    print(f"Total GCA-GCF pairs with both: {gca_gcf_with_both.shape}")

    # Step 1.3: Build a mask to drop the GCA rows from prot_df
    drop_mask = prot_df.apply(
        lambda row: (
            row['assembly_prefix'] == 'GCA'
            and (row['prot_seq'], row['assembly_id']) in 
                set(zip(gca_gcf_with_both['prot_seq'], gca_gcf_with_both['assembly_id']))
        ),
        axis=1
    )

    # Step 1.4: Apply the filtering
    prot_df_filtered = prot_df[~drop_mask].copy()

    # Show how many rows were dropped
    num_dropped = drop_mask.sum()
    print(prot_df_filtered.shape)
    print(num_dropped)

    """
    Step 2: Remove duplicate protein sequences within the same file
    """
    # Goal: For any file, if the same protein sequence appears more than once, keep only one instance

    # Drop duplicates based on (file, protein_translation)
    protein_df_dedup = protein_df_filtered.drop_duplicates(subset=["file", "protein_translation"])

    # Show how many were removed
    num_removed = len(protein_df_filtered) - len(protein_df_dedup)
    protein_df_dedup.shape, num_removed





