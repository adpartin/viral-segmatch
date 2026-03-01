"""
Preprocess DNA/RNA data from GTO files for Bunyavirales.

TODO
1. Finish and test src/utils/dna_utils.py
"""
import sys
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# print(f'project_root: {project_root}')

from src.utils.gto_utils import (
    extract_assembly_info,
    enforce_single_file,
    classify_dup_groups,
    SequenceType
)
from src.utils.dna_utils import summarize_dna_qc, clean_dna_sequences
from src.utils.timer_utils import Timer

total_timer = Timer()

# Config
data_version = 'April_2025'
virus_name = 'bunya'
main_data_dir = project_root / 'data'
raw_data_dir = main_data_dir / 'raw' / 'Anno_Updates' / data_version
quality_gto_dir = raw_data_dir / 'bunya-from-datasets' / 'Quality_GTOs'
output_dir = main_data_dir / 'processed' / virus_name / data_version
output_dir.mkdir(parents=True, exist_ok=True)

print(f'main_data_dir:   {main_data_dir}')
print(f'raw_data_dir:    {raw_data_dir}')
print(f'quality_gto_dir: {quality_gto_dir}')
print(f'output_dir:      {output_dir}')


def get_dna_data_from_gto(gto_file_path: Path) -> pd.DataFrame:
    """
    Extract DNA/RNA data and metadata from a GTO file into a DataFrame.

    The 'id' in 'contigs' contains values like: NC_038733.1, OL774846.1, etc.
    These are GenBank accession numbers, corresponding to individual RNA segments.     
    """
    with open(gto_file_path, 'r') as file:
        gto = json.load(file)
    
    # Extract data from GTO's 'contigs'
    # Each item in 'contigs' is a dict with keys: 'id', 'replicon_type', 'replicon_geometry', 'dna'
    # The created contigs is a list of dicts, each representing a 'contigs' item
    contigs = [
        {k: d[k] for k in sorted(d.keys())}
        for d in gto.get('contigs', [])
    ]
    
    # Aggregate contigs into DataFrame
    contigs_columns = ['id', 'replicon_type', 'replicon_geometry', 'dna']
    df = []
    for item in contigs:
        df.append([item.get(k, np.nan) for k in contigs_columns])
    df = pd.DataFrame(df, columns=contigs_columns)
    df['length'] = df['dna'].apply(lambda x: len(x) if isinstance(x, str) else 0)

    # Extract additional metadata from GTO file
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


def aggregate_dna_data_from_gto_files(gto_dir: Path) -> pd.DataFrame:
    """Aggregate DNA data from all GTO files."""
    dfs = []
    gto_files = sorted(gto_dir.glob('*.qual.gto'))
    for fpath in tqdm(gto_files, desc='Aggregating DNA/RNA data from GTO files'):
        dfs.append(get_dna_data_from_gto(fpath))
    return pd.concat(dfs, axis=0).reset_index(drop=True)


# Aggregate DNA data from GTO files
# breakpoint()
print(f"\nAggregating DNA/RNA data from {len(sorted(quality_gto_dir.glob('*.qual.gto')))} GTO files.")
dna_df = aggregate_dna_data_from_gto_files(quality_gto_dir)
print(f'dna_df: {dna_df.shape}')

# Save initial data
dna_df.to_csv(output_dir / 'dna_agg_from_GTOs.csv', sep=',', index=False)
dna_df_no_seq = dna_df[dna_df['dna'].isna()]
print(f'Records with missing DNA sequence: {dna_df_no_seq.shape}')
dna_df_no_seq.to_csv(output_dir / 'dna_missing_seqs.csv', sep=',', index=False)

# Check unique GenBank accession numbers
print(f"\nTotal genbank_ctg_id:  {dna_df['genbank_ctg_id'].count()}")
print(f"Unique genbank_ctg_id: {dna_df['genbank_ctg_id'].nunique()}")

# Clean DNA sequences
# TODO Finish and test src/utils/dna_utils.py
# Five samples with ambiguous_frac of > 0.1 (April 2025)
print('\nCleaning DNA sequences.')
dna_qc = summarize_dna_qc(dna_df, seq_col='dna')
print(dna_qc[:3])
print(dna_qc['ambig_frac'].value_counts().sort_index(ascending=False))
# dna_df, ambig_df = clean_dna_sequences(dna_df, max_ambig_frac=0.1)
# if not ambig_df.empty:
#     ambig_df.to_csv(output_dir / 'dna_ambiguous.csv', sep=',', index=False)
#     print(f"Saved ambiguous DNA sequences to {output_dir / 'dna_ambiguous.csv'}")

# Explore and handle duplicates
print("\nHandling DNA duplicates.")
dna_dups = dna_df[dna_df.duplicated(subset=['dna'], keep=False)].sort_values('dna').reset_index(drop=True)
dna_dups.to_csv(output_dir / 'dna_all_duplicates.csv', sep=',', index=False)
print(f"Duplicates on 'dna': {dna_dups.shape}")

dup_summary = classify_dup_groups(dna_dups, SequenceType.DNA)
dup_summary.to_csv(output_dir / 'dna_duplicates_summary.csv', sep=',', index=False)

case1 = dup_summary[dup_summary['case'] == 'Case 1']
case2 = dup_summary[dup_summary['case'] == 'Case 2']
other = dup_summary[dup_summary['case'] == 'Other']
print(f"Case 1: {case1.shape[0]} sequences")
print(f"Case 2: {case2.shape[0]} sequences")
print(f"Other:  {other.shape[0]} sequences")

show_cols = [c for c in dup_summary.columns if c not in
    [SequenceType.DNA.value, SequenceType.PROTEIN.value]]
print(f'Case 1:\n{case1[:3][show_cols]}')
print(f'Case 2:\n{case2[:3][show_cols]}')

# Enforce single file
print("\nEnforcing single file per assembly_id.")
print(f'dna_df before: {dna_df.shape}')
dna_df = enforce_single_file(dna_df)
print(f'dna_df after:  {dna_df.shape}')

# Save final DNA data
print("\nSaving final DNA data.")
dna_df.to_csv(output_dir / 'dna_filtered.csv', sep=',', index=False)
print(f'dna_df final: {dna_df.shape}')
print(f"Unique DNA sequences: {dna_df['dna'].nunique()}")

total_timer.display_timer()
print('\nDone!')