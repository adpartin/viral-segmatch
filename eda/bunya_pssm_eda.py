"""
Exploratory analysis of Bunyavirales Viral_PSSM.json.

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
"""
import sys
import json
from pathlib import Path
from pprint import pprint
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

data_version = 'Feb_2025'
# data_version = 'April_2025'
virus_name = 'bunya'
main_data_dir = project_root / 'data'
raw_data_dir = main_data_dir / 'raw' / 'Anno_Updates' / data_version
# quality_gto_dir = raw_data_dir / 'bunya-from-datasets' / 'Quality_GTOs'
# output_dir = main_data_dir / 'processed' / virus_name / data_version

with open(raw_data_dir / 'bunya-from-datasets' / 'Viral_PSSM.json', 'r') as f:
    viral_pssm = json.load(f)

# Parse into DataFrame
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

breakpoint()
pssm_df = pd.DataFrame(records)
# pssm_df.to_csv(output_dir / 'viral_pssm.csv', sep=',', index=False)
# print(f'\npssm_df {pssm_df.shape}')
# print(f"\n{pssm_df['family'].value_counts()}")
# print(f"\n{pssm_df['segment'].value_counts()}")
# print(f"\n{pssm_df['segment_name'].value_counts()}")