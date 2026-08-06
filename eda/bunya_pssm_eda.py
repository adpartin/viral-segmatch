"""
Exploratory analysis of Bunyavirales Viral_PSSM.json.

Status (2026-05-13): NOT MAINTAINED. Bunya preprocessing is in the
"What Is NOT Maintained" list in CLAUDE.md, and the data_version this
script points at (`Feb_2025`) is itself superseded. Preserved as a
reference for the Viral_PSSM.json schema should the Bunya path ever
be revived.

Context
-------
The BV-BRC viral annotation script reads from Viral_PSSM.json. If a
protein has a segment assigned to it, that segment is used in defining
the genome and verifying it's intact.

Viral_PSSM.json is a reference schema for the expected structure and
composition of segmented viral genomes — primarily Bunyavirales but
also other segmented RNA virus orders (Arenaviridae, Phasmaviridae,
etc.). It describes:
  1. Which segments (replicons) are expected for each viral family.
  2. Which features (usually protein-coding genes) are expected on each segment.
  3. Expected length bounds for both segments and features.
  4. Which feature belongs to which segment, by curated domain knowledge.

What this script does
---------------------
- Loads `Viral_PSSM.json` from
  `data/raw/Anno_Updates/<data_version>/bunya-from-datasets/`.
- Parses the family-keyed JSON into a flat DataFrame `pssm_df` with
  one row per segment and one row per feature, columns:
  `family, segment_name, segment_min_len, segment_max_len,
   replicon_geometry, feature_name, feature_type, anno, segment,
   min_len, max_len, type` (where `type` ∈ {segment, feature}).
- Drops into `breakpoint()` for interactive inspection of `pssm_df`.

Suggested follow-ups (commented out at the bottom): write `pssm_df` to
`viral_pssm.csv`, summarize family / segment distributions.

Run
---
    python eda/bunya_pssm_eda.py

Outputs
-------
stdout + a Python debugger prompt for interactive exploration.
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