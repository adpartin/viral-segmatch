"""
Exploratory analysis of Bunyavirales GTO file structure.

Status (2026-05-13): NOT MAINTAINED. Bunya preprocessing is in the
"What Is NOT Maintained" list in CLAUDE.md, and the data_version this
script points at (`April_2025`) is itself superseded. Preserved as a
reference for how to inspect a quality-annotated GTO file triplet
should the Bunya path ever be revived; do not expect it to run
unmodified against current data layouts.

Purpose
-------
Inspects the per-assembly outputs of the Bunya quality-annotation
pipeline. Each assembly produces three files in
`data/raw/Anno_Updates/<data_version>/bunya-from-datasets/Quality_GTOs/`:

- `<accession>.contig_quality` — per-contig quality flags (TSV)
- `<accession>.feature_quality` — per-feature (CDS) quality flags (TSV)
- `<accession>.qual.gto` — the full GTO JSON (same schema family as Flu
  GTOs; see `docs/methods/gto_format_reference.md`)

The script picks one assembly (hard-coded `ex_file` constant) and:
- prints the contig_quality and feature_quality tables
- prints the top-level keys of the GTO JSON
- enumerates the per-feature keys
- prints one representative feature in full
- lists every feature's (id, type, function) triple

Run
---
    python eda/bunya_gto_eda.py

To inspect a different assembly, edit `ex_file` near the top. To
point at a different data version, edit `data_version`.

Outputs
-------
stdout only.
"""

import sys
import json
from pathlib import Path
from pprint import pprint
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

data_version = 'April_2025'
virus_name = 'bunya'
main_data_dir = project_root / 'data'
raw_data_dir = main_data_dir / 'raw' / 'Anno_Updates' / data_version
quality_gto_dir = raw_data_dir / 'bunya-from-datasets' / 'Quality_GTOs'
output_dir = main_data_dir / 'processed' / virus_name / data_version

# breakpoint()
# ex_file = 'GCF_031497195.1'
ex_file = 'GCF_013086535.1' # had issues with this file (two GPC proteins no duplicates)

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