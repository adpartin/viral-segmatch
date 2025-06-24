import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Config
data_version = 'April_2025'
virus_name = 'bunya'
main_data_dir = project_root / 'data'
processed_data_dir = main_data_dir / 'processed' / virus_name / data_version
outdir = project_root / 'eda' / virus_name / data_version
outdir.mkdir(parents=True, exist_ok=True)

# Load protein data
print('\nLoad filtered protein data.')
fname = 'protein_filtered.csv'
datapath = processed_data_dir / fname
try:
    df = pd.read_csv(datapath)
except FileNotFoundError:
    raise FileNotFoundError(f"Data file not found at: {datapath}")

# Histogram by canonical_segment
plt.figure(figsize=(10, 6))
for segment in ['L', 'M', 'S']:
    sns.histplot(
        data=df[df['canonical_segment'] == segment],
        x='length',
        label=segment,
        alpha=0.5,
        bins=30
    )
plt.title('Protein Length Distribution by Canonical Segment')
plt.xlabel('Sequence Length (Residues)')
plt.ylabel('Count')
plt.legend()
plt.savefig(outdir / 'histogram_protein_length_by_segment.png')
plt.close()

# Box plot by canonical_segment
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='canonical_segment', y='length', order=['L', 'M', 'S'])
plt.title('Protein Length Distribution by Canonical Segment')
plt.xlabel('Canonical Segment')
plt.ylabel('Sequence Length (Residues)')
plt.savefig(outdir / 'boxplot_protein_length_by_segment.png')
plt.close()

# Summary statistics
print("\nLength statistics by canonical_segment:")
print(df.groupby('canonical_segment')['length'].describe())

print(f'\nPlots saved to: {outdir}')
print('Done.')