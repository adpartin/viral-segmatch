"""
Create dataset for a classifier to predict whether an input protein is
cytosolic or membrane-bound.
"""

import os
import requests
from pathlib import Path

import numpy as np
import pandas as pd

from io import BytesIO
from sklearn.model_selection import train_test_split

filepath = Path(__file__).resolve().parent

## Config
seed = 42  # for reproducibility (config!)
main_data_dir = filepath / '../../data'
raw_data_dir = main_data_dir / 'raw'

task_name = 'cyto_memb'
datasets_dir = main_data_dir / 'datasets' / task_name

output_dir = datasets_dir
os.makedirs(output_dir, exist_ok=True)

print(f'\nmain_data_dir:      {main_data_dir}')
# print(f'processed_data_dir: {processed_data_dir}')
print(f'datasets_dir:       {datasets_dir}')

"""
To get this query URL, we searched for `(organism_id:9606) AND
(reviewed:true) AND (length:[80 TO 500])` on UniProt. This retrieves a list of
reasonably-sized human proteins, then selected 'Download', and set the format
to TSV and the columns to `Sequence` and `Subcellular location [CC]`, since
those contain the data we need for this task.

Once that's done, selecting `Generate URL for API` provides a URL we can pass
to Requests. Alternatively, we can just download the data through the web
interface and open the file locally.
"""
# print("Querying data from UniProt ...")
# query_url = "https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Csequence%2Ccc_subcellular_location&format=tsv&query=%28%28organism_id%3A9606%29%20AND%20%28reviewed%3Atrue%29%20AND%20%28length%3A%5B80%20TO%20500%5D%29%29"
# uniprot_request = requests.get(query_url)
# bio = BytesIO(uniprot_request.content)
# df = pd.read_csv(bio, compression='gzip', sep='\t')

print('\nLoad protein data (downloaded from UniProt).')
fname = 'uniprotkb_organism_id_9606_AND_reviewed_2025_03_11.tsv.gz'
df = pd.read_csv(raw_data_dir / fname, compression='gzip', sep='\t')

df = df.dropna()  # Drop proteins with missing columns
print(f'Data shape: {df.shape}')
print(df[:4])

# Get info whether proteins are cytosolic or membrane-bound
cyto_mask = df['Subcellular location [CC]'].str.contains("Cytosol|Cytoplasm", regex=True)
memb_mask = df['Subcellular location [CC]'].str.contains("Membrane|Cell membrane", regex=True)

# Split the data into cytosolic and membrane-bound proteins
cyto_df = df[cyto_mask & ~memb_mask]  # Cytosolic only (cytosol or cytoplasm)
memb_df = df[memb_mask & ~cyto_mask]   # Membrane only (membrane or cell membrane)

# Cytosolic sequences (x) and labels (y)
cyto_seqs = cyto_df['Sequence'].tolist()
cyto_labels = [0 for protein in cyto_seqs]

# Membrane sequences (x) and labels (y)
memb_seqs = memb_df['Sequence'].tolist()
memb_labels = [1 for protein in memb_seqs]

# Concatenate the lists to get the `sequences` and `labels`
seqs = cyto_seqs + memb_seqs
labels = cyto_labels + memb_labels

# Quick check to check we got it right
assert len(seqs) == len(labels), 'Sequences and labels must be the same size'

# Split
print('\nSplit data into train and test sets.')
train_seqs, test_seqs, train_labels, test_labels = train_test_split(
    seqs, labels, test_size=0.25, shuffle=True, random_state=seed)
print(f'Train sequences: {len(train_seqs)}')
print(f'Test sequences:  {len(test_seqs)}')
print(f'type(train_sequences):    {type(train_seqs)}')
print(f'type(train_sequences[0]): {type(train_seqs[0])}')
print(f'train_sequences[0]:       {train_seqs[0]}')

print('\nSave data (sequences and labels).')
np.save(output_dir / 'train_sequences.npy', train_seqs)
np.save(output_dir / 'test_sequences.npy', test_seqs)
np.save(output_dir / 'train_labels.npy', train_labels)
np.save(output_dir / 'test_labels.npy', test_labels)

print(f"Saved train_sequences: {output_dir / 'train_sequences.npy'}")
print(f"Saved test_sequences:  {output_dir / 'test_sequences.npy'}")
print(f"Saved train_labels:    {output_dir / 'train_labels.npy'}")
print(f"Saved test_labels:     {output_dir / 'test_labels.npy'}")

print('Done.')