"""
Preprocess data for training a model to classify proteins as cytosolic or
membrane-bound.
"""
import os
import requests
from pathlib import Path

import numpy as np
import pandas as pd

from io import BytesIO
from sklearn.model_selection import train_test_split

seed = 42  # for reproducibility (config!)

filepath = Path(__file__).resolve().parent

task_name = 'cyto_vs_memb' # config!
data_dir = filepath / '../../data' # global config!
raw_data_dir = data_dir / 'raw' # global config!

output_dir = data_dir / task_name

"""
To get this query URL, we searched for `(organism_id:9606) AND
(reviewed:true) AND (length:[80 TO 500])` on UniProt to get a list of
reasonably-sized human proteins, then selected 'Download', and set the
format to TSV and the columns to `Sequence` and `Subcellular location [CC]`,
since those contain the data we care about for this task.

Once that's done, selecting `Generate URL for API` gives you a URL you can
pass to Requests. Alternatively, can just download the data through the web
interface and open the file locally.
"""
# print("Querying data from UniProt ...")
# query_url = "https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Csequence%2Ccc_subcellular_location&format=tsv&query=%28%28organism_id%3A9606%29%20AND%20%28reviewed%3Atrue%29%20AND%20%28length%3A%5B80%20TO%20500%5D%29%29"
# uniprot_request = requests.get(query_url)
# bio = BytesIO(uniprot_request.content)
# df = pd.read_csv(bio, compression='gzip', sep='\t')

print("Loading protein data ...")
fname = 'uniprotkb_organism_id_9606_AND_reviewed_2025_03_11.tsv.gz'
df = pd.read_csv(raw_data_dir / fname, compression='gzip', sep='\t')

df = df.dropna()  # Drop proteins with missing columns

# Get info whether proteins are cytosolic or membrane-bound
cytosolic = df['Subcellular location [CC]'].str.contains("Cytosol|Cytoplasm", regex=True)
membrane = df['Subcellular location [CC]'].str.contains("Membrane|Cell membrane", regex=True)

# Split the data into cytosolic and membrane-bound proteins
cytosolic_df = df[cytosolic & ~membrane]  # Cytosolic only (cytosol or cytoplasm)
membrane_df = df[membrane & ~cytosolic]   # Membrane only (membrane or cell membrane)

# Cytosolic sequences (x) and labels (y)
cytosolic_sequences = cytosolic_df['Sequence'].tolist()
cytosolic_labels = [0 for protein in cytosolic_sequences]

# Membrane sequences (x) and labels (y)
membrane_sequences = membrane_df['Sequence'].tolist()
membrane_labels = [1 for protein in membrane_sequences]

# Concatenate the lists to get the `sequences` and `labels`
sequences = cytosolic_sequences + membrane_sequences
labels = cytosolic_labels + membrane_labels

# Quick check to check we got it right
assert len(sequences) == len(labels), "Sequences and labels should have the same length"

# Split
train_sequences, test_sequences, train_labels, test_labels = train_test_split(
    sequences, labels, test_size=0.25, shuffle=True, random_state=seed)
print(f'Train sequences: {len(train_sequences)}')
print(f'Test sequences:  {len(test_sequences)}')
print(f'type(train_sequences):    {type(train_sequences)}')
print(f'type(train_sequences[0]): {type(train_sequences[0])}')
print(f'train_sequences[0]:       {train_sequences[0]}')

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print("Saving data (sequences and labels) ...")
np.save(output_dir / 'train_sequences.npy', train_sequences)
np.save(output_dir / 'test_sequences.npy', test_sequences)
np.save(output_dir / 'train_labels.npy', train_labels)
np.save(output_dir / 'test_labels.npy', test_labels)

print(f"Saved train_sequences to: {output_dir / 'train_sequences.npy'}")
print(f"Saved test_sequences to:  {output_dir / 'test_sequences.npy'}")
print(f"Saved train_labels to:    {output_dir / 'train_labels.npy'}")
print(f"Saved test_labels to:     {output_dir / 'test_labels.npy'}")

print("Done!")