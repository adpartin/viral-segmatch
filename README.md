# Viral-SegMatch
A pipeline for analyzing and processing genomic and proteomic data from viral isolates to address a key challenge in segmented RNA virus databases: determining if two viral segments belong to the same isolate. Specifically designed for viruses like Bunyavirales, it provides predictions crucial for understanding viral assembly and evolution.


## Setup and Installation

### Prerequisites
- python 3.9+
- torch 2.6+
- transformers 4.49
- scikit-learn 1.6

### Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/adpartin/viral-segmatch.git
   cd viral-segmatch
   ```

2. **Create and Activate Conda Environment**
   ```bash
   conda env create -f environment.yml
   conda activate viral-segmatch
   ```

### Running the Pipeline
1. **Preprocess Protein Data**

Process protein data from GTO files for Bunyavirales, including canonical segment assignment (S, M, L) and duplicate handling (e.g., GCA/GCF).
  ```bash
  python src/preprocess/preprocess_bunya_protein.py
  ```
* Input: GTO files (e.g., `data/raw/Anno_Updates/April_2025/bunya-from-datasets/Quality_GTOs`)
* Output: `data/processed/bunya/April_2025/protein_filtered.csv` (columns: `brc_fea_id`, `prot_seq`, `assembly_id`, `function`, `canonical_segment`, etc.).


2. **Generate Segment Pair Dataset**

Create a dataset of segment pairs for training, validating, and testing a segment matcher.
```bash
python src/datasets/dataset_segment_pairs.py
```
* Input: `protein_filtered.csv`
* Output: `train_pairs.csv`, `val_pairs.csv`, `test_pairs.csv` containing protein pairs are saved in `data/processed/bunya/April_2025/segment_pairs_classifier` (positive: same isolate, negative: different isolates).


3. **Compute ESM-2 Embeddings**
Computes ESM-2 embeddings for each unique protein sequence, keyed by brc_fea_id.
```bash
python src/embeddings/compute_esm2_embeddings.py
```
* Input: `protein_filtered.csv`
* Output: `data/embeddings/bunya/April_2025/esm2_embeddings.h5` (HDF5 file with embeddings, e.g., 1280D per protein).


4. **Train Segment Matcher (Placeholder)**

Loads pair data, retrieves embeddings for `brc_a` and `brc_b` from `esm2_embeddings.h5`, and train model to predict whether two protein belong to the same viral isolate
```bash
# Placeholder: script not yet available
python src/models/train_esm2_frozen_pair_classifier.py
```
* Input: `train_pairs.csv`, `val_pairs.csv`, `test_pairs.csv`, `esm2_embeddings.h5`
* Output: Trained model (e.g., `model.pt`), metrics (F1, AUC-ROC), and logs.


### Visualizations
Generate histograms and box plots of protein lengths by canonical segment.
```bash
python src/eda/visualize_protein_lengths.py
```
* Input: `data/processed/bunya/April_2025/protein_filtered.csv`
* Output: Plots in `eda/bunya/April_2025`


## Contributing
Contributions are welcome! Please open a pull request or issue for bugs, features, or improvements.
