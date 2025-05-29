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
Process protein data from GTO files for Bunyavirales, including segment assignment and duplicate handling.
  ```bash
  python src/preprocess/preprocess_bunya_protein.py
  ```
* Input: GTO files in `data/raw/Anno_Updates/April_2025/bunya-from-datasets/Quality_GTOs`
* Output: Processed CSVs in `data/processed/bunya/April_2025` (e.g., `protein_filtered.csv`)

2. **Generate Segment Pair Dataset**
Create a dataset of segment pairs for training, validating, and testing a segment matcher.
```bash
python src/datasets/dataset_segment_pairs.py
```
* Input: `protein_filtered.csv` from preprocessing
* Output: Dataset CSV in `data/processed/bunya/April_2025/segmatch` (e.g., `train_pairs.csv`, `val_pairs.csv`, `test_pairs.csv`)

3. **Train Segment Matcher (Placeholder)**
Train a model to match genomic segments (script not yet implemented).
```bash
# Placeholder: script not yet available
python src/models/train_segment_matcher.py
```
* Input: `train_pairs.csv`, `val_pairs.csv`, `test_pairs.csv`, and more (TBD)
* Output: Trained model (TBD)

## Contributing
Contributions are welcome! Please open a pull request or issue for bugs, features, or improvements.
