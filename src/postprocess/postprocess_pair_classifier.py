"""
Postprocessing of model predictions.

TODO:
- [ ] utilize config.yml
- [ ] utilize weights & biases

The plots related to model predictions are very helpful. We may explore it more in the future. Next, I'd like to create some results for the other stages. Specifically, I'm thinking about creating some stats for the src/datasets/dataset_segment_pairs.py. This script, generate three files: train_pairs.csv, val_pairs.csv, test_pairs.csv. Do you have access to this file? Here are some ideas:
1. Histogram of segmant pairs in each of the train/val/test sets, grouped by positive and negative groups
2. Histogram of unique isolates in each set
In addition to the code, you may find useful the log: logs/dataset_segment_pairs.log
Would you suggest to put it in src/datasets/dataset_segment_pairs.py? Or, maybe something similar to src/postprocess/analyze_segment_classifier_results.py?

The plots related to datasets are very helpful (we may get back to it in the future). Next, let's work on Stage 3, the embeddings (src/embeddings/embeddings_segment_pairs.py). The script loads three datasets of proteins pairs: train_pairs.csv, val_pairs.csv, test_pairs.csv, and saves embeddings in data/embeddings/bunya/April_2025/esm2_embeddings.h5. While the input files contain the full protein sequences (cleaned for ESM-2), these sequences are truncated to 1022 residues for the embeddings. What intermediate results would you suggest to generate?

# Preprocssing:
# UMAP of ESM-ready proteins, color-coded by function or segment (see potential batches that could cause leakage)
# UMAP of ESM-ready but truncated (to 1022) proteins, color-coded by function or segment

# Embeddings:
# UMAP of protein embeddings, color-coded by function or segment
# UMAP of protein embeddings but truncated (to 1022) proteins, color-coded by function or segment

# Can we apply some clustering (and maybe quantify clustering quality)?
"""
