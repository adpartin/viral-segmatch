# old_scripts/

These scripts are **superseded** by the current pipeline and are kept for historical reference only. They are **not maintained**.

## What replaced them

| Old script | Replaced by |
|------------|-------------|
| `Dec_scripts/esm2_*.sh` | `scripts/stage2_esm2.sh` |
| `Dec_scripts/dataset_*.sh` | `scripts/stage3_dataset.sh` |
| `Dec_scripts/classifier_*.sh` | `scripts/stage4_train.sh` |
| `Oct_scripts/compute_esm2_embeddings*` | `src/embeddings/compute_esm2_embeddings.py` |
| `Oct_scripts/dataset_segment_pairs*` | `src/datasets/dataset_segment_pairs.py` |
| `Oct_scripts/train_esm2_frozen_pair_classifier*` | `src/models/train_esm2_frozen_pair_classifier.py` |
| `Oct_scripts/analyze_segment_classifier_results.py` | `src/analysis/analyze_stage4_train.py` |
| `Oct_scripts/create_presentation_plots.py` | `src/analysis/create_presentation_plots.py` |

Do not run these scripts. Do not update them to match the current pipeline.
They are preserved here so that git history remains accessible without needing to recover deleted files.
