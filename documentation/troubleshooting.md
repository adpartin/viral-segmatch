# Troubleshooting Guide

Common errors when running the viral-segmatch pipeline, and what to do
about them. For technical documentation, see [`../docs/`](../docs/).

## Errors during Stage 3 (dataset construction) or Stage 4 (training)

### `FileNotFoundError: Test results file not found`

The training run didn't finish, or you're pointing at the wrong run
directory.

```bash
# List recent training runs
ls -lt models/flu/July_2025/runs/ | head -5

# Find the most recent training run for a bundle
ls -td models/flu/July_2025/runs/training_flu_ha_na_* | head -1

# Re-run training against an existing dataset directory
./scripts/stage4_train.sh --config_bundle flu_ha_na --cuda_name cuda:0 \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_YYYYMMDD_HHMMSS
```

### `Can't find dataset directory`

Stage 4's shell script requires `--dataset_dir` explicitly. It does not
extract a bundle name from the path or autodiscover the latest
dataset.

```bash
# List all dataset runs for the bundle
ls -lt data/datasets/flu/July_2025/runs/dataset_flu_ha_na_* | head -5

# Use the most recent
DATASET_DIR=$(ls -td data/datasets/flu/July_2025/runs/dataset_flu_ha_na_* | head -1)
./scripts/stage4_train.sh --config_bundle flu_ha_na --cuda_name cuda:0 --dataset_dir "$DATASET_DIR"
```

### `ConfigKeyError: Missing key …`

Bundle name typo or trying to read a key that isn't in the resolved
config.

```bash
# Check available bundles
ls conf/bundles/*.yaml

# Inspect what's actually in the resolved config for a bundle
python -c "
from src.utils.config_hydra import get_virus_config_hydra, print_config_summary
config = get_virus_config_hydra('flu_ha_na')
print_config_summary(config)
"
```

When accessing config keys in code, use dotted access (`config.virus.virus_name`)
rather than dict-style (`config['virus_name']`).

### Stage 3 raises on `seq_hash_overlap` or `dna_hash_overlap`

The v2 builder's hard-fail audit detected cross-split sequence-hash
overlap on the active hash family (`hash_key=seq` by default, or
`hash_key=dna`). This usually means:

- An upstream change to the routing logic introduced a bug.
- The bundle has an incompatible combination of `split_strategy.mode`
  and `hash_key`.

The audit dict (`split_strategy_audit` in the run directory) reports
both `seq_hash_overlap` and `dna_hash_overlap` regardless of which
family is active, so you can see the diagnostic side too. See
[`../docs/methods/leakage_definitions.md`](../docs/methods/leakage_definitions.md)
(mode #3) for the full taxonomy.

## Data setup issues

### Missing input files

```bash
# Stage 1 outputs (must run first if missing)
ls -la data/processed/flu/July_2025/protein_final.csv data/processed/flu/July_2025/genome_final.csv

# Stage 2 outputs
ls -la data/embeddings/flu/July_2025/master_esm2_embeddings.h5
ls -la data/embeddings/flu/July_2025/kmer_features_k6.npz

# Re-run a missing stage
./scripts/stage1_preprocess_flu.sh  --config_bundle flu_base
./scripts/stage2_esm2.sh           --config_bundle flu_base --cuda_name cuda:0
./scripts/stage2b_kmer.sh          --config_bundle flu_base
```

### Insufficient pairs after Stage 3

If `dataset_stats.json` shows a tiny `split_sizes`, check whether the
bundle's metadata filters (host, year, hn_subtype) are too restrictive
for the dataset:

```bash
# Inspect the bundle for filter knobs
grep -E "host|year|hn_subtype|max_isolates_to_process" conf/bundles/flu_ha_na.yaml

# Or lower max_isolates_to_process for a quick smoke test
# (override at the CLI rather than editing the bundle)
python src/datasets/dataset_segment_pairs.py \
    --config_bundle flu_ha_na \
    --override dataset.max_isolates_to_process=500
```

## Python environment issues

### `ModuleNotFoundError: No module named 'fair_esm' / 'lightgbm' / ...`

Activate the conda env and reinstall:

```bash
conda activate cepi
pip install fair-esm transformers h5py lightgbm
```

### CUDA not available or out of memory

```bash
# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Force CPU
export CUDA_VISIBLE_DEVICES=""

# Lower batch size in the bundle override
# (or edit conf/training/base.yaml directly for a global change)
./scripts/stage4_train.sh --config_bundle flu_ha_na --cuda_name cuda:0 \
    --dataset_dir <DIR> training.batch_size=16
```

## Post-hoc analysis issues

### `analyze_stage4_train.py` complains about missing `test_predicted.csv`

The training run didn't finish or its output is in a different directory.
Pass `--model_dir` explicitly:

```bash
python src/analysis/analyze_stage4_train.py --config_bundle flu_ha_na \
    --model_dir models/flu/July_2025/runs/training_flu_ha_na_YYYYMMDD_HHMMSS
```

### `aggregate_baselines_vs_mlp.py` finds no matching runs

The aggregator autodiscovers runs by bundle name with anchored matching
(so `--bundle flu_ha_na` does NOT pick up `flu_ha_na_regimes`).
For an explicit list of run directories instead of autodiscovery:

```bash
python src/analysis/aggregate_baselines_vs_mlp.py --model_dirs \
    models/flu/July_2025/runs/training_flu_ha_na_YYYYMMDD_HHMMSS \
    models/flu/July_2025/runs/baseline_lgbm_flu_ha_na_YYYYMMDD_HHMMSS \
    models/flu/July_2025/runs/baseline_knn1_margin_flu_ha_na_YYYYMMDD_HHMMSS
```

It cross-checks `training_info.json::dataset_dir` across the picks and
refuses to aggregate on a mismatch — if you get a refusal there, the
runs aren't from the same dataset build.

## Debugging tips

### Check the logs

Every shell-wrapped stage writes a log under `logs/`. Tail the most
recent log to see where a failure happened:

```bash
ls -t logs/datasets/ | head -3
ls -t logs/training/ | head -3
tail -200 logs/training/<file>.log
```

### Test on a tiny subset first

Most bundles support `dataset.max_isolates_to_process` for a quick
smoke test:

```bash
python src/datasets/dataset_segment_pairs.py \
    --config_bundle flu_ha_na \
    --override dataset.max_isolates_to_process=200
```

The resulting dataset directory will run end-to-end in minutes.

### Verify configuration before running

```bash
python -c "
from src.utils.config_hydra import get_virus_config_hydra, print_config_summary
config = get_virus_config_hydra('flu_ha_na')
print_config_summary(config)
"
```

## Related documentation

- [`../docs/conf_guide.md`](../docs/conf_guide.md) — Hydra configuration system.
- [`../docs/methods/pipeline_overview.md`](../docs/methods/pipeline_overview.md) — pipeline architecture.
- [`../docs/methods/leakage_definitions.md`](../docs/methods/leakage_definitions.md) — leakage taxonomy.
- [`quick-start.md`](quick-start.md) — first-time setup.
