"""
Build train/val/test sets of labeled protein pairs for the segment-matching task.

Goal
----
Create labeled protein pairs for the task:
  label=1  => two proteins co-occur within the same isolate (same assembly_id)
  label=0  => two proteins are sampled from different isolates AND are blocked from
              being a negative if their sequences ever co-occur in ANY isolate
              (see "blocked negatives" below).

Important nuance for label=0:
  A negative here means "not observed co-occurring (and not allowed to contradict any
  observed co-occurrence)". It is NOT a proof of biological incompatibility; it is a
  dataset construction label.

The downstream classifier consumes embedding pairs derived from precomputed ESM-2
(e.g., [emb_a, emb_b], and optionally |A-B| and A*B).

Key columns:
- assembly_id: identifies isolates
- canonical_segment: virus-specific canonical segment label (e.g., L/M/S for bunyavirus;
  for influenza you may see segment identifiers depending on preprocessing)
- prot_seq: protein sequence
- function: protein function
- brc_fea_id: unique feature ID
- prot_hash: deterministic hash of prot_seq used for deduplication / pair keys

Key Concepts
------------
- pair_key:
    A canonical, order-invariant key derived from (prot_hash_a, prot_hash_b)
    (e.g., canonical_pair_key(prot_hash_a, prot_hash_b)).
    Used for:
      (1) deduplicating identical sequence pairs
      (2) preventing contradictory labeling (e.g., a pair that is both positive and negative)
      (3) preventing split leakage (same sequence-pair appearing in multiple splits)

- cooccur_pairs:
    Set of all pair_key values that are observed as positives (co-occurring within
    any isolate). A candidate negative is rejected if its pair_key is in cooccur_pairs.
    Note: this implementation is conservative — it blocks a sequence pair if the two
    sequences appear in the same isolate at all, regardless of whether that pair would
    have been included as a positive under additional constraints (e.g., different functions).

- canonicalization of stored orientation (optional):
    The semantic task is undirected: (a,b) ≡ (b,a).
    However, the default model input can be directed (e.g., [emb_a, emb_b]).
    To prevent "direction -> label" shortcut learning, we can enforce a deterministic
    orientation rule for BOTH positives and negatives by swapping all *_a/*_b fields
    when needed (e.g., order by (prot_hash, brc_fea_id) or a fallback).
    This does NOT change pair_key (pair_key is always order-invariant).

Dataset Construction Steps
--------------------------
1) Split isolates (assembly_id) into train/val/test
   - Splitting is performed at the isolate level (not pair level) to avoid leakage
     of isolate-specific signals across splits.

2) Create positive pairs within each isolate (label=1)
   - For each isolate, generate all within-isolate combinations of proteins:
       for (row_a, row_b) in combinations(isolate_rows, 2)
   - Keep only pairs that satisfy required constraints (currently: different brc_fea_id
     AND different function; same-function pairs are not used as positives).
   - For each positive pair, compute:
       pair_key = canonical_pair_key(row_a.prot_hash, row_b.prot_hash)   (stored as column 'pair_key')
     and store paired fields:
       assembly_id_a/b, brc_a/b, prot_seq_a/b, seg_a/b, func_a/b, prot_hash_a/b, label=1
   - (Optional) Canonicalize stored orientation by swapping all *_a/*_b fields
     according to a deterministic rule that does NOT reference label, function, or segment.

3) Build cooccur_pairs (blocking set for negatives)
   - Build a set of all sequence-hash pairs that appear together in any isolate.
   - Any candidate negative whose pair_key is in this set is rejected to prevent
     contradictory labels.

4) Create negative pairs across isolates (label=0)
   - Repeatedly sample two distinct isolates (aid1, aid2), then sample one random
     protein from each isolate.
   - Reject a candidate negative if:
       a) its pair_key is in cooccur_pairs (would contradict observed co-occurrence),
       b) it duplicates a previously added negative by brc_id-pair (symmetric duplicate),
       c) it duplicates a previously added negative by prot_hash-pair,
       d) it violates configured same-function constraints (optional ratio cap).
   - For accepted negatives, store the same paired fields as positives with label=0.
   - (Optional) Canonicalize stored orientation by the same deterministic rule used for positives.

5) Prevent split leakage by pair_key (order-invariant)
   - Ensure that the same pair_key does not appear in more than one split.
   - If overlaps exist, remove/resolve them (and log how many were removed per split/label).

6) Write outputs and run metadata
   - Save train_pairs.csv / val_pairs.csv / test_pairs.csv containing:
       brc_a, brc_b, label, pair_key, and associated paired metadata columns.
   - Save sampled_isolates.txt (when max_isolates_to_process is set).
   - Save isolate_metadata.csv (per-isolate metadata used for diagnostics).
   - Save dataset_stats.json (split sizes, label ratios, metadata distributions).
   - Save duplicate_stats.json (cooccurrence stats + negative rejection stats + pair_key overlaps).
   - Optionally save cooccurring_sequence_pairs.csv (for analysis).

Duplicate handling (blocked negatives):
- Identical amino-acid sequences can occur in multiple genomes/isolates
- A pair (prot_seq_a, prot_seq_b) can appear as positive (same isolate) AND negative (different isolates)
- This creates contradictory labels and potential data leakage
- Solution: Block negative pairs where sequences co-occur in ANY isolate
- Split by pair_key (not just isolate) to prevent same pair appearing in train and test
"""

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.timer_utils import Timer
from src.utils.config_hydra import get_virus_config_hydra, print_config_summary, save_config
from src.utils.seed_utils import resolve_process_seed, set_deterministic_seeds
from src.utils.path_utils import resolve_run_suffix, build_dataset_paths, load_dataframe
from src.utils.metadata_enrichment import enrich_prot_data_with_metadata
from src.datasets._pair_helpers import (
    canonical_pair_key,
    attach_ctg_dna_to_prot_df,
    select_balanced_isolate_pool,
    orient_pair_by_schema,
    compute_isolate_pair_counts,
    build_cooccurrence_set,
    get_metadata_distributions,
    filter_by_metadata,
)

total_timer = Timer()


# Removed 2026-05-11: generate_temporal_split() and dataset.year_train /
# dataset.year_test config keys. The temporal-holdout mechanism is superseded
# by the general metadata_holdout path (year-axis holdout is its degenerate
# case). See docs/plans/2026-05-11_metadata_holdout_plan.md.


# Parser
parser = argparse.ArgumentParser(description='Create segment pairs dataset')
parser.add_argument(
    '--config_bundle',
    type=str, default=None,
    help='Config bundle to use (e.g., flu_a, bunya).'
)
parser.add_argument(
    '--input_file',
    type=str, default=None,
    help='Path to input CSV file (e.g., protein_final.csv). If not provided, derived from config.'
)
parser.add_argument(
    '--output_dir',
    type=str, default=None,
    help='Path to output directory for datasets. If not provided, derived from config.'
)
parser.add_argument(
    '--run_output_subdir',
    type=str, default=None,
    help='Optional subdirectory name under default output_dir (e.g., experiment/run id).'
)
parser.add_argument(
    '--override',
    type=str, nargs='+', default=None,
    help='Hydra-style dotlist overrides applied on top of the bundle (e.g., '
         'dataset.hn_subtype=H3N2 dataset.host=human). Useful for filter sweeps '
         'without creating new bundles.'
)
args = parser.parse_args()

# Load config
config_path = str(project_root / 'conf')  # Pass the config path explicitly
config_bundle = args.config_bundle
if config_bundle is None:
    raise ValueError("Must provide --config_bundle")
config = get_virus_config_hydra(config_bundle, config_path=config_path)
# Apply CLI dotlist overrides (e.g., dataset.hn_subtype=H3N2)
if args.override:
    from omegaconf import OmegaConf
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.override))
    print(f"Applied CLI overrides: {args.override}")
print_config_summary(config)

# Extract config values. v1-only and v2-only knobs are read inside their
# dispatch branches below, not here.
VIRUS_NAME = config.virus.virus_name
DATA_VERSION = config.virus.data_version
RANDOM_SEED = resolve_process_seed(config, 'datasets')
NEG_TO_POS_RATIO = config.dataset.neg_to_pos_ratio
TRAIN_RATIO = config.dataset.train_ratio
VAL_RATIO = config.dataset.val_ratio
PAIR_MODE = getattr(config.dataset, 'pair_mode', 'schema_ordered')
SCHEMA_PAIR_RAW = getattr(config.dataset, 'schema_pair', None)
MAX_ISOLATES_TO_PROCESS = getattr(config.dataset, 'max_isolates_to_process', None)
SHUFFLE_TRAIN_LABELS = getattr(config.dataset, 'shuffle_train_labels', False)
SHUFFLE_TRAIN_LABELS_SEED = getattr(config.dataset, 'shuffle_train_labels_seed', None)
N_FOLDS = getattr(config.dataset, 'n_folds', None)
GENERATE_VISUALIZATIONS = getattr(config.dataset, 'generate_visualizations', True)
SKIP_ESM_PCA_PLOTS = getattr(config.dataset, 'skip_esm_pca_plots', False)
SKIP_KMER_PCA_PLOTS = getattr(config.dataset, 'skip_kmer_pca_plots', False)

# Pair builder selection (default v1 for backward compatibility).
PAIR_BUILDER_VERSION = getattr(config.dataset, 'pair_builder_version', 'v1')

# Loud rejection of legacy year_train / year_test keys. They were removed
# 2026-05-11 (replaced by metadata_holdout under v2). Surface a clear migration
# message rather than letting OmegaConf return None silently if a bundle still
# carries them.
for _legacy_key in ('year_train', 'year_test'):
    if getattr(config.dataset, _legacy_key, None) is not None:
        raise ValueError(
            f"dataset.{_legacy_key} is no longer supported; the temporal-holdout "
            f"mechanism was replaced by dataset.metadata_holdout under "
            f"pair_builder_version=v2 on 2026-05-11. See "
            f"docs/plans/2026-05-11_metadata_holdout_plan.md."
        )

print(f"\n{'='*40}")
print(f"Virus: {VIRUS_NAME}")
print(f"Config bundle: {config_bundle}")
print(f"{'='*40}")

# Resolve run suffix (manual override in config or auto-generate from sampling params)
RUN_SUFFIX = resolve_run_suffix(
    config=config,
    max_isolates=MAX_ISOLATES_TO_PROCESS,  # Dataset-specific isolate sampling
    seed=RANDOM_SEED,
    auto_timestamp=True
)

# Set deterministic seeds for reproducible dataset creation
if RANDOM_SEED is not None:
    set_deterministic_seeds(RANDOM_SEED, cuda_deterministic=False) # No CUDA for dataset creation
    print(f'Set deterministic seeds for dataset creation (seed: {RANDOM_SEED})')
else:
    print('No seed set - dataset creation will be non-deterministic')

# Build dataset paths
paths = build_dataset_paths(
    project_root=project_root,
    virus_name=VIRUS_NAME,
    data_version=DATA_VERSION,
    run_suffix=RUN_SUFFIX,
    config=config
)

# Default input and output paths are derived from the config and virus, but can be overridden via CLI args.
default_input_file = paths['input_file']
default_output_dir = paths['output_dir']

# Apply CLI overrides if provided
input_file = Path(args.input_file) if args.input_file else default_input_file
if args.output_dir:
    output_dir = Path(args.output_dir)
elif args.run_output_subdir:
    # Always use runs/ subdirectory for consistency
    # Structure: data/datasets/{virus}/{data_version}/runs/{run_id}/
    # run_id includes config_bundle name: dataset_{config_bundle}_{timestamp}
    output_dir = default_output_dir / 'runs' / args.run_output_subdir
else:
    # Fallback: create a run directory with config bundle name
    # This shouldn't happen if shell script is used correctly
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fallback_run_id = f"dataset_{config_bundle}_{timestamp}"
    output_dir = default_output_dir / 'runs' / fallback_run_id
    print(f"WARNING: No run_output_subdir provided, using fallback: {fallback_run_id}")
output_dir.mkdir(parents=True, exist_ok=True)

# Save resolved config snapshot for reproducibility
save_config(config, str(output_dir / 'resolved_config.yaml'))
print(f'Saved resolved config snapshot to: {output_dir / "resolved_config.yaml"}')

print(f'\nConfig bundle:  {config_bundle}')
print(f'Run suffix:     {RUN_SUFFIX if RUN_SUFFIX else "(none)"}')
print(f'Input file:     {input_file}')
print(f'Output dir:     {output_dir}')
print(f'Run ID:         {args.run_output_subdir if args.run_output_subdir else "auto-generated"}')

# Load protein data
print('\nLoad preprocessed protein sequence data.')
try:
    prot_df = load_dataframe(input_file)
    print(f"Loaded {len(prot_df):,} protein records")
except FileNotFoundError:
    raise FileNotFoundError(f"Data file not found at: {input_file}")
except Exception as e:
    raise RuntimeError(f"Error loading data from {input_file}: {e}")

# Attach DNA sequences to protein dataframe (carried through into the output
# pair CSVs so we can run nucleotide-level leakage audits post-hoc without
# re-running Stage 3).
print('\nAttach DNA sequences to protein dataframe.')
prot_df = attach_ctg_dna_to_prot_df(prot_df, input_file)

# Enrich the df with metadata (e.g., host, year, hn_subtype)
print('\nEnrich dataframe with metadata.')
prot_df = enrich_prot_data_with_metadata(prot_df, project_root=project_root)

# Drop isolates with non-conforming hn_subtype (ambiguous typing). See
# drop_ambiguous_hn_subtype docstring in _pair_helpers.py for the rationale.
# Toggle via `dataset.drop_ambiguous_subtype` (default true). Setting it false
# preserves the legacy behavior where ~1,006 'HN' / 'H?N' isolates flow
# through to the regime sampler as `unknown_metadata_neg`.
if bool(getattr(config.dataset, 'drop_ambiguous_subtype', True)):
    from src.datasets._pair_helpers import drop_ambiguous_hn_subtype as _drop_ambig
    prot_df, _ambig_summary = _drop_ambig(prot_df)
    if _ambig_summary['n_isolates_dropped'] > 0:
        print(f"\nWARNING: dropped {_ambig_summary['n_isolates_dropped']:,} "
              f"isolates with non-HxNy hn_subtype: "
              f"{_ambig_summary['value_counts']}")
        print(f"   Protein records: "
              f"{_ambig_summary['n_rows_total']:,} -> "
              f"{_ambig_summary['n_rows_kept']:,} "
              f"(dropped {_ambig_summary['n_rows_dropped']:,} rows)")

# Filter isolates by metadata criteria (host, year, hn_subtype, geo_location, passage)
# year vs year_range: mutually exclusive; year for scalar/list set membership,
# year_range for inclusive [min, max]. The helper enforces the exclusion.
# OmegaConf returns ListConfig for list-typed fields; coerce to Python list
# at the boundary so filter_by_metadata's isinstance(..., (list, tuple))
# checks succeed.
from omegaconf import ListConfig
def _coerce_filter(v):
    return list(v) if isinstance(v, ListConfig) else v
hn_subtype_filter   = _coerce_filter(getattr(config.dataset, 'hn_subtype', None))
host_filter         = _coerce_filter(getattr(config.dataset, 'host', None))
year_filter         = _coerce_filter(getattr(config.dataset, 'year', None))
year_range_filter   = _coerce_filter(getattr(config.dataset, 'year_range', None))
geo_location_filter = _coerce_filter(getattr(config.dataset, 'geo_location', None))
passage_filter      = _coerce_filter(getattr(config.dataset, 'passage', None))
prot_df = filter_by_metadata(
    prot_df,
    hn_subtype=hn_subtype_filter,
    host=host_filter,
    year=year_filter,
    year_range=year_range_filter,
    geo_location=geo_location_filter,
    passage=passage_filter,
)


# Jim's balancing request
# Subtype balancing (downsample to equal per-subtype representation).
# No-op when mode=natural, which is the default and preserves natural subtype distribution.
# See docs/plans/subtype_balancing_plan.md.
# Mutually exclusive with max_isolates_to_process (both sample isolates) and
# with dataset.hn_subtype (single-subtype filter contradicts multi-subtype balancing).
print("\nApply subtype balancing (if requested)...")
subtype_sel_cfg = getattr(config.dataset, 'subtype_selection', None)
subtype_sel_mode = getattr(subtype_sel_cfg, 'mode', 'natural') if subtype_sel_cfg is not None else 'natural'
if subtype_sel_mode == 'balanced':
    # Subtype balancing and max_isolates_to_process can't be used together (since
    # both aim to sub-sample isolates so it's unclear which should take precedence)
    if MAX_ISOLATES_TO_PROCESS is not None:
        raise ValueError(
            f"dataset.subtype_selection.mode=balanced is incompatible with "
            f"dataset.max_isolates_to_process={MAX_ISOLATES_TO_PROCESS} (both perform sub-sampling of isolates). "
            "Set max_isolates_to_process=null or subtype_selection.mode=natural."
        )
    # Subtype balancing and single-subtype filtering are incompatible (contradictory goals)
    if hn_subtype_filter is not None:
        raise ValueError(
            f"dataset.subtype_selection.mode=balanced is incompatible with "
            f"dataset.hn_subtype={hn_subtype_filter!r} (single-subtype filter contradicts "
            "multi-subtype balancing). Unset the single-subtype filter or use "
            "subtype_selection.mode=natural."
        )
prot_df = select_balanced_isolate_pool(
    prot_df,
    subtype_sel_cfg,
    master_seed=RANDOM_SEED,
    output_dir=output_dir,
)

# Sample a subset of isolates from the dataframe
sampled_isolates_file = output_dir / 'sampled_isolates.txt'
if MAX_ISOLATES_TO_PROCESS:
    unique_isolates = prot_df['assembly_id'].unique()
    total_isolates = len(unique_isolates)
    print(f"\nSample {MAX_ISOLATES_TO_PROCESS} isolates (out of {total_isolates} total unique isolates).")

    if MAX_ISOLATES_TO_PROCESS >= total_isolates:
        sampled_isolates = sorted(unique_isolates)
        print("Requested isolates more than available; using all isolates.")
    else:
        print(f"Sampling {MAX_ISOLATES_TO_PROCESS} isolates (seed: {RANDOM_SEED}).")
        sampled_isolates = np.random.choice(
            unique_isolates,
            size=MAX_ISOLATES_TO_PROCESS,
            replace=False,
        )
        sampled_isolates = sorted(sampled_isolates)
    sampled_isolates_file.parent.mkdir(parents=True, exist_ok=True)
    with open(sampled_isolates_file, 'w') as f:
        for isolate in sampled_isolates:
            f.write(f"{isolate}\n")
    print(f"Wrote list of {len(sampled_isolates)} sampled isolates to {sampled_isolates_file}")

    before_count = len(prot_df)
    prot_df = prot_df[prot_df['assembly_id'].isin(sampled_isolates)].reset_index(drop=True)
    print(f"Filtered {len(prot_df)} protein records from {before_count} after isolate sampling.")

# Restrict to config.virus.selected_functions. Both Flu A and Bunya require
# this: Flu A picks major protein functions; Bunya picks core L/M/S.
if not (hasattr(config.virus, 'selected_functions') and config.virus.selected_functions):
    raise ValueError("config.virus.selected_functions must be set and non-empty")
if 'function' not in prot_df.columns:
    raise ValueError("'function' column not found in protein data")
print(f"Filtering to selected functions: {config.virus.selected_functions}")
mask = prot_df['function'].isin(config.virus.selected_functions)
df = prot_df[mask].reset_index(drop=True)
print(f"Filtered {len(df)} protein records from {len(prot_df)} based on selected functions.")

# Add sequence hash for duplicate detection (used by canonical_pair_key, cooccur_pairs, etc.).
# Stage 1 (preprocess_flu.py) already writes `prot_hash` to protein_final.csv; reuse it when
# present so the hash algorithm lives in exactly one place.
if 'prot_hash' not in df.columns:
    df['prot_hash'] = df['prot_seq'].apply(lambda x: hashlib.md5(str(x).encode()).hexdigest())

# Schema-pair parsing (shared by v1 and v2). PAIR_MODE validity check and
# v1-only notes (canonicalize override, same-func controls) live in the v1
# dispatch branch; v2 enforces schema_ordered via _validate_v2_config.
schema_pair: Optional[Tuple[str, str]] = None

if PAIR_MODE == "schema_ordered":
    if SCHEMA_PAIR_RAW is None:
        raise ValueError("dataset.schema_pair must be set when dataset.pair_mode='schema_ordered'")
    schema_list = list(SCHEMA_PAIR_RAW)
    if len(schema_list) != 2:
        raise ValueError(f"dataset.schema_pair must be length 2, got: {schema_list!r}")
    f0, f1 = str(schema_list[0]), str(schema_list[1])
    if f0 == f1:
        raise ValueError(f"dataset.schema_pair must contain two different functions, got: {schema_list!r}")

    # Canonicalize schema order using virus.protein_order so [A,B] and [B,A]
    # produce identical datasets. Without this, unit_diff would silently
    # sign-flip between bundles that differ only in YAML ordering.
    if not (hasattr(config.virus, 'protein_order') and config.virus.protein_order):
        raise ValueError(
            "config.virus.protein_order must be set and non-empty for pair_mode='schema_ordered'"
        )
    canonical_order = list(config.virus.protein_order)
    unknown = [f for f in (f0, f1) if f not in canonical_order]
    if unknown:
        raise ValueError(
            f"dataset.schema_pair contains functions not in virus.protein_order: {unknown!r}. "
            f"Add them to conf/virus/{VIRUS_NAME}.yaml:protein_order or fix the bundle."
        )
    # Ensure canonical order for left and right functions in the pair
    if canonical_order.index(f0) <= canonical_order.index(f1):
        func_left, func_right = f0, f1
    else:
        func_left, func_right = f1, f0
        print(
            f"INFO: schema_pair reordered to canonical segment order: "
            f"({f0!r}, {f1!r}) -> ({func_left!r}, {func_right!r})"
        )
    schema_pair = (func_left, func_right)

    # Validate that schema functions exist in the filtered dataframe.
    available_funcs = set(df["function"].unique().tolist()) if "function" in df.columns else set()
    missing = [f for f in [func_left, func_right] if f not in available_funcs]
    if missing:
        raise ValueError(
            "schema_ordered mode: schema_pair functions missing from filtered data. "
            f"missing={missing!r}, available_n={len(available_funcs):,}"
        )

# Validate brc_fea_id uniqueness within isolates
dups = df[df.duplicated(subset=['assembly_id', 'brc_fea_id'], keep=False)]
if not dups.empty:
    raise ValueError(f"Duplicate brc_fea_id found within isolates: \
        {dups[['assembly_id', 'brc_fea_id']]}")

# filters_applied dict reused by save_split_output_v2 for provenance
filters_applied = {
    'hn_subtype': hn_subtype_filter,
    'host': host_filter,
    'year': year_filter,
    'year_range': list(year_range_filter) if year_range_filter is not None else None,
    'geo_location': geo_location_filter,
    'passage': passage_filter,
    'pair_mode': PAIR_MODE,
    'schema_pair': list(schema_pair) if schema_pair is not None else None,
}

if PAIR_BUILDER_VERSION == 'v2':
    # v2 dispatch. v2 is opt-in via dataset.pair_builder_version=v2; v1 path
    # below is unchanged. See docs/plans/done/design_dataset_gen_v2.md.

    # v2-only config knobs.
    MAX_ATTEMPTS_PER_SEQ = getattr(config.dataset, 'max_attempts_per_seq', 50)
    AXES_FOR_FLAGS = list(getattr(config.dataset, 'axes_for_flags',
                                  ['hn_subtype', 'host', 'year', 'geo_location', 'passage']))

    NEG_SAMPLING_CFG = getattr(config.dataset, 'negative_sampling', None)
    AXIS_QUOTAS = None
    SAMPLING_AXES = None
    YEAR_MATCH = 'binned'
    YEAR_BIN_EDGES = None
    ON_SHORTFALL = 'redistribute'
    # Regime-aware coverage gate; default False keeps existing builds identical.
    # See docs/plans/2026-05-14_regime_aware_coverage_plan.md.
    REGIME_AWARE_COVERAGE = False

    # Split-strategy dispatch (validated by _validate_v2_config below). Default
    # 'random' preserves existing v2 behavior. See
    # docs/plans/2026-05-10_seq_disjoint_routing_plan.md and
    # docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md.
    SPLIT_STRATEGY_CFG = getattr(config.dataset, 'split_strategy', None)
    SPLIT_STRATEGY_MODE = 'random'
    # hash_key picks the routing hash family when mode=seq_disjoint:
    # 'seq' = protein (default, stricter); 'dna' = nucleotide (looser, but
    # appropriate when training features are DNA-derived such as k-mer).
    SPLIT_STRATEGY_HASH_KEY = 'seq'
    # cluster_id_path and cluster_id_threshold only consumed when mode='cluster_disjoint'.
    # cluster_alphabet picks aa vs nt clustering (default aa).
    # cds_final_path is mandatory when cluster_alphabet='nt' (Experiment B-nt).
    CLUSTER_ID_PATH = None
    CLUSTER_ID_THRESHOLD = None
    CLUSTER_ALPHABET = 'aa'
    CDS_FINAL_PATH = None
    # pair_key_alphabet (Phase 2 of clustering cleanup plan): which hash
    # family the pair_key is built on (aa = prot_hash, nt_cds = cds_dna_hash).
    # Default tied to cluster_alphabet for cluster_disjoint routings; 'aa'
    # for random / seq_disjoint. Explicit override via
    # `dataset.split_strategy.pair_key_alphabet` if a bundle needs a different
    # mapping (e.g., random routing + nt_cds dedup).
    PAIR_KEY_ALPHABET = 'aa'
    # single_slot is a cluster_disjoint-only sub-knob. None = bilateral (both
    # slots' clusters disjoint, default); 'a' or 'b' = constrain only that
    # slot's clusters, leaving the other slot unconstrained. Unlocks lower
    # idXX thresholds when bilateral cliffs into a mega-component — see
    # docs/results/2026-05-24_cluster_disjoint_feasibility_HA_NA.md.
    SINGLE_SLOT = None
    if SPLIT_STRATEGY_CFG is not None:
        m = getattr(SPLIT_STRATEGY_CFG, 'mode', None)
        if m is not None:
            SPLIT_STRATEGY_MODE = str(m)
        hk = getattr(SPLIT_STRATEGY_CFG, 'hash_key', None)
        if hk is not None:
            SPLIT_STRATEGY_HASH_KEY = str(hk)
        cp = getattr(SPLIT_STRATEGY_CFG, 'cluster_id_path', None)
        if cp is not None:
            CLUSTER_ID_PATH = str(cp)
        ct = getattr(SPLIT_STRATEGY_CFG, 'cluster_id_threshold', None)
        if ct is not None:
            CLUSTER_ID_THRESHOLD = float(ct)
        ca = getattr(SPLIT_STRATEGY_CFG, 'cluster_alphabet', None)
        if ca is not None:
            CLUSTER_ALPHABET = str(ca)
        cf = getattr(SPLIT_STRATEGY_CFG, 'cds_final_path', None)
        if cf is not None:
            CDS_FINAL_PATH = str(cf)
        ss = getattr(SPLIT_STRATEGY_CFG, 'single_slot', None)
        if ss is not None:
            SINGLE_SLOT = str(ss)
            if SINGLE_SLOT not in ('a', 'b'):
                raise ValueError(
                    f"dataset.split_strategy.single_slot must be 'a' or 'b' or null; "
                    f"got {SINGLE_SLOT!r}"
                )
        # pair_key_alphabet: explicit override; default inferred from
        # cluster_alphabet (or 'aa' if no cluster routing). Inference happens
        # after the CLUSTER_ALPHABET block above has run.
        pk = getattr(SPLIT_STRATEGY_CFG, 'pair_key_alphabet', None)
        if pk is not None:
            PAIR_KEY_ALPHABET = str(pk)
            if PAIR_KEY_ALPHABET not in ('aa', 'nt_cds', 'nt_ctg'):
                raise ValueError(
                    f"dataset.split_strategy.pair_key_alphabet must be 'aa', 'nt_cds', "
                    f"or 'nt_ctg', got {PAIR_KEY_ALPHABET!r}"
                )
        else:
            # Default inference: tie to cluster_alphabet for cluster_disjoint;
            # 'aa' otherwise. Matches the Phase 2 test plan's 4-bundle set
            # (cluster_nt_t099 -> nt_cds; cluster_t099 / regimes -> aa).
            if SPLIT_STRATEGY_MODE == 'cluster_disjoint' and CLUSTER_ALPHABET == 'nt_cds':
                PAIR_KEY_ALPHABET = 'nt_cds'
            else:
                PAIR_KEY_ALPHABET = 'aa'
    # drop_budget sub-knob (P2): mega-CC edge min-cut for bilateral 2D-CD.
    # Absent or enabled=false -> no cut (existing behavior, byte-identical).
    # Threaded split_dataset_v2 -> cluster_disjoint_route_pos_df -> _megacc_cut.
    # See docs/plans/2026-06-04_2d_cd_drop_budget_router_plan.md.
    DROP_BUDGET = None
    if SPLIT_STRATEGY_CFG is not None:
        _db = getattr(SPLIT_STRATEGY_CFG, 'drop_budget', None)
        if _db is not None:
            from omegaconf import OmegaConf
            DROP_BUDGET = OmegaConf.to_container(_db, resolve=True)

    # D3 feasibility knobs for k-fold cluster_disjoint (read from
    # split_strategy.feasibility.*; defaults match D3 of the k-fold plan).
    # See docs/plans/done/2026-05-27_kfold_variance_estimation_plan.md D3.
    MAX_ACCEPTABLE_DRIFT_PP = 0.05
    MIN_TEST_FRAC = 0.05
    if SPLIT_STRATEGY_CFG is not None:
        feas = getattr(SPLIT_STRATEGY_CFG, 'feasibility', None)
        if feas is not None:
            mdp = getattr(feas, 'max_acceptable_drift_pp', None)
            if mdp is not None:
                MAX_ACCEPTABLE_DRIFT_PP = float(mdp)
            mtf = getattr(feas, 'min_test_frac', None)
            if mtf is not None:
                MIN_TEST_FRAC = float(mtf)
    # Default CDS path: data/processed/<virus>/<version>/cds_dna_final.parquet
    # (alongside the input protein_final).
    if CLUSTER_ALPHABET == 'nt_cds' and CDS_FINAL_PATH is None:
        CDS_FINAL_PATH = str(input_file.parent / 'cds_dna_final.parquet')
    if NEG_SAMPLING_CFG is not None:
        from omegaconf import OmegaConf
        rt = OmegaConf.to_container(NEG_SAMPLING_CFG.regime_targets, resolve=True)
        AXIS_QUOTAS = {k: float(v) for k, v in rt.items()}
        ax = getattr(NEG_SAMPLING_CFG, 'axes', None)
        if ax is not None:
            SAMPLING_AXES = list(ax)
        ym = getattr(NEG_SAMPLING_CFG, 'year_match', None)
        if ym is not None:
            YEAR_MATCH = str(ym)
        yb = getattr(NEG_SAMPLING_CFG, 'year_bin_edges', None)
        if yb is not None:
            YEAR_BIN_EDGES = [tuple(row) for row in OmegaConf.to_container(yb, resolve=True)]
        os_v = getattr(NEG_SAMPLING_CFG, 'on_shortfall', None)
        if os_v is not None:
            ON_SHORTFALL = str(os_v)
        rac = getattr(NEG_SAMPLING_CFG, 'regime_aware_coverage', None)
        if rac is not None:
            REGIME_AWARE_COVERAGE = bool(rac)

    from src.datasets.dataset_segment_pairs_v2 import (
        split_dataset_v2,
        generate_all_cv_folds_v2,
        generate_all_cluster_disjoint_cv_folds_v2,
        save_split_output_v2,
        compute_metadata_coverage,
        _validate_v2_config,
    )

    _validate_v2_config(config)

    # Narrow df to schema_pair rows. v2 hard-codes pair_mode='schema_ordered',
    # so only (func_left, func_right) rows can appear in pair generation,
    # cooccur queries, axis flags, or exposure tables. Restricting df once here
    # avoids a ~C(|selected_functions|, 2)x bloat in build_cooccurrence_set
    # and seq_to_isolates, and keeps the QC artifact (metadata_coverage.json)
    # describing the same population that drives pair generation.
    df = df[df['function'].isin(schema_pair)].reset_index(drop=True)

    # Pre-attach cds_dna_hash when pair_key_alphabet='nt_cds'. v2's
    # create_positive_pairs_v2 + build_cooccurrence_set expect the column
    # on df; pair_key is built on cds_dna_hash_{a,b} instead of prot_hash_{a,b}.
    # Resolves cds_final path from CDS_FINAL_PATH (set above when
    # cluster_alphabet='nt_cds') or from the input file's parent dir.
    if PAIR_KEY_ALPHABET == 'nt_cds':
        from src.datasets._pair_helpers import attach_cds_dna_hash_to_prot_df
        cds_path = CDS_FINAL_PATH
        if cds_path is None:
            cds_path = str(input_file.parent / 'cds_dna_final.parquet')
        print(f'\nv2: pair_key_alphabet=nt_cds — attaching cds_dna_hash to df '
              f'from {cds_path}')
        df = attach_cds_dna_hash_to_prot_df(df, cds_path)

    # Per-run artifact: written once after df is finalized and before split /
    # CV branching. The launcher (run_cv_lambda.py) needs no changes -- the
    # file simply lands in the parent of fold_*/ in CV mode.
    # TODO. Check the issue in compute_metadata_coverage() function docstring
    print('\nv2: computing metadata_coverage.json (per-run artifact)...')
    coverage = compute_metadata_coverage(df, axes=AXES_FOR_FLAGS)
    with open(output_dir / 'metadata_coverage.json', 'w') as f:
        json.dump(coverage, f, indent=2, default=str)
    print(f"Saved metadata coverage to: {output_dir / 'metadata_coverage.json'}")

    if N_FOLDS is not None and N_FOLDS > 1:
        # v2 CV mode
        print(f'\nv2 CV mode: generating {N_FOLDS} folds (seed={RANDOM_SEED}).')
        cv_info = {
            'n_folds': N_FOLDS,
            'master_seed': RANDOM_SEED,
            'fold_seeds': {i: RANDOM_SEED + i for i in range(N_FOLDS)},
            'bundle': config_bundle,
            'fold_dirs': [f'fold_{i}' for i in range(N_FOLDS)],
            'pair_builder_version': 'v2',
        }
        with open(output_dir / 'cv_info.json', 'w') as f:
            json.dump(cv_info, f, indent=2)
        print(f"Saved CV metadata to: {output_dir / 'cv_info.json'}")

        # Dispatch the CV generator by split mode:
        # - cluster_disjoint + single_slot: new k-fold generator from Phase 3
        #   of the k-fold variance plan (GroupKFold-on-cluster_id +
        #   collect-all D3 + D4 menu).
        # - all other modes (random / seq_disjoint*): existing
        #   isolate-KFold-on-random generator. (*seq_disjoint with N_FOLDS>1
        #   is still rejected by _validate_v2_config at the moment.)
        use_cluster_disjoint_kfold = (
            SPLIT_STRATEGY_MODE == 'cluster_disjoint' and SINGLE_SLOT is not None
        )
        if use_cluster_disjoint_kfold:
            print(f'v2 CV mode: routing via cluster_disjoint k-fold '
                  f"(single_slot={SINGLE_SLOT!r}, alphabet={CLUSTER_ALPHABET!r}, "
                  f'threshold={CLUSTER_ID_THRESHOLD}, '
                  f'max_acceptable_drift_pp={MAX_ACCEPTABLE_DRIFT_PP}, '
                  f'min_test_frac={MIN_TEST_FRAC})')
            cv_gen = generate_all_cluster_disjoint_cv_folds_v2(
                df=df,
                n_folds=N_FOLDS,
                seed=RANDOM_SEED,
                neg_to_pos_ratio=NEG_TO_POS_RATIO,
                val_ratio=VAL_RATIO,
                schema_pair=schema_pair,
                single_slot=SINGLE_SLOT,
                cluster_id_path=CLUSTER_ID_PATH,
                cluster_id_threshold=CLUSTER_ID_THRESHOLD,
                cluster_alphabet=CLUSTER_ALPHABET,
                cds_final_path=CDS_FINAL_PATH,
                max_acceptable_drift_pp=MAX_ACCEPTABLE_DRIFT_PP,
                min_test_frac=MIN_TEST_FRAC,
                max_attempts_per_seq=MAX_ATTEMPTS_PER_SEQ,
                axes_for_flags=AXES_FOR_FLAGS,
                axis_quotas=AXIS_QUOTAS,
                sampling_axes=SAMPLING_AXES,
                year_match=YEAR_MATCH,
                year_bin_edges=YEAR_BIN_EDGES,
                on_shortfall=ON_SHORTFALL,
                regime_aware_coverage=REGIME_AWARE_COVERAGE,
                pair_key_alphabet=PAIR_KEY_ALPHABET,
            )
        else:
            cv_gen = generate_all_cv_folds_v2(
                df=df,
                n_folds=N_FOLDS,
                seed=RANDOM_SEED,
                neg_to_pos_ratio=NEG_TO_POS_RATIO,
                val_ratio=VAL_RATIO,
                schema_pair=schema_pair,
                max_attempts_per_seq=MAX_ATTEMPTS_PER_SEQ,
                axes_for_flags=AXES_FOR_FLAGS,
                axis_quotas=AXIS_QUOTAS,
                sampling_axes=SAMPLING_AXES,
                year_match=YEAR_MATCH,
                year_bin_edges=YEAR_BIN_EDGES,
                on_shortfall=ON_SHORTFALL,
                regime_aware_coverage=REGIME_AWARE_COVERAGE,
                pair_key_alphabet=PAIR_KEY_ALPHABET,
            )

        # Stream each fold to disk as it's produced (reduces peak memory; lets
        # progress show up on disk while later folds are still computing).
        for fold_data in cv_gen:
            fold_dir = output_dir / f"fold_{fold_data['fold_id']}"
            print(f"\nSaving fold {fold_data['fold_id'] + 1}/{N_FOLDS} to: {fold_dir}")
            save_split_output_v2(
                output_dir=fold_dir,
                train_pairs=fold_data['train_pairs'],
                val_pairs=fold_data['val_pairs'],
                test_pairs=fold_data['test_pairs'],
                duplicate_stats=fold_data['duplicate_stats'],
                exposure_tables=fold_data['exposure_tables'],
                df=df,
                config_bundle=config_bundle,
                schema_pair=schema_pair,
                filters_applied=filters_applied,
                axes_for_flags=AXES_FOR_FLAGS,
                generate_visualizations=GENERATE_VISUALIZATIONS,
                skip_esm_pca_plots=SKIP_ESM_PCA_PLOTS,
                skip_kmer_pca_plots=SKIP_KMER_PCA_PLOTS,
                holdout_cfg=None,  # validator forbids holdout + CV combo
            )
    else:
        # v2 single-split mode. (Temporal/legacy year_train was removed
        # 2026-05-11; see docs/plans/2026-05-11_metadata_holdout_plan.md.)
        # If dataset.metadata_holdout is set, compute the three isolate-id
        # lists from the per-slot filters and pass them through the existing
        # *_isolates_override hook on split_dataset_v2.
        HOLDOUT_CFG = getattr(config.dataset, 'metadata_holdout', None)
        holdout_train_ids = holdout_val_ids = holdout_test_ids = None
        holdout_dropped_df = None
        holdout_dict = None
        if HOLDOUT_CFG is not None:
            from omegaconf import OmegaConf
            from src.datasets._pair_helpers import compute_metadata_holdout_isolates
            holdout_dict = OmegaConf.to_container(HOLDOUT_CFG, resolve=True)
            holdout_train_ids, holdout_val_ids, holdout_test_ids, holdout_dropped_df = \
                compute_metadata_holdout_isolates(
                    df, holdout_dict, seed=RANDOM_SEED, val_ratio=VAL_RATIO,
                )

        print('\nv2: single-split mode: generate train/val/test...')
        _t = time.time()
        train_pairs, val_pairs, test_pairs, duplicate_stats, exposure_tables = split_dataset_v2(
            df=df,
            schema_pair=schema_pair,
            neg_to_pos_ratio=NEG_TO_POS_RATIO,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            seed=RANDOM_SEED,
            max_attempts_per_seq=MAX_ATTEMPTS_PER_SEQ,
            axes_for_flags=AXES_FOR_FLAGS,
            axis_quotas=AXIS_QUOTAS,
            sampling_axes=SAMPLING_AXES,
            year_match=YEAR_MATCH,
            year_bin_edges=YEAR_BIN_EDGES,
            on_shortfall=ON_SHORTFALL,
            regime_aware_coverage=REGIME_AWARE_COVERAGE,
            split_strategy_mode=SPLIT_STRATEGY_MODE,
            split_strategy_hash_key=SPLIT_STRATEGY_HASH_KEY,
            cluster_id_path=CLUSTER_ID_PATH,
            cluster_id_threshold=CLUSTER_ID_THRESHOLD,
            cluster_alphabet=CLUSTER_ALPHABET,
            cds_final_path=CDS_FINAL_PATH,
            single_slot=SINGLE_SLOT,
            train_isolates_override=holdout_train_ids,
            val_isolates_override=holdout_val_ids,
            test_isolates_override=holdout_test_ids,
            pair_key_alphabet=PAIR_KEY_ALPHABET,
            drop_budget=DROP_BUDGET,
        )
        print(f"stage3 v2: split_dataset_v2 (done in {time.time()-_t:.2f}s)", flush=True)

        # Persist metadata_holdout artifacts next to dataset_stats.json.
        if holdout_dropped_df is not None:
            holdout_dropped_path = output_dir / 'metadata_holdout_dropped.csv'
            holdout_dropped_df.to_csv(holdout_dropped_path, index=False)
            print(f"Saved metadata_holdout dropped-isolates manifest "
                  f"({len(holdout_dropped_df):,} isolate(s)) to: {holdout_dropped_path}")
            # Stash a summary to be merged into dataset_stats.json by the saver.
            duplicate_stats['metadata_holdout'] = {
                'config': holdout_dict,
                'n_train_isolates': len(holdout_train_ids),
                'n_val_isolates': len(holdout_val_ids),
                'n_test_isolates': len(holdout_test_ids),
                'n_dropped': int(len(holdout_dropped_df)),
                'val_source': (
                    'explicit_filter' if (holdout_dict.get('val') is not None)
                    else 'carved_from_train'
                ),
            }

        print(f'\nSave datasets: {output_dir}')
        # breakpoint()
        save_split_output_v2(
            output_dir=output_dir,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            test_pairs=test_pairs,
            duplicate_stats=duplicate_stats,
            exposure_tables=exposure_tables,
            df=df,
            config_bundle=config_bundle,
            schema_pair=schema_pair,
            filters_applied=filters_applied,
            axes_for_flags=AXES_FOR_FLAGS,
            generate_visualizations=GENERATE_VISUALIZATIONS,
            skip_esm_pca_plots=SKIP_ESM_PCA_PLOTS,
            skip_kmer_pca_plots=SKIP_KMER_PCA_PLOTS,
            holdout_cfg=holdout_dict,
        )

else:
    raise ValueError(
        f"pair_builder_version={PAIR_BUILDER_VERSION!r} is not supported. The v1 builder "
        "was retired 2026-06-03; only 'v2' remains. See "
        "docs/plans/2026-06-03_deprecate_v1_builder_plan.md."
    )

print(f'\nDone. Finished {Path(__file__).name}.')
total_timer.stop_timer()
total_timer.display_timer()
total_timer.save_timer(output_dir)
