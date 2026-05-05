# Logistic Regression Baseline Integration — Plan

**Status: IN PROGRESS**

Add a logistic regression baseline alongside the MLP classifier in Stage 4. Land it
behind a clean module boundary so XGBoost (and other baselines) can drop in later
without touching the MLP path.

## Decided up front

- **Output layout: flat.** Each baseline run writes to its own sibling directory
  next to MLP runs:
  `models/flu/{version}/runs/baseline_<name>_<bundle>_<TS>/`.
- **Default LR `feature_scaling: standard`.** LR coefficients depend on feature
  scale. Bundles that want unscaled LR set `feature_scaling: none` explicitly.
- **Default `bundle.baselines: null`.** Bundles run MLP only unless they list
  baselines (e.g., `baselines: [logistic]`).
- **Interaction: concat only.** Per user direction, baselines start with simple
  concatenation. Other interactions (`unit_diff`, `diff`, `prod`) are currently
  ESM-only experiments; we don't replicate them on the LR path.
- **`slot_transform`: `none` only**, with a numpy LayerNorm placeholder + NOTE
  comment in `_pair_features.py` for future use.
- **Scope of this plan: steps 1–4.** XGBoost (step 5) is deferred.

## Non-goals

- No change to MLP behavior. Step 1 is a pure refactor; the MLP still produces
  byte-identical artifacts after it.
- No baseline support for ESM-2 features. LR runs on k-mer only. (ESM-2 path
  raises `NotImplementedError`, mirroring the current StandardScaler behavior.)
- No CV support for baselines. Single-split only.
- `stage4_train.sh` is unchanged — still runs MLP only. Baselines live in new
  shell wrappers.

---

## Step 1 — Extract `_pair_metrics.py` (pure refactor)

**Goal:** lift the prediction/metric/writer helpers out of
`src/models/train_pair_classifier.py` so both MLP and baseline scripts share them.
No behavior change.

**Move out** (verbatim, no signature changes):
- `find_optimal_threshold_pr(y_true, y_probs, metric)` — currently L64
- `swap_pairs_df_columns(pairs_df)` — currently L999
- A new helper `compute_pair_metrics(y_true, y_probs, threshold)` that wraps the
  metric-block currently inline in `evaluate_on_split` (L968–L995), returning a
  dict `{f1, f1_macro, precision, recall, auc, mean_loss?}` plus a results
  DataFrame containing `pred_label`, `pred_prob`, `pred_logit?`. The
  numpy/sklearn parts are model-agnostic; the loop over `data_loader` is not.

**Keep in `train_pair_classifier.py`:**
- `evaluate_on_split` itself stays (it owns the torch loop) but its metric
  computation calls the new helper. The function shrinks; semantics are
  preserved.

**Files touched:**
- New: `src/models/_pair_metrics.py`
- Edit: `src/models/train_pair_classifier.py` (drop helpers, import them, rewire
  `evaluate_on_split` to call `compute_pair_metrics`).

**Verification:**
- Re-run `flu_ha_na_kmer_scale_standard.yaml` (or `flu_ha_na`) end-to-end and
  diff the artifacts against the pre-refactor run:
  - `test_predicted.csv` — exact equality (same threshold, same probs, same
    labels).
  - `training_info.json` — equality except `timestamp`.
  - F1/AUC numbers in stdout — exact match.
- The diff is the contract for "no behavior change."

**Ship:** one commit, `refactor(models): extract pair metric helpers into _pair_metrics`.

---

## Step 2 — Create `_pair_features.py`

**Goal:** a reusable feature loader that returns dense numpy arrays
`(X_train, y_train), (X_val, y_val), (X_test, y_test)` plus an optional fitted
`StandardScaler`. Used by every baseline. The MLP keeps its `Dataset`-based path
untouched.

**Public surface (one entry point):**

```python
def load_pair_features_for_baselines(
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    *,
    feature_source: str,        # 'kmer' (esm2 raises NotImplementedError)
    feature_scaling: str,       # 'none' | 'standard'
    kmer_dir: Path,
    kmer_k: int,
    output_dir: Path,           # for saving feature_scaler.joblib
) -> tuple[
    tuple[np.ndarray, np.ndarray],   # train: (X, y)
    tuple[np.ndarray, np.ndarray],   # val
    tuple[np.ndarray, np.ndarray],   # test
    Optional[StandardScaler],        # fitted scaler (or None)
]:
```

**Internals:**
1. Build per-side dense arrays `(emb_a, emb_b)` per split. For k-mer: reuse
   `KmerPairDataset`-style row lookup but materialize the pair matrix instead of
   a torch dataset. For ESM-2: `raise NotImplementedError`.
2. Concatenate per-row: `X = np.hstack([emb_a, emb_b])`. No interaction
   options surfaced — baselines start with concat-only by design.
3. Add a private `_apply_slot_norm(emb)` placeholder function with a NOTE
   comment explaining it's a numpy LayerNorm equivalent reserved for future
   ESM-on-LR experiments. Not called in this round.
   ```python
   def _apply_slot_norm(emb: np.ndarray) -> np.ndarray:
       # NOTE: numpy LayerNorm placeholder. Mirrors nn.LayerNorm(D)(emb) along
       # the feature axis: zero-mean, unit-variance per row. Reserved for
       # future ESM-on-LR baselines where slot-level normalization matters
       # (cf. K-mer features don't benefit from it). Not currently called.
       eps = 1e-5
       mu = emb.mean(axis=-1, keepdims=True)
       sd = emb.std(axis=-1, keepdims=True)
       return (emb - mu) / (sd + eps)
   ```
4. Fit `StandardScaler` on `X_train` (only) when `feature_scaling == 'standard'`,
   transform all splits, dump to `output_dir / 'feature_scaler.joblib'`. When
   `none`, no scaler is fit and no file is written.

**Files touched:**
- New: `src/models/_pair_features.py`

**Verification:**
- Sanity: for k-mer + `feature_scaling=none`, the resulting `X_train` shape is
  `(N_train, 2 * 4096)`. With `feature_scaling=standard`, every column has
  ~0 mean and ~1 std on the train split.

**Ship:** one commit, `feat(models): shared pair feature loader for baselines`.

---

## Step 3 — `baselines/logistic.py` + `train_pair_baselines.py` + `stage4_baselines.sh`

**Goal:** end-to-end LR baseline runnable by itself.

### `src/models/baselines/__init__.py` + `baselines/logistic.py`

A baseline = a small module exposing two functions:

```python
def get_estimator(config) -> sklearn-like-estimator:
    # config is config.baselines.logistic (or whatever the bundle nests)
    # Returns a configured but unfitted LogisticRegression.

def name() -> str:
    return "logistic"
```

`logistic.py` reads `config.baselines.logistic.{C, penalty, max_iter,
solver, class_weight, n_jobs}` with sensible defaults
(`C=1.0, penalty='l2', solver='lbfgs', max_iter=200, class_weight=None,
n_jobs=-1`). Plus a `random_state` it pulls from the resolved process seed.

### `src/models/train_pair_baselines.py` (entry point)

CLI mirrors `train_pair_classifier.py`:
```
--config_bundle <name>
--baseline <name>           # required; one of {logistic}
--dataset_dir <path>
--output_dir <path>         # optional, else build from path_utils
--run_output_subdir <id>    # optional
--cuda_name <ignored>       # accepted but ignored, kept for shell-script symmetry
```

Flow:
1. Resolve config, paths, seeds — same as MLP. Log feature_source,
   interaction, slot_transform, feature_scaling, baseline name.
2. Read `train_pairs.csv`, `val_pairs.csv`, `test_pairs.csv` from `--dataset_dir`.
3. Call `load_pair_features_for_baselines(...)`. Write `feature_scaler.joblib`
   if applicable (already done by the loader).
4. Resolve baseline module via a small registry:
   ```python
   BASELINE_REGISTRY = {"logistic": "src.models.baselines.logistic"}
   ```
5. Build estimator → `estimator.fit(X_train, y_train)`. Save with
   `joblib.dump(estimator, output_dir / 'best_model.joblib')`.
6. Predict probabilities on train/val/test → reuse `find_optimal_threshold_pr`
   on val to pick threshold (same logic as MLP) → reuse
   `compute_pair_metrics` → write `train_predicted.csv`, `val_predicted.csv`,
   `test_predicted.csv` with the same columns as MLP outputs
   (`pred_label, pred_prob`; `pred_logit` is the LR `decision_function` output
   for parity).
7. Write `training_info.json` with the same shape as MLP's, plus a
   `baseline: 'logistic'` field. Fields that don't apply (e.g.,
   `epochs`, `patience`, `optimizer`) are recorded as `null` rather than
   omitted, so downstream tooling that reads the file doesn't have to special-case
   missing keys.
8. Save resolved config via `save_config(...)` (same as MLP).

### `scripts/stage4_baselines.sh`

Mirrors `stage4_train.sh` line-for-line but:
- Required argument: `--baseline <name>` in addition to `--dataset_dir`.
- `RUN_ID="baseline_${BASELINE}_${CONFIG_BUNDLE}_${TIMESTAMP}"`.
- Calls `python src/models/train_pair_baselines.py ...`.
- Postprocessing: skip `analyze_stage4_train.py` for now (it's
  MLP-shaped). Keep the log-symlink + log-copy steps. We'll teach the analyzer
  to handle baselines in a follow-up; for the first cut, baseline runs ship
  with their `metrics_summary.json` + predictions CSVs only.

**Files touched:**
- New: `src/models/baselines/__init__.py`
- New: `src/models/baselines/logistic.py`
- New: `src/models/train_pair_baselines.py`
- New: `scripts/stage4_baselines.sh`
- New: `conf/baselines/logistic.yaml` — defaults for the LR estimator.
  Bundles can override.

### `conf/training/base.yaml`

Add nothing here; baseline params live under `config.baselines.<name>`, a
separate config group. Avoids polluting the MLP-specific training group.

### `conf/<root_config>` defaults

Hydra wiring: the baseline config group is opt-in. Bundles that want LR add:
```yaml
defaults:
  - flu_ha_na
  - /baselines: logistic   # makes config.baselines.logistic available
  - _self_
```

**Verification:**
- Run `./scripts/stage4_baselines.sh flu_ha_na --baseline logistic
  --dataset_dir <existing-stage3-dir>`. Confirm:
  - `models/flu/v1/runs/baseline_logistic_flu_ha_na_<TS>/` is created.
  - It contains `best_model.joblib`, `feature_scaler.joblib` (since default is
    `standard`), `train_predicted.csv`, `val_predicted.csv`,
    `test_predicted.csv`, `metrics_summary.json`, `training_info.json`,
    `config.yaml`, `stage4_baseline.log`.
  - The script runs to completion and writes finite F1/AUC numbers. The actual
    LR performance on concat'd k-mer features is an open question — that's
    what running this is for. We don't pre-claim a number.
- Compare `test_predicted.csv` row count and `assembly_id` columns against the
  MLP run on the same dataset directory. Should be identical pair sets.

**Ship:** one commit, `feat(baselines): logistic regression baseline + stage4_baselines wrapper`.

---

## Step 4 — `stage4_full.sh` + bundle-level `baselines` field

**Goal:** one command runs MLP + every requested baseline against the same
dataset, sequentially. Bundles declare which baselines they want.

### Bundle field

Add an optional top-level field `baselines` in any bundle:
```yaml
baselines:
  - logistic
```
Default: absent / null → MLP only. Read by `stage4_full.sh` (not by
`train_pair_classifier.py` itself — keeping the trainer ignorant of baselines).

### `scripts/stage4_full.sh`

```
./scripts/stage4_full.sh <bundle> --dataset_dir <dir> [other-flags-passed-through]
```

Flow:
1. Run `./scripts/stage4_train.sh <bundle> --dataset_dir <dir> ...`. (MLP run.)
2. Read `bundle.baselines` from the resolved Hydra config. We can shell out to
   a tiny helper:
   ```bash
   BASELINES=$(python -c "from src.utils.config_hydra import get_virus_config_hydra; \
     c=get_virus_config_hydra('$CONFIG_BUNDLE'); \
     print(' '.join(getattr(c, 'baselines', None) or []))")
   ```
   (Real implementation will be tighter; this is the shape.)
3. For each `b` in `$BASELINES`: run
   `./scripts/stage4_baselines.sh <bundle> --baseline $b --dataset_dir <dir>`.
4. Each run writes its own flat directory under `runs/`. No nesting.

**Files touched:**
- New: `scripts/stage4_full.sh`
- Edit: `conf/bundles/README.md` — document the `baselines` field and the
  full-vs-baseline-only run scripts.
- Optional edit: `conf/bundles/flu_ha_na.yaml` — add `baselines: [logistic]`
  as a working example for the first end-to-end smoke test.

**Verification:**
- Add `baselines: [logistic]` to `flu_ha_na.yaml`.
- Run `./scripts/stage4_full.sh flu_ha_na --dataset_dir <existing-stage3-dir>`.
- Confirm two flat run dirs are created:
  `training_flu_ha_na_<TS>/` and `baseline_logistic_flu_ha_na_<TS>/`.
- Both should have their own log files in `logs/training/`.

**Ship:** one commit, `feat(scripts): stage4_full runs MLP + bundle-listed baselines`.

---

## Deferred

Keep things simple; revisit once we have initial LR results on `flu_ha_na`.

- Analyzer / aggregator support for baselines.
- CV-aware baselines.
- ESM-2 baselines.
- Other interactions (`unit_diff`, `diff`, `prod`) and `slot_norm` for
  baselines.
- XGBoost (step 5).

## Order of operations and rollback

Steps land as four commits, each independently revertable:
1. Refactor metrics. Verified by artifact diff.
2. Add feature loader. Internal-only; no callers yet.
3. Add LR baseline + standalone wrapper. Verified by smoke run.
4. Add full wrapper + bundle field. Verified by smoke run with both runners.

If step 1's artifact diff isn't byte-identical, stop and investigate before
proceeding. Step 2 has no behavioral effect on MLP runs. Steps 3–4 add new
entry points only.
