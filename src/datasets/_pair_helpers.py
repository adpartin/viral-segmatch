"""Shared helpers for the segment-pair dataset builders (v1 and v2).

This module exists because `dataset_segment_pairs.py` (v1) has script-level
code that runs at import time (config loading, file I/O, dispatch). v2 cannot
`from .dataset_segment_pairs import canonical_pair_key` without triggering the
entire v1 pipeline. Pulling the pure helpers out into this module lets both
v1 and v2 import them safely.

Functions exposed here are pure (no module-level side effects). v1-specific
pair generation, splitting, and saving stay in `dataset_segment_pairs.py`;
v2-specific equivalents stay in `dataset_segment_pairs_v2.py`.

Refactor history: extracted from v1 as part of the v2 implementation; v1's
behavior is unchanged. See `docs/plans/done/design_dataset_gen_v2.md` §3.
"""

import hashlib
import json
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Project root is two levels up (src/datasets/_pair_helpers.py -> src/datasets -> src -> root)
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

from src.utils import schema
from src.utils.path_utils import load_dataframe


def canonical_pair_key(hash_a: str, hash_b: str) -> str:
    """Create a canonical pair key from two hashes.

    Ensures consistent ordering so (a, b) and (b, a) produce the same key.

    The hashes can be from any alphabet — `prot_hash` (protein) for the
    aa pair_key family, `cds_dna_hash` for the nt_cds family. Callers
    decide the alphabet; this function just joins the two inputs in
    sorted order. Argument names preserve the historical `prot_hash`
    naming because the bulk of callers still use protein hashes; under
    pair_key_alphabet='nt_cds' (Phase 2 of clustering cleanup plan)
    the inputs are `cds_dna_hash` values instead. See pair_key plan
    docs/plans/2026-06-02_pair_key_alphabet_plan.md for the bias
    direction and adoption rationale.
    """
    return "__".join(sorted([hash_a, hash_b]))


def _validate_schema_pair(schema_pair, fn_name: str) -> Tuple[str, str]:
    """Validate schema_pair and return (func_left, func_right).

    Raises ValueError if schema_pair is None, not a 2-tuple, or contains two
    equal functions. fn_name is embedded in error messages so the user can
    identify which caller rejected their input.
    """
    if schema_pair is None or len(schema_pair) != 2:
        raise ValueError(f"{fn_name} requires schema_pair=(func_left, func_right)")
    func_left, func_right = schema_pair
    if func_left == func_right:
        raise ValueError(
            f"{fn_name}: schema_pair must contain two different functions "
            f"(func_left != func_right), got {schema_pair!r}"
        )
    return func_left, func_right


def attach_dna_to_prot_df(prot_df: pd.DataFrame, protein_input_path: Path) -> pd.DataFrame:
    """Attach nucleotide sequence (`ctg_dna_seq`) and md5 hash (`ctg_dna_hash`) to each
    protein row via a join with `ctg_dna_final.*` in the same processed dir.

    Rationale: the existing pipeline dedups/checks leakage using `prot_hash` over
    protein sequences -- the correct check for ESM-2 features, but not for nucleotide
    based k-mer features (which are derived from DNA). Carrying DNA into the pair
    CSVs makes post-hoc nucleotide-level leakage audits possible without
    re-running Stage 3.

    Join key: (assembly_id, genbank_ctg_id). Verified on the full Flu July 2025
    dataset: genome side is unique on this key, every protein row matches
    exactly one genome row, and canonical_segment agrees on both sides. Multiple
    proteins can share one DNA contig (M1/M2, NS1/NEP, PA/PA-X) -- that's
    semantically correct; those proteins do originate from the same nucleotide
    sequence, and DNA-derived features (k-mers) would be identical for them.

    Fails loudly on: missing join columns, duplicate genome-side keys, row
    count drift post-merge, or any unmatched protein row.
    """
    required = {"assembly_id", "genbank_ctg_id"}
    missing = required - set(prot_df.columns)
    if missing:
        raise ValueError(
            f"prot_df is missing columns required for DNA join: {sorted(missing)}"
        )

    genome_path = protein_input_path.parent / "ctg_dna_final"
    print(f"\nAttach DNA sequences from ctg_dna_final (sibling of {protein_input_path.name}).")
    genome_df = load_dataframe(genome_path)

    genome_cols = ["assembly_id", "genbank_ctg_id", "ctg_dna_seq"]
    miss_g = set(genome_cols) - set(genome_df.columns)
    if miss_g:
        raise ValueError(f"ctg_dna_final is missing required columns: {sorted(miss_g)}")
    genome_small = genome_df[genome_cols].copy()

    dup = genome_small.duplicated(["assembly_id", "genbank_ctg_id"]).sum()
    if dup:
        raise ValueError(
            f"ctg_dna_final has {dup} duplicate (assembly_id, genbank_ctg_id) rows. "
            "The DNA join would fan out protein rows; refusing to proceed."
        )

    before = len(prot_df)
    merged = prot_df.merge(
        genome_small,
        on=["assembly_id", "genbank_ctg_id"],
        how="left",
        validate="many_to_one", # each protein row matches at most one genome row
    )
    # If len(merged) == before, then each protein row matched exactly one genome row;
    # if not, something went wrong (e.g., fan-out from duplicate genome keys)
    if len(merged) != before:
        raise RuntimeError(
            f"Row count changed after DNA merge: {before} -> {len(merged)}. "
            "Merge fan-out suspected."
        )

    # Check for unmatched protein rows (missing DNA). This is a sanity check that the
    # ctg_dna_final and protein_final came from the same preprocess_flu.py run; they
    # should match perfectly on (assembly_id, genbank_ctg_id).
    missing_dna = merged["ctg_dna_seq"].isna().sum()
    if missing_dna:
        raise RuntimeError(
            f"{missing_dna:,} protein rows have no matching ctg_dna_seq. "
            "Check that protein_final and ctg_dna_final came from the same "
            "preprocess_flu.py run."
        )

    merged["ctg_dna_hash"] = merged["ctg_dna_seq"].apply(
        lambda s: hashlib.md5(str(s).encode()).hexdigest()
    )

    n_unique_dna = merged["ctg_dna_hash"].nunique()
    print(f"  genome_df: Total rows: {len(genome_small):,};  unique ctg_dna_seq: {genome_small['ctg_dna_seq'].nunique():,}")
    print(f"  protein rows merged (with genome data):  {before:,} (100.0% matched)")
    print(f"  unique ctg_dna_hash:      {n_unique_dna:,} "
          f"({n_unique_dna / before * 100:.1f}% of protein rows -- many proteins share a contig)")
    return merged


def attach_cds_dna_hash_to_prot_df(
    prot_df: pd.DataFrame,
    cds_final_path: Path,
    ) -> pd.DataFrame:
    """Add a `cds_dna_hash` column to a protein-level DataFrame.

    Required by `pair_key_alphabet='nt_cds'` (Phase 2 of clustering
    cleanup plan) — `create_positive_pairs_v2` constructs the pair_key
    on `cds_dna_hash_{a,b}` rather than `prot_hash_{a,b}`, and
    `build_cooccurrence_set` needs to key cooccur pairs on
    `cds_dna_hash` to match. Orchestrator (Stage 3 entry point) calls
    this once on the full prot_df before passing to `split_dataset_v2`
    or the k-fold variants.

    Lookup key: `(assembly_id, function)`. cds_dna_final.parquet is unique
    on this key for the 8 Flu A majors (Stage 1.5 invariant; see
    `src/preprocess/extract_cds_dna.py`).

    Args:
        prot_df: must contain `assembly_id` + `function` columns
            (standard prot_df shape from Stage 1 + DNA join).
        cds_final_path: path to `cds_dna_final.parquet`. Must contain
            `assembly_id`, `function`, `cds_dna_hash`.

    Returns a new DataFrame with `cds_dna_hash` added. Raises on
    missing prot_df rows, fan-out, or unmatched (assembly_id, function)
    pairs (Stage 3 strictness: no silent drops).
    """
    cds_final_path = Path(cds_final_path)
    if not cds_final_path.exists():
        raise FileNotFoundError(
            f"attach_cds_dna_hash_to_prot_df: cds_final not found at "
            f"{cds_final_path}. Build via "
            f"`python src/preprocess/extract_cds_dna.py --config_bundle <bundle>`."
        )
    for col in ('assembly_id', 'function'):
        if col not in prot_df.columns:
            raise ValueError(
                f"attach_cds_dna_hash_to_prot_df: prot_df missing '{col}' column"
            )

    cds_df = pd.read_parquet(
        cds_final_path,
        columns=['assembly_id', 'function', 'cds_dna_hash'],
    )
    cds_df['assembly_id'] = cds_df['assembly_id'].astype(str)

    if cds_df.duplicated(['assembly_id', 'function']).any():
        n_dup = int(cds_df.duplicated(['assembly_id', 'function']).sum())
        raise AssertionError(
            f"attach_cds_dna_hash_to_prot_df: cds_final has {n_dup} duplicate "
            f"(assembly_id, function) rows — would fan out the join."
        )

    out = prot_df.copy()
    out['assembly_id'] = out['assembly_id'].astype(str)
    before = len(out)
    out = out.merge(cds_df, on=['assembly_id', 'function'], how='left')
    if len(out) != before:
        raise RuntimeError(
            f"attach_cds_dna_hash_to_prot_df: row count changed "
            f"({before} -> {len(out)}); fan-out from non-unique join key."
        )

    missing = out['cds_dna_hash'].isna().sum()
    if missing:
        # Only the 8 selected_functions have CDS; rows for other functions
        # (M2, NEP, PB1-F2, etc.) won't match. For the standard schema_pair
        # flow these are filtered upstream, but if any unmatched rows remain
        # we DROP them with a warning (a NaN cds_dna_hash would break
        # downstream hashing).
        missing_funcs = sorted(out.loc[out['cds_dna_hash'].isna(), 'function'].unique())
        print(f"WARNING: attach_cds_dna_hash_to_prot_df: {missing:,} rows have "
              f"no cds_dna_hash (functions: {missing_funcs}); dropping. "
              f"These functions are not in selected_functions / not covered "
              f"by cds_final.")
        out = out.dropna(subset=['cds_dna_hash']).reset_index(drop=True)

    print(f"attach_cds_dna_hash_to_prot_df: attached cds_dna_hash to "
          f"{len(out):,} prot_df rows (unique cds_dna_hash="
          f"{out['cds_dna_hash'].nunique():,}).")
    return out


def attach_cds_dna_hash_to_pos_df(
    pos_df: pd.DataFrame,
    cds_final_path: Path,
    schema_pair: Tuple[str, str],
    ) -> pd.DataFrame:
    """Add `cds_dna_hash_a` and `cds_dna_hash_b` columns to pos_df.

    Required by the nt-alphabet branch of cluster_disjoint routing
    (Experiment B-nt, see `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md`).
    The values come from `cds_dna_final.parquet` (built by
    `src/preprocess/extract_cds_dna.py`), keyed on
    `(assembly_id, function)` — unique by construction for the 8 majors.

    Args:
        pos_df: must contain `assembly_id` plus the prot_hash_a/b /
            ctg_dna_hash_a/b columns from `create_positive_pairs_v2`.
        cds_final_path: path to `cds_dna_final.parquet`. Must contain columns
            `assembly_id`, `function`, `cds_dna_hash`.
        schema_pair: `(func_left, func_right)` — slot A and slot B.

    Returns a new DataFrame with `cds_dna_hash_a` and `cds_dna_hash_b`
    attached. Raises if any pos_df row fails the lookup (Stage 3 strictness:
    no silent drops).
    """
    cds_final_path = Path(cds_final_path)
    if not cds_final_path.exists():
        raise FileNotFoundError(
            f"attach_cds_dna_hash_to_pos_df: cds_final not found at "
            f"{cds_final_path}. Build it with "
            f"`python src/preprocess/extract_cds_dna.py --config_bundle <virus_bundle>`."
        )
    func_left, func_right = _validate_schema_pair(
        schema_pair, "attach_cds_dna_hash_to_pos_df"
    )
    # pos_df from v2's create_positive_pairs_v2 carries assembly_id_a and
    # assembly_id_b (one per slot). Under v2's schema_ordered + one-pair-per-
    # isolate invariant these are equal, so either is fine — use them both
    # explicitly so the join works even on a hypothetical future v1-shape pos_df.
    for col in ('assembly_id_a', 'assembly_id_b'):
        if col not in pos_df.columns:
            raise ValueError(f"pos_df must contain '{col}'")

    cds_df = pd.read_parquet(
        cds_final_path,
        columns=['assembly_id', 'function', 'cds_dna_hash'],
    )
    cds_df['assembly_id'] = cds_df['assembly_id'].astype(str)

    def _lookup_for(func: str, suffix: str) -> pd.DataFrame:
        sub = cds_df[cds_df['function'] == func][['assembly_id', 'cds_dna_hash']]
        if sub['assembly_id'].duplicated().any():
            raise AssertionError(
                f"cds_final has duplicate assembly_id rows for function={func!r}"
            )
        return sub.rename(columns={
            'assembly_id': f'assembly_id_{suffix}',
            'cds_dna_hash': f'cds_dna_hash_{suffix}',
        })

    out = pos_df.copy()
    out['assembly_id_a'] = out['assembly_id_a'].astype(str)
    out['assembly_id_b'] = out['assembly_id_b'].astype(str)
    before = len(out)
    out = out.merge(_lookup_for(func_left, 'a'), on='assembly_id_a', how='left')
    out = out.merge(_lookup_for(func_right, 'b'), on='assembly_id_b', how='left')
    if len(out) != before:
        raise RuntimeError(
            f"attach_cds_dna_hash_to_pos_df: row count changed "
            f"({before} -> {len(out)}); fan-out from non-unique key suspected."
        )
    missing = out[['cds_dna_hash_a', 'cds_dna_hash_b']].isna().any(axis=1).sum()
    if missing:
        raise RuntimeError(
            f"attach_cds_dna_hash_to_pos_df: {missing} rows have no "
            f"matching CDS — check that schema_pair functions match "
            f"those used to build cds_final."
        )
    print(f"attach_cds_dna_hash_to_pos_df: attached cds_dna_hash to "
          f"{len(out):,} pos_df rows "
          f"(unique cds_dna_hash_a={out['cds_dna_hash_a'].nunique():,}, "
          f"cds_dna_hash_b={out['cds_dna_hash_b'].nunique():,}).")
    return out


def select_balanced_isolate_pool(
    prot_df: pd.DataFrame,
    subtype_selection_cfg,
    master_seed: int,
    output_dir: Path,
    ) -> pd.DataFrame:
    """Downsample isolates to equal per-subtype representation.

    No-op when mode=natural (returns prot_df unchanged). When mode=balanced,
    filters `prot_df` to only isolates drawn from `included_subtypes`, sampling
    N assembly_ids per subtype where N is min(available counts) if
    target_count_per_subtype=='min' or the configured integer otherwise.

    Determinism: pure function of (seed, included_subtypes,
    target_count_per_subtype, prot_df's per-subtype assembly_id sets). Uses a
    local random.Random instance (no reads from the process-global random
    stream) and iterates subtypes / assembly_ids in sorted order.

    Upsampling is not supported -- errors if target_count_per_subtype exceeds
    any subtype's available count. See docs/plans/subtype_balancing_plan.md
    for the design and error rules.

    When mode=balanced, writes subtype_selection_manifest.json to output_dir
    with per-subtype counts, selected assembly_ids, and the seed used.
    """
    mode = getattr(subtype_selection_cfg, 'mode', 'natural') if subtype_selection_cfg is not None else 'natural'
    if mode == 'natural':
        print("Subtype balancing: mode=natural, no-op")
        return prot_df
    if mode != 'balanced':
        raise ValueError(
            f"dataset.subtype_selection.mode must be 'natural' or 'balanced'; got {mode!r}"
        )

    if 'hn_subtype' not in prot_df.columns:
        raise RuntimeError(
            "hn_subtype column missing from prot_df. "
            "select_balanced_isolate_pool must run after enrich_prot_data_with_metadata."
        )

    included_raw = getattr(subtype_selection_cfg, 'included_subtypes', None)
    included = list(included_raw) if included_raw else []
    if not included:
        raise ValueError(
            "dataset.subtype_selection.mode=balanced requires a non-empty "
            "dataset.subtype_selection.included_subtypes list."
        )
    included_sorted = sorted(set(included))

    counts = (
        prot_df[prot_df['hn_subtype'].isin(included_sorted)]
        .groupby('hn_subtype')['assembly_id']
        .nunique()
    )

    missing = [st for st in included_sorted if st not in counts.index or int(counts[st]) == 0]
    if missing:
        available_top = (
            prot_df['hn_subtype']
            .value_counts(dropna=False)
            .head(20)
            .to_dict()
        )
        raise ValueError(
            f"dataset.subtype_selection.included_subtypes references subtype(s) "
            f"not present in the dataframe: {missing}. "
            f"Top-20 available subtype counts (after upstream filters): {available_top}"
        )

    target = getattr(subtype_selection_cfg, 'target_count_per_subtype', 'min')
    min_count = int(counts.min())
    if target == 'min':
        N = min_count
    elif isinstance(target, int) and not isinstance(target, bool):
        if target <= 0:
            raise ValueError(
                f"dataset.subtype_selection.target_count_per_subtype must be positive; got {target}"
            )
        if target > min_count:
            per_st = {st: int(counts[st]) for st in included_sorted}
            raise ValueError(
                f"target_count_per_subtype={target} exceeds the minimum available count "
                f"({min_count}). Upsampling is not supported. "
                f"Per-subtype available counts: {per_st}. "
                f"Lower the target or set target_count_per_subtype='min'."
            )
        N = target
    else:
        raise ValueError(
            f"dataset.subtype_selection.target_count_per_subtype must be 'min' or "
            f"a positive int; got {target!r}"
        )

    seed_cfg = getattr(subtype_selection_cfg, 'seed', None)
    seed = int(seed_cfg) if seed_cfg is not None else int(master_seed)
    rng = random.Random(seed)

    selected_ids: list[str] = []
    per_subtype_report: dict = {}
    for st in included_sorted:
        aids = sorted(prot_df.loc[prot_df['hn_subtype'] == st, 'assembly_id'].unique().tolist())
        picked = rng.sample(aids, N)
        selected_ids.extend(picked)
        per_subtype_report[st] = {'available': int(counts[st]), 'selected': N}

    before_isolates = int(prot_df['assembly_id'].nunique())
    filtered = prot_df[prot_df['assembly_id'].isin(set(selected_ids))].reset_index(drop=True)
    after_isolates = int(filtered['assembly_id'].nunique())

    manifest = {
        'mode': 'balanced',
        'included_subtypes': included_sorted,
        'target_count_per_subtype': target if target == 'min' else int(target),
        'N': N,
        'seed': seed,
        'per_subtype': per_subtype_report,
        'isolates_before': before_isolates,
        'isolates_after': after_isolates,
        'selected_assembly_ids': sorted(selected_ids),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'subtype_selection_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nSubtype balancing: {included_sorted} at N={N} per subtype (seed={seed})")
    for st in included_sorted:
        r = per_subtype_report[st]
        print(f"  {st}: selected {r['selected']:,} / {r['available']:,} available")
    print(f"Isolates before balancing: {before_isolates:,}   after: {after_isolates:,}")
    print(f"Wrote manifest: {manifest_path}")
    return filtered


def orient_pair_by_schema(dct: dict, func_left: str, func_right: str) -> Optional[dict]:
    """Force pair orientation by a (func_left, func_right) schema.

    In schema-ordered mode, slot semantics are fixed:
      - slot A must be func_left
      - slot B must be func_right

    If the input pair matches the schema in either order, this function returns
    a dict where all *_a/*_b fields are oriented so that func_a==func_left and
    func_b==func_right. If the pair does not match the schema, returns None.
    """
    fa = dct.get("func_a")
    fb = dct.get("func_b")
    if fa == func_left and fb == func_right:
        return dct
    if fa == func_right and fb == func_left:
        for k in list(dct.keys()):
            if not k.endswith("_a"):
                continue
            kb = k[:-2] + "_b"
            if kb in dct:
                dct[k], dct[kb] = dct[kb], dct[k]
        return dct
    return None


def compute_isolate_pair_counts(
    df: pd.DataFrame,
    verbose: bool = False,
    pair_mode: str = "unordered",
    schema_pair: Optional[Tuple[str, str]] = None,
    ) -> dict:
    """Compute positive pair counts per isolate for stratified splitting.

    Counts how many positive pairs (cross-function pairs within the same
    isolate) can be generated from each isolate. This is used to stratify
    isolates by positive pair count for balanced train/val/test splits.

    Note: data filtering (e.g., by selected_functions) should be done before
    calling this function. This function does not validate data quality.
    """
    isolate_pos_counts = {}

    if pair_mode not in {"unordered", "schema_ordered"}:
        raise ValueError(f"Unknown pair_mode: {pair_mode!r}")
    if pair_mode == "schema_ordered":
        if schema_pair is None or len(schema_pair) != 2:
            raise ValueError("schema_ordered mode requires schema_pair=(func_left, func_right)")
        func_left, func_right = schema_pair
        if func_left == func_right:
            raise ValueError("schema_pair must contain two different functions (func_left != func_right)")

    for aid, grp in df.groupby('assembly_id'):
        n_proteins = len(grp)

        if verbose:
            print(f"assembly_id {aid}: {n_proteins} proteins, Functions: {grp['function'].tolist()}")

        if n_proteins < 2:
            isolate_pos_counts[aid] = 0
        elif pair_mode == "schema_ordered":
            n_left = int((grp['function'] == func_left).sum())
            n_right = int((grp['function'] == func_right).sum())
            isolate_pos_counts[aid] = n_left * n_right
        else:
            pairs = list(combinations(grp.itertuples(), 2))
            pos_count = sum(1 for row_a, row_b in pairs if row_a.function != row_b.function)
            isolate_pos_counts[aid] = pos_count

    return isolate_pos_counts


def build_cooccurrence_set(df: pd.DataFrame, hash_col: str = 'prot_hash') -> tuple[set, dict]:
    """Build a set of all hash pairs that co-occur in any isolate.

    Two sequences "co-occur" if they appear together in the same isolate (same
    assembly_id). Used to prevent contradictory labels: if (A, B) co-occur in
    isolate X (positive), they cannot also be sampled as a negative.

    Args:
        df: row-per-(isolate, protein) DataFrame; must contain `assembly_id`
            and the selected `hash_col`.
        hash_col: which hash column to use as the cooccurrence key. Default
            'prot_hash' (protein-level; aa pair_key family). Pass
            'cds_dna_hash' for the nt_cds pair_key family. The choice must
            match the alphabet of pair_keys used downstream (in
            `create_positive_pairs_v2` and `create_negative_pairs_v2`);
            mismatched cooccur and pair_key alphabets would silently allow
            (or block) the wrong pairs as negatives.

    Returns:
        - cooccur_pairs: Set of canonical pair keys (`canonical_pair_key`
          format) for all hash pairs that co-occur in at least one isolate.
        - cooccur_stats: dict with `total_cooccur_pairs`,
          `max_isolates_per_pair`, `pairs_in_multiple_isolates`,
          `isolate_pair_counts` (full pair_key -> count mapping), and
          `hash_col` (the alphabet this set was built on).
    """
    if hash_col not in df.columns:
        raise ValueError(
            f"build_cooccurrence_set: hash_col={hash_col!r} not in df.columns. "
            f"For pair_key_alphabet='nt_cds' the caller must attach "
            f"cds_dna_hash to df before calling."
        )
    cooccur_pairs = set()
    isolate_pair_counts = {}
    for aid, grp in df.groupby('assembly_id'):
        if len(grp) < 2:
            continue

        # Unique hashes for this isolate, alphabet selected by hash_col.
        hashes = grp[hash_col].unique().tolist()

        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                seq_pair_key = canonical_pair_key(hashes[i], hashes[j])
                cooccur_pairs.add(seq_pair_key)

                if seq_pair_key not in isolate_pair_counts:
                    isolate_pair_counts[seq_pair_key] = 0
                isolate_pair_counts[seq_pair_key] += 1

    # - max_isolates_per_pair is the maximum number of isolates that a specific sequence pair co-occurs in.
    # - pairs_in_multiple_isolates counts how many unique sequence pairs co-occur in more than one isolate.
    # - pairs_in_multiple_isolates / total_cooccur_pairs is measure: out of all co-occurring unique sequence pairs
    #   how many are (biologically) duplicated across multiple isolates
    cooccur_stats = {
        'total_cooccur_pairs': len(cooccur_pairs),
        'max_isolates_per_pair': max(isolate_pair_counts.values()) if isolate_pair_counts else 0,
        'pairs_in_multiple_isolates': sum(1 for c in isolate_pair_counts.values() if c > 1),
        'isolate_pair_counts': isolate_pair_counts,
        'hash_col': hash_col,
    }

    return cooccur_pairs, cooccur_stats


def get_metadata_distributions(df: pd.DataFrame, isolate_set: set) -> dict:
    """Return metadata value-count dicts for a set of isolates."""
    if len(isolate_set) == 0:
        return {'host': {}, 'year': {}, 'hn_subtype': {}, 'geo_location_clean': {}, 'passage': {}}
    isolate_meta = df[df['assembly_id'].isin(isolate_set)].groupby('assembly_id').first()
    distributions = {}
    for col in ['host', 'year', 'hn_subtype', 'geo_location_clean', 'passage']:
        if col in isolate_meta.columns:
            counts = isolate_meta[col].value_counts(dropna=False)
            distributions[col] = {
                str(k) if pd.notna(k) else 'null': int(v) for k, v in counts.items()
            }
        else:
            distributions[col] = {}
    return distributions


def bipartite_components(
    pos_df: pd.DataFrame,
    hash_key: str = 'seq',
    col_a: Optional[str] = None,
    col_b: Optional[str] = None,
    ) -> tuple[pd.Series, dict]:
    """Connected components of the bipartite (side-A, side-B) hash graph.

    Two parameter styles, mutually exclusive:

    1. `hash_key` (legacy): selects a hash family — 'seq' (protein-level,
       columns prot_hash_a / prot_hash_b — stricter) or 'dna' (nucleotide-
       level — looser; allows synonymous-mutation variants to land in
       different splits).
    2. `col_a` / `col_b` (explicit): pass arbitrary node-id column names.
       Used by cluster-disjoint routing (col_a='cluster_id_a',
       col_b='cluster_id_b'). When given, `hash_key` is ignored.

    Edges: each unique `(node_a, node_b)` tuple. Computed by iterative-
    path-compression union-find (no networkx dependency).

    Returns:
        (component_id, summary)
        component_id: pd.Series aligned with pos_df.index, dtype int.
            Each row is labelled with its component's representative id
            (a contiguous integer 0..n_components-1).
        summary: dict with `n_components`, `n_pairs`, `hash_key`,
            `n_hashes_a`, `n_hashes_b`, `largest_component_pairs`,
            `top_10_sizes`, `singleton_components`.

    The component label is stable under reordering of pos_df rows but
    not under reordering of (a, b) sides. v2 schema-mode positives are
    always (func_left, func_right) so this is fine.
    """
    if col_a is not None or col_b is not None:
        if col_a is None or col_b is None:
            raise ValueError("bipartite_components: pass both col_a and col_b, or neither.")
        node_key_label = f'explicit({col_a},{col_b})'
    else:
        if hash_key not in {'seq', 'dna'}:
            raise ValueError(
                f"bipartite_components: hash_key must be 'seq' or 'dna'; "
                f"got {hash_key!r}."
            )
        # hash_key names the family ('seq'/'dna'); the column is the molecule-level
        # hash from the schema registry (seq -> prot_hash, dna -> ctg_dna_hash).
        _hk_col = {'seq': schema.hash_col('aa'), 'dna': schema.hash_col('nt_ctg')}
        col_a = f'{_hk_col[hash_key]}_a'
        col_b = f'{_hk_col[hash_key]}_b'
        node_key_label = hash_key
    if not {col_a, col_b}.issubset(pos_df.columns):
        raise ValueError(
            f"bipartite_components: pos_df must contain {col_a} and {col_b} "
            f"columns (key={node_key_label!r})."
        )

    # Union-find on string node ids ('a:'+hash, 'b:'+hash). String prefix
    # avoids the (rare) chance of a side-a hash colliding with a side-b
    # hash collapsing two distinct nodes; md5 collision is astronomically
    # unlikely but the prefix costs nothing.
    parent: dict = {}
    def find(x: str) -> str:
        # Iterative path compression
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root
    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    edges = pos_df[[col_a, col_b]].drop_duplicates()
    for h in pos_df[col_a].unique():
        parent[f'a:{h}'] = f'a:{h}'
    for h in pos_df[col_b].unique():
        parent[f'b:{h}'] = f'b:{h}'
    for h_a, h_b in zip(edges[col_a].values, edges[col_b].values):
        union(f'a:{h_a}', f'b:{h_b}')

    # Map each row to its component's root, then to a contiguous int id.
    row_roots = [find(f'a:{h}') for h in pos_df[col_a].values]
    root_to_int: dict = {}
    for r in sorted(set(row_roots)):
        root_to_int[r] = len(root_to_int)
    component_id = pd.Series(
        [root_to_int[r] for r in row_roots],
        index=pos_df.index,
        dtype='int64',
        name='component_id',
    )

    sizes = component_id.value_counts().sort_values(ascending=False)
    summary = {
        'n_components': int(component_id.nunique()),
        'n_pairs': int(len(pos_df)),
        'hash_key': node_key_label,
        'col_a': col_a,
        'col_b': col_b,
        'n_hashes_a': int(pos_df[col_a].nunique()),
        'n_hashes_b': int(pos_df[col_b].nunique()),
        'largest_component_pairs': int(sizes.iloc[0]) if len(sizes) else 0,
        'top_10_sizes': [int(s) for s in sizes.head(10).tolist()],
        'singleton_components': int((sizes == 1).sum()),
    }
    return component_id, summary


def _lpt_bin_pack(
    sizes: pd.Series,
    targets: dict,
    bin_order: list,
) -> dict:
    """LPT-greedy bin-packing: largest group to bin with biggest deficit.

    Args:
        sizes: pd.Series mapping group_id -> count.
        targets: dict mapping bin_name -> target count (raw, not fraction).
        bin_order: list of bin names defining tie-break order when two bins
            have equal deficit. Python's `max` returns the first element of
            a tied maximum, so this order is deterministic.

    Returns:
        dict mapping group_id -> bin_name.

    Atom ordering is pinned to `(-size, cluster_id)` ascending — the same
    key sklearn `GroupKFold` derives via `np.unique` (sorted) +
    `np.argsort(-counts)`, so the LPT and GroupKFold paths agree on which
    atom is "largest" (per D1 of
    docs/plans/done/2026-05-27_kfold_variance_estimation_plan.md).

    Shared by seq_disjoint_route_pos_df (this module) and the cluster_disjoint
    router in _split_helpers.py.
    """
    def _sort_key(c):
        try:
            return (-int(sizes.loc[c]), int(c))
        except (ValueError, TypeError):
            return (-int(sizes.loc[c]), str(c))
    sorted_groups = sorted(sizes.index, key=_sort_key)

    bin_count = {b: 0 for b in bin_order}
    group_to_bin: dict = {}
    for g in sorted_groups:
        s = int(sizes.loc[g])
        deficits = {b: targets[b] - bin_count[b] for b in bin_order}
        winner = max(bin_order, key=lambda b: deficits[b])
        group_to_bin[g] = winner
        bin_count[winner] += s
    return group_to_bin


def route_holdout(
    pos_df: pd.DataFrame,
    atom_id: pd.Series,
    train_ratio: float,
    val_ratio: float,
) -> tuple:
    """Pack atoms into an 80/10/10-target holdout and return the row partition.

    The shared 'router' core for every atom-based mode (seq_disjoint,
    cluster_disjoint): given a per-row atom label, LPT-greedy bin-pack the atoms
    by pair count, then slice `pos_df` into (train, val, test). The atom
    definition (bipartite CC on hash / on cluster / single-slot cluster) is the
    mode-specific part; this packing + slicing is identical across modes.

    Args:
        pos_df: positive-pair rows.
        atom_id: Series aligned to pos_df.index -> atom label; rows sharing an
            atom are indivisible (stay in one split).
        train_ratio, val_ratio: split-size targets; test = 1 - train - val.

    Returns:
        (train_pos, val_pos, test_pos, atom_to_split, targets) -- three
        row-disjoint DataFrames (index reset), the atom -> split-name dict, and
        the per-split target pair counts (for the caller's audit reporting).
    """
    n_pairs = int(len(pos_df))
    targets = {
        'train': train_ratio * n_pairs,
        'val':   val_ratio * n_pairs,
        'test':  (1.0 - train_ratio - val_ratio) * n_pairs,
    }
    sizes = atom_id.value_counts().sort_index()
    atom_to_split = _lpt_bin_pack(sizes, targets, ['train', 'val', 'test'])
    split_for_row = atom_id.map(atom_to_split)
    train_pos = pos_df[split_for_row == 'train'].reset_index(drop=True)
    val_pos = pos_df[split_for_row == 'val'].reset_index(drop=True)
    test_pos = pos_df[split_for_row == 'test'].reset_index(drop=True)
    return train_pos, val_pos, test_pos, atom_to_split, targets


def seq_disjoint_route_pos_df(
    pos_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    hash_key: str = 'seq',
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Route `pos_df` rows into train/val/test by bipartite-component LPT-greedy.

    Each connected component in the bipartite (side-A, side-B) hash graph
    is indivisible: the whole component lands in one split. The graph is
    built on the hash family selected by `hash_key`:

      - hash_key='seq' (default): protein-level partitioning via
        prot_hash_a / prot_hash_b. The stricter choice — guarantees no
        protein appears in two splits. The right key for ESM-2 features
        (a protein-level feature) and also a strict superset of the DNA
        guarantee since two synonymous-mutation DNAs share their protein.
      - hash_key='dna': nucleotide-level partitioning via ctg_dna_hash_a /
        ctg_dna_hash_b. Looser — synonymous-mutation variants of the same
        protein can land in different splits. Appropriate when the
        downstream feature is DNA-derived (k-mer) and you want maximum
        sample-level granularity.

    LPT-greedy bin-packing: sort components by size desc (then by
    component-id for deterministic tie-breaking), assign each to the bin
    whose remaining-capacity deficit (target - current) is largest.

    `seed` is reserved for future tie-break shuffling but is not consumed
    in this implementation (the algorithm is fully deterministic without
    it).

    The returned audit dict reports overlap counts for BOTH prot_hash and
    ctg_dna_hash families regardless of `hash_key`, so the diagnostic value
    is preserved either way. The by-construction guarantee is on the
    `hash_key`-selected family; the other family is reported as a
    secondary diagnostic.

    Returns:
        (train_pos, val_pos, test_pos, audit) -- the three DataFrames are
        row-disjoint partitions of pos_df preserving original column
        order; `audit` is a JSON-serializable dict.
    """
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError(
            f"seq_disjoint_route_pos_df: invalid ratios "
            f"train_ratio={train_ratio}, val_ratio={val_ratio}"
        )
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError(
            f"seq_disjoint_route_pos_df: train+val ratios sum to >1 "
            f"({train_ratio} + {val_ratio})"
        )
    if hash_key not in {'seq', 'dna'}:
        raise ValueError(
            f"seq_disjoint_route_pos_df: hash_key must be 'seq' or 'dna'; "
            f"got {hash_key!r}."
        )

    component_id, cc_summary = bipartite_components(pos_df, hash_key=hash_key)

    # Atom = bipartite connected component; the shared router (P2) packs atoms
    # LPT-greedy and slices pos_df. Only the atom definition is seq_disjoint-
    # specific — seq_disjoint and cluster_disjoint share route_holdout.
    n_pairs = int(len(pos_df))
    train_pos, val_pos, test_pos, _atom_to_split, targets = route_holdout(
        pos_df, component_id, train_ratio, val_ratio,
    )

    achieved = {'train': len(train_pos), 'val': len(val_pos), 'test': len(test_pos)}

    # Cross-split hash overlap. For the family selected by `hash_key` the
    # by-construction guarantee is that all six counters are 0; the
    # saver hard-fails if any are non-zero. The OTHER family is reported
    # as a diagnostic — under hash_key='seq', ctg_dna_hash overlaps are
    # always 0 too (protein equality implies DNA grouping). Under
    # hash_key='dna', prot_hash overlaps are usually non-zero (synonymous
    # mutations of the same protein land in different splits).
    def _set(df: pd.DataFrame, col: str) -> set:
        return set(df[col].dropna())
    overlaps_by_family: dict = {}
    # 'seq'/'dna' track the split_strategy.hash_key knob (NOT renamed); the
    # column each reads is the molecule-level hash from the schema registry
    # (aa -> prot_hash, nt_ctg -> ctg_dna_hash).
    for family, _alphabet in (('seq', 'aa'), ('dna', 'nt_ctg')):
        _hcol = schema.hash_col(_alphabet)
        family_overlaps: dict = {}
        for side in ('a', 'b'):
            col = f'{_hcol}_{side}'
            if col not in pos_df.columns:
                continue
            sets = {sp: _set(d, col) for sp, d in
                    (('train', train_pos), ('val', val_pos), ('test', test_pos))}
            family_overlaps[side] = {
                'train_val':   len(sets['train'] & sets['val']),
                'train_test':  len(sets['train'] & sets['test']),
                'val_test':    len(sets['val'] & sets['test']),
            }
        if family_overlaps:
            overlaps_by_family[family] = family_overlaps

    # targets_pct / achieved_pct kept on 0-100 (backwards-compat).
    # max_target_deviation_pct -> _pp + rescale to 0-1 per D5 of the k-fold
    # plan, extended to seq_disjoint for cross-audit consistency.
    target_pcts = {k: 100.0 * v / n_pairs for k, v in targets.items()}
    achieved_pcts = {k: 100.0 * v / n_pairs for k, v in achieved.items()}
    target_frac   = {k: v / n_pairs for k, v in targets.items()}
    achieved_frac = {k: v / n_pairs for k, v in achieved.items()}
    audit = {
        'mode': 'seq_disjoint',
        'algorithm': 'bipartite_cc_lpt_greedy',
        'hash_key': hash_key,
        'seed': int(seed),
        'cc_summary': cc_summary,
        'targets_pairs': {k: int(round(v)) for k, v in targets.items()},
        'targets_pct':   {k: round(v, 4) for k, v in target_pcts.items()},
        'achieved_pairs': achieved,
        'achieved_pct':   {k: round(v, 4) for k, v in achieved_pcts.items()},
        'max_target_deviation_pp': round(
            max(abs(achieved_frac[k] - target_frac[k]) for k in achieved_frac), 4
        ),
        'pairs_dropped': 0,  # CC-bin-packing never splits a component.
        # Primary guarantee (hard-fail in saver): overlaps_by_family[hash_key].
        # Other family is diagnostic. Audit-output keys stay 'seq'/'dna' (keyed
        # by hash_key in the saver); only the columns they read are renamed.
        'seq_hash_overlap': overlaps_by_family.get('seq', {}),
        'dna_hash_overlap': overlaps_by_family.get('dna', {}),
    }
    return train_pos, val_pos, test_pos, audit


# Axes supported on every slot of a metadata_holdout filter dict. Used both
# by compute_metadata_holdout_isolates and by the v2 config validator to
# reject unknown keys with a clear migration message.
_HOLDOUT_AXES = ('hn_subtype', 'host', 'year', 'year_range', 'geo_location', 'passage')
_HOLDOUT_ORDERED_AXES = ('year',)  # axes that accept a *_range counterpart


def _validate_holdout_filter_spec(spec: dict, slot: str) -> dict:
    """Validate a single train/val/test filter dict; return a clean dict.

    Catches typos (unknown axis keys), ordered-axis misuse (host_range etc.),
    and structural errors (empty list, year + year_range both set) before
    handing the spec to filter_by_metadata, which does its own value-level
    validation.
    """
    if spec is None:
        raise ValueError(
            f"metadata_holdout.{slot}: filter spec is null. To opt out of "
            f"this slot, omit the key (only 'val' is optional)."
        )
    if not isinstance(spec, dict):
        raise ValueError(
            f"metadata_holdout.{slot}: filter must be a dict; got {type(spec).__name__}."
        )

    # Unknown keys -- look for typos first since they're the most common
    # mistake (e.g., 'subtype' instead of 'hn_subtype'). Range keys on
    # unordered axes get a more specific message.
    bad_range_axes = {
        f'{axis}_range' for axis in ('hn_subtype', 'host', 'geo_location', 'passage')
    }
    for key in spec.keys():
        if key in _HOLDOUT_AXES:
            continue
        if key in bad_range_axes:
            raise ValueError(
                f"metadata_holdout.{slot}: '{key}' is not a supported key -- "
                f"{key.split('_range')[0]} is not an ordered axis. Pass a "
                f"list to '{key.split('_range')[0]}' for set membership instead."
            )
        raise ValueError(
            f"metadata_holdout.{slot}: unknown axis key {key!r}. Supported: "
            f"{sorted(_HOLDOUT_AXES)}."
        )

    return dict(spec)


def filter_by_metadata(
    prot_df: pd.DataFrame,
    hn_subtype=None,
    host=None,
    year=None,
    year_range=None,
    geo_location=None,
    passage=None,
    ) -> pd.DataFrame:
    """Filter protein records by isolate-level metadata criteria.

    Filtering is performed at the isolate level: isolates matching ALL specified
    criteria are identified, then ALL protein records from those isolates are
    kept. This ensures we don't lose proteins due to metadata merge issues
    (e.g., an isolate has host metadata on one protein but not another).

    Each axis parameter accepts:
      - None: no constraint on that axis (default).
      - scalar (str/int): exact match.
      - list / tuple: **set membership** (any length, including 1-element
        ``[2020]``). Length-based semantics are deliberately rejected -- a
        2-element list is always a set, never a range, to avoid surprises.

    ``year_range`` (year-axis only) accepts a 2-element ``[min, max]`` list
    for inclusive-range matching; mutually exclusive with ``year``. There is
    no ``host_range`` / ``hn_subtype_range`` / etc. -- those axes are not
    ordered. Pass a list for set membership instead.

    Raises ValueError on: ``year`` and ``year_range`` both set; malformed
    ``year_range``; empty list; geo_location / passage filter requested but
    the corresponding column is missing from ``prot_df``.
    """
    if year is not None and year_range is not None:
        raise ValueError(
            "filter_by_metadata: 'year' and 'year_range' are mutually exclusive."
        )

    year_min, year_max = None, None
    if year_range is not None:
        if not isinstance(year_range, (list, tuple)) or len(year_range) != 2:
            raise ValueError(
                f"filter_by_metadata: year_range must be a 2-element [min, max] "
                f"list; got {year_range!r}."
            )
        try:
            year_min = int(year_range[0])
            year_max = int(year_range[1])
        except (TypeError, ValueError):
            raise ValueError(
                f"filter_by_metadata: year_range elements must be ints; "
                f"got {year_range!r}."
            )
        if year_min > year_max:
            raise ValueError(
                f"filter_by_metadata: year_range min ({year_min}) is greater "
                f"than max ({year_max})."
            )

    def _to_list(value, axis: str, coerce_int: bool = False):
        """Normalize scalar -> [scalar], pass through list/tuple as list.

        Empty lists are rejected (probably a config typo: an empty filter
        means "match nothing" which is never what a user wants). When
        ``coerce_int`` is set, all entries are int-coerced (year semantics).
        """
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            items = list(value)
        else:
            items = [value]
        if len(items) == 0:
            raise ValueError(
                f"filter_by_metadata: {axis} list cannot be empty."
            )
        if coerce_int:
            try:
                items = [int(v) for v in items]
            except (TypeError, ValueError):
                raise ValueError(
                    f"filter_by_metadata: {axis} values must be ints; got {value!r}."
                )
        return items

    hn_set = _to_list(hn_subtype, 'hn_subtype')
    host_set = _to_list(host, 'host')
    year_set = _to_list(year, 'year', coerce_int=True)
    geo_set = _to_list(geo_location, 'geo_location')
    passage_set = _to_list(passage, 'passage')

    any_filter = any(v is not None for v in
                     [hn_set, host_set, year_set, year_range, geo_set, passage_set])
    if not any_filter:
        return prot_df

    year_disp = year_set if year_set is not None else (
        f"[{year_min}..{year_max}]" if year_range is not None else None)
    print('\nMetadata filtering enabled.')
    print(f'  hn_subtype:   {hn_set}')
    print(f'  host:         {host_set}')
    print(f'  year:         {year_disp}')
    print(f'  geo_location: {geo_set}')
    print(f'  passage:      {passage_set}')

    meta_cols = ['assembly_id', 'hn_subtype', 'host', 'year']
    if 'geo_location_clean' in prot_df.columns:
        meta_cols.append('geo_location_clean')
    if 'passage' in prot_df.columns:
        meta_cols.append('passage')
    aid_meta = prot_df.groupby('assembly_id')[meta_cols].first().reset_index(drop=True)

    aid_mask = pd.Series([True] * len(aid_meta))

    if host_set is not None:
        before = aid_mask.sum()
        aid_mask = aid_mask & aid_meta['host'].isin(host_set)
        print(f"   host {host_set}: {before:,} -> {aid_mask.sum():,} isolates")

    if year_set is not None:
        before = aid_mask.sum()
        aid_mask = aid_mask & aid_meta['year'].isin(year_set)
        print(f"   year {year_set}: {before:,} -> {aid_mask.sum():,} isolates")
    elif year_range is not None:
        before = aid_mask.sum()
        aid_mask = aid_mask & aid_meta['year'].between(year_min, year_max, inclusive='both')
        print(f"   year [{year_min}..{year_max}]: {before:,} -> {aid_mask.sum():,} isolates")

    if hn_set is not None:
        before = aid_mask.sum()
        aid_mask = aid_mask & aid_meta['hn_subtype'].isin(hn_set)
        print(f"   hn_subtype {hn_set}: {before:,} -> {aid_mask.sum():,} isolates")

    if geo_set is not None:
        if 'geo_location_clean' not in aid_meta.columns:
            raise ValueError(
                "filter_by_metadata: geo_location filter requested but "
                "'geo_location_clean' column is missing from prot_df."
            )
        before = aid_mask.sum()
        aid_mask = aid_mask & aid_meta['geo_location_clean'].isin(geo_set)
        print(f"   geo_location {geo_set}: {before:,} -> {aid_mask.sum():,} isolates")

    if passage_set is not None:
        if 'passage' not in aid_meta.columns:
            raise ValueError(
                "filter_by_metadata: passage filter requested but "
                "'passage' column is missing from prot_df."
            )
        before = aid_mask.sum()
        aid_mask = aid_mask & aid_meta['passage'].isin(passage_set)
        print(f"   passage {passage_set}: {before:,} -> {aid_mask.sum():,} isolates")

    matching_isolates = aid_meta[aid_mask]['assembly_id'].tolist()
    n_before = len(prot_df)
    prot_df = prot_df[prot_df['assembly_id'].isin(matching_isolates)].reset_index(drop=True)
    n_after = len(prot_df)

    print(f"   Filtered to {len(matching_isolates):,} isolates matching criteria")
    print(f"   Protein records: {n_before:,} -> {n_after:,} ({100*n_after/n_before:.1f}%)")

    return prot_df


def drop_ambiguous_hn_subtype(prot_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop isolates whose hn_subtype is not fully specified (``^H\\d+N\\d+$``).

    On the full Flu A enriched set (~108,530 isolates) this removes ~1,006 rows
    (~0.93%) -- almost all are ``'HN'`` where both H and N indices are unknown,
    plus a couple of singletons with incomplete N typing (``'H1N'``, ``'H3N'``).
    Dropping them at source eliminates the ``unknown_metadata_neg`` regime from
    downstream stratified eval, which otherwise gets a small "ambiguous"
    bucket that complicates per-regime comparisons.

    The caller is expected to honor the ``dataset.drop_ambiguous_subtype``
    config knob (default True); this helper does the unconditional work and
    returns a summary the caller can log.

    Returns:
        (filtered_prot_df, summary) where summary is::

            {
              'n_isolates_total':   int,
              'n_isolates_dropped': int,
              'n_isolates_kept':    int,
              'n_rows_total':       int,
              'n_rows_dropped':     int,
              'n_rows_kept':        int,
              'value_counts':       dict[str, int],   # per-value drop tally
            }

    No-op when ``hn_subtype`` is missing from ``prot_df`` (returns the df
    unchanged with a summary noting zero drops).
    """
    summary = {
        'n_isolates_total': int(prot_df['assembly_id'].nunique()),
        'n_isolates_dropped': 0,
        'n_isolates_kept': int(prot_df['assembly_id'].nunique()),
        'n_rows_total': int(len(prot_df)),
        'n_rows_dropped': 0,
        'n_rows_kept': int(len(prot_df)),
        'value_counts': {},
    }
    if 'hn_subtype' not in prot_df.columns:
        return prot_df, summary

    iso_sub = prot_df.groupby('assembly_id')['hn_subtype'].first()
    hxny = iso_sub.fillna('').astype(str).str.fullmatch(r'H\d+N\d+').fillna(False)
    ambig = iso_sub[~hxny]
    if len(ambig) == 0:
        return prot_df, summary

    summary['value_counts'] = {
        # value_counts keys can be NaN; coerce to the literal string '(null)'
        # for JSON-friendliness and to keep the dict serializable as-is.
        ('(null)' if pd.isna(k) else str(k)): int(v)
        for k, v in ambig.value_counts(dropna=False).items()
    }
    keep_aids = set(iso_sub.index) - set(ambig.index)
    n_rows_before = len(prot_df)
    out_df = prot_df[prot_df['assembly_id'].isin(keep_aids)].reset_index(drop=True)

    summary['n_isolates_dropped'] = int(len(ambig))
    summary['n_isolates_kept'] = int(len(keep_aids))
    summary['n_rows_dropped'] = int(n_rows_before - len(out_df))
    summary['n_rows_kept'] = int(len(out_df))
    return out_df, summary


def _isolates_matching_holdout_spec(meta_per_iso: pd.DataFrame, spec: dict) -> set:
    """Return the set of assembly_ids that match a metadata_holdout filter spec.

    Operates on a per-isolate metadata frame (one row per assembly_id) and
    returns just the matching assembly_ids -- no row-level protein-df
    filtering. This is what compute_metadata_holdout_isolates needs: three
    isolate-id sets to hand to split_dataset_v2's override hook.

    Validation matches filter_by_metadata's: scalar/list set membership,
    year_range for inclusive range on year, no _range on unordered axes,
    no empty lists, year + year_range mutually exclusive.
    """
    if not spec:
        # Empty spec means "match everything" -- every isolate qualifies.
        return set(meta_per_iso['assembly_id'])

    year = spec.get('year')
    year_range = spec.get('year_range')
    if year is not None and year_range is not None:
        raise ValueError(
            "metadata_holdout filter: 'year' and 'year_range' are mutually exclusive."
        )

    mask = pd.Series([True] * len(meta_per_iso), index=meta_per_iso.index)

    def _normalize(value, axis: str, coerce_int: bool = False):
        if value is None:
            return None
        items = list(value) if isinstance(value, (list, tuple)) else [value]
        if len(items) == 0:
            raise ValueError(f"metadata_holdout filter: {axis} list cannot be empty.")
        if coerce_int:
            try:
                items = [int(v) for v in items]
            except (TypeError, ValueError):
                raise ValueError(
                    f"metadata_holdout filter: {axis} values must be ints; got {value!r}."
                )
        return items

    for axis, col, coerce in [
        ('host', 'host', False),
        ('hn_subtype', 'hn_subtype', False),
        ('passage', 'passage', False),
        ('geo_location', 'geo_location_clean', False),
        ('year', 'year', True),
    ]:
        items = _normalize(spec.get(axis), axis, coerce_int=coerce)
        if items is None:
            continue
        if col not in meta_per_iso.columns:
            raise ValueError(
                f"metadata_holdout filter: '{axis}' requested but column "
                f"{col!r} is missing from the per-isolate metadata."
            )
        mask = mask & meta_per_iso[col].isin(items)

    if year_range is not None:
        if not isinstance(year_range, (list, tuple)) or len(year_range) != 2:
            raise ValueError(
                f"metadata_holdout filter: year_range must be a 2-element "
                f"[min, max] list; got {year_range!r}."
            )
        try:
            ymin, ymax = int(year_range[0]), int(year_range[1])
        except (TypeError, ValueError):
            raise ValueError(
                f"metadata_holdout filter: year_range elements must be ints; "
                f"got {year_range!r}."
            )
        if ymin > ymax:
            raise ValueError(
                f"metadata_holdout filter: year_range min ({ymin}) is greater "
                f"than max ({ymax})."
            )
        if 'year' not in meta_per_iso.columns:
            raise ValueError(
                "metadata_holdout filter: 'year_range' requested but 'year' "
                "column is missing from the per-isolate metadata."
            )
        mask = mask & meta_per_iso['year'].between(ymin, ymax, inclusive='both')

    return set(meta_per_iso.loc[mask, 'assembly_id'])


def _describe_holdout_spec(spec: dict) -> str:
    """One-line human-readable description of a filter spec; used for error
    messages and the dropped-isolates manifest's `excluded_reason` column."""
    if not spec:
        return '(no constraints)'
    parts = []
    for k, v in spec.items():
        parts.append(f"{k}={v}")
    return ', '.join(parts)


def compute_split_shares(split_sizes: dict) -> None:
    """In-place: add `pair_share` and `isolate_share` per split entry.

    Each share is the slot's count divided by the cross-split sum
    (train+val+test), as a float in [0, 1]. This is the right denominator
    for "what fraction of the dataset (post-filtering) is this split?"
    — metadata_holdout commonly produces splits that don't follow the
    configured train/val/test ratios, so this exposes the actual achieved
    split balance to downstream consumers.
    """
    slots = ('train', 'val', 'test')
    total_pairs = sum(split_sizes.get(s, {}).get('pairs', 0) for s in slots)
    total_iso = sum(split_sizes.get(s, {}).get('isolates', 0) for s in slots)
    for s in slots:
        if s not in split_sizes:
            continue
        p = split_sizes[s].get('pairs', 0)
        i = split_sizes[s].get('isolates', 0)
        split_sizes[s]['pair_share'] = round(p / total_pairs, 4) if total_pairs > 0 else 0.0
        split_sizes[s]['isolate_share'] = round(i / total_iso, 4) if total_iso > 0 else 0.0


def format_split_summary_banner(
    split_sizes: dict,
    holdout_cfg: Optional[dict] = None,
    ) -> str:
    """Return a one-line summary string for end-of-Stage-3 stdout.

    Format example::

        train=8,814 (45.7%) [hn_subtype=['H3N2']], val=981 (5.1%) [carved from train], test=9,503 (49.2%) [hn_subtype=['H1N1']]

    Driver text per slot:
      - random split (no holdout_cfg): ``random``
      - holdout train/test/val (explicit filter): ``_describe_holdout_spec`` output
      - holdout val implicit (cfg.val is None): ``carved from train``

    Requires `compute_split_shares` to have already run so the share fields
    are present; falls back to recomputing on-the-fly if missing.
    """
    slots = ('train', 'val', 'test')
    # Recompute shares if absent (defensive — callers should call compute_split_shares first).
    have_shares = all(
        'isolate_share' in split_sizes.get(s, {}) for s in slots if s in split_sizes
    )
    if not have_shares:
        compute_split_shares(split_sizes)

    parts = []
    for slot in slots:
        if slot not in split_sizes:
            continue
        info = split_sizes[slot]
        n_iso = info.get('isolates', 0)
        share = info.get('isolate_share', 0.0)
        if holdout_cfg is None:
            driver = 'random'
        else:
            slot_spec = holdout_cfg.get(slot)
            if slot == 'val' and slot_spec is None:
                driver = 'carved from train'
            elif slot_spec is None:
                driver = '(unconstrained)'
            else:
                driver = _describe_holdout_spec(slot_spec)
        parts.append(f"{slot}={n_iso:,} ({share:.1%}) [{driver}]")
    return ', '.join(parts)


def compute_metadata_holdout_isolates(
    df: pd.DataFrame,
    holdout_cfg: dict,
    seed: int,
    val_ratio: float,
    ) -> tuple:
    """Compute (train, val, test) isolate-id lists for cross-population holdout.

    See `docs/plans/2026-05-11_metadata_holdout_plan.md` for the design.
    holdout_cfg has shape ``{'train': {filter dict}, 'test': {filter dict},
    'val': null | {filter dict}}``. Each filter dict carries axis-keyed
    constraints (host / hn_subtype / year / year_range / geo_location /
    passage); scalars and lists are both supported (see filter_by_metadata
    for value-level rules). When ``val`` is null, val is carved off the
    train pool at ``val_ratio`` by deterministic random partition under
    ``seed``.

    Returns ``(train_ids, val_ids, test_ids, dropped_df)``:
      - The three id lists are sorted, disjoint, and pairwise non-empty.
        Empty pools raise.
      - ``dropped_df`` has one row per isolate that matched no slot, with
        identifier + metadata columns + ``excluded_reason`` (which filters
        it failed to satisfy).

    Raises ``ValueError`` on: missing required slot; multi-slot overlap;
    empty pool after filtering; val carve too small for val_ratio.
    """
    # Structural validation: train/test required, val optional.
    if 'train' not in holdout_cfg:
        raise ValueError("metadata_holdout.train is required.")
    if 'test' not in holdout_cfg:
        raise ValueError("metadata_holdout.test is required.")
    train_spec = _validate_holdout_filter_spec(holdout_cfg['train'], 'train')
    test_spec = _validate_holdout_filter_spec(holdout_cfg['test'], 'test')
    val_spec_raw = holdout_cfg.get('val')
    explicit_val = val_spec_raw is not None
    val_spec = _validate_holdout_filter_spec(val_spec_raw, 'val') if explicit_val else None

    # Per-isolate metadata frame (one row per assembly_id, first observation).
    meta_cols = ['assembly_id', 'hn_subtype', 'host', 'year']
    if 'geo_location_clean' in df.columns:
        meta_cols.append('geo_location_clean')
    if 'passage' in df.columns:
        meta_cols.append('passage')
    meta_per_iso = df.groupby('assembly_id', as_index=False)[meta_cols[1:]].first()

    # Apply filters per slot.
    train_set = _isolates_matching_holdout_spec(meta_per_iso, train_spec)
    test_set = _isolates_matching_holdout_spec(meta_per_iso, test_spec)
    val_set = (
        _isolates_matching_holdout_spec(meta_per_iso, val_spec) if explicit_val else set()
    )

    # Multi-slot overlap check. Each pair must be disjoint; report offenders.
    def _overlap_msg(a_name: str, a_set: set, b_name: str, b_set: set) -> str:
        common = sorted(a_set & b_set)
        if not common:
            return ''
        head = common[:20]
        more = f" (+ {len(common)-20:,} more)" if len(common) > 20 else ''
        return (f"\n  {a_name} and {b_name} share {len(common):,} isolate(s): "
                f"{head}{more}")

    overlap_msgs = []
    overlap_msgs.append(_overlap_msg('train', train_set, 'test', test_set))
    if explicit_val:
        overlap_msgs.append(_overlap_msg('train', train_set, 'val', val_set))
        overlap_msgs.append(_overlap_msg('val', val_set, 'test', test_set))
    overlap_msgs = [m for m in overlap_msgs if m]
    if overlap_msgs:
        raise ValueError(
            "metadata_holdout: train/val/test filters yield overlapping "
            "isolate sets -- the same isolate matches >1 slot, which would "
            "leak across splits. Disambiguate the filter specs."
            + ''.join(overlap_msgs)
        )

    # Val carving from train when val is implicit.
    val_source = 'explicit_filter'
    if not explicit_val:
        val_source = 'carved_from_train'
        if len(train_set) == 0:
            raise ValueError(
                f"metadata_holdout.train pool is empty under filter "
                f"({_describe_holdout_spec(train_spec)}); cannot carve val from it."
            )
        n_val = int(round(len(train_set) * val_ratio))
        if n_val < 1:
            raise ValueError(
                f"metadata_holdout: train pool has {len(train_set)} isolate(s); "
                f"val_ratio={val_ratio} would carve 0. Either lower val_ratio, "
                f"loosen the train filter, or pass an explicit val filter."
            )
        rng = np.random.RandomState(int(seed))
        train_sorted = sorted(train_set)
        idx = rng.permutation(len(train_sorted))
        val_set = set(train_sorted[i] for i in idx[:n_val])
        train_set = train_set - val_set

    # Empty-pool tripwires (after the carve).
    if len(train_set) == 0:
        raise ValueError(
            f"metadata_holdout.train pool is empty under filter "
            f"({_describe_holdout_spec(train_spec)})."
        )
    if len(val_set) == 0:
        raise ValueError(
            "metadata_holdout.val pool is empty " + (
                f"under filter ({_describe_holdout_spec(val_spec)})."
                if explicit_val else "after carving from train."
            )
        )
    if len(test_set) == 0:
        raise ValueError(
            f"metadata_holdout.test pool is empty under filter "
            f"({_describe_holdout_spec(test_spec)})."
        )

    # Dropped-isolates manifest. One row per isolate not in any slot, with
    # identifier + metadata columns + a short reason naming the offending
    # axis(es) per slot.
    assigned = train_set | val_set | test_set
    dropped_iso = meta_per_iso[~meta_per_iso['assembly_id'].isin(assigned)].copy()
    if len(dropped_iso) > 0:
        train_desc = _describe_holdout_spec(train_spec)
        test_desc = _describe_holdout_spec(test_spec)
        val_desc = (_describe_holdout_spec(val_spec) if explicit_val
                    else f'carved_from_train @ val_ratio={val_ratio}')
        # Annotate which slots this isolate failed to match (None means it
        # matched, str means it failed and the str names the slot's filter).
        dropped_iso['matches_train'] = False
        dropped_iso['matches_val'] = False
        dropped_iso['matches_test'] = False
        dropped_iso['excluded_reason'] = (
            f"matched no slot. train={{{train_desc}}}; "
            f"val={{{val_desc}}}; test={{{test_desc}}}."
        )
        if 'file' not in dropped_iso.columns:
            # Attach file column from df if available (one file per assembly_id
            # for Flu A; cheap groupby-first).
            if 'file' in df.columns:
                aid_to_file = df.groupby('assembly_id')['file'].first()
                dropped_iso = dropped_iso.merge(
                    aid_to_file.rename('file'), on='assembly_id', how='left'
                )

    train_ids = sorted(train_set)
    val_ids = sorted(val_set)
    test_ids = sorted(test_set)

    print(f"\nmetadata_holdout: computed isolate partitions "
          f"(train={len(train_ids):,}, val={len(val_ids):,} [{val_source}], "
          f"test={len(test_ids):,}, dropped={len(dropped_iso):,})")
    return train_ids, val_ids, test_ids, dropped_iso
