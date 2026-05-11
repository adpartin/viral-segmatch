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

import pandas as pd

# Project root is two levels up (src/datasets/_pair_helpers.py -> src/datasets -> src -> root)
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

from src.utils.path_utils import load_dataframe


def canonical_pair_key(seq_hash_a: str, seq_hash_b: str) -> str:
    """Create a canonical pair key from two sequence hashes.

    Ensures consistent ordering so (a, b) and (b, a) produce the same key.
    """
    return "__".join(sorted([seq_hash_a, seq_hash_b]))


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
    """Attach nucleotide sequence (`dna_seq`) and md5 hash (`dna_hash`) to each
    protein row via a join with `genome_final.*` in the same processed dir.

    Rationale: the existing pipeline dedups/checks leakage using `seq_hash` over
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

    genome_path = protein_input_path.parent / "genome_final"
    print(f"\nAttach DNA sequences from genome_final (sibling of {protein_input_path.name}).")
    genome_df = load_dataframe(genome_path)

    genome_cols = ["assembly_id", "genbank_ctg_id", "dna_seq"]
    miss_g = set(genome_cols) - set(genome_df.columns)
    if miss_g:
        raise ValueError(f"genome_final is missing required columns: {sorted(miss_g)}")
    genome_small = genome_df[genome_cols].copy()

    dup = genome_small.duplicated(["assembly_id", "genbank_ctg_id"]).sum()
    if dup:
        raise ValueError(
            f"genome_final has {dup} duplicate (assembly_id, genbank_ctg_id) rows. "
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
    # genome_final and protein_final came from the same preprocess_flu.py run; they
    # should match perfectly on (assembly_id, genbank_ctg_id).
    missing_dna = merged["dna_seq"].isna().sum()
    if missing_dna:
        raise RuntimeError(
            f"{missing_dna:,} protein rows have no matching dna_seq. "
            "Check that protein_final and genome_final came from the same "
            "preprocess_flu.py run."
        )

    merged["dna_hash"] = merged["dna_seq"].apply(
        lambda s: hashlib.md5(str(s).encode()).hexdigest()
    )

    n_unique_dna = merged["dna_hash"].nunique()
    print(f"  genome_df: Total rows: {len(genome_small):,};  unique dna_seq: {genome_small['dna_seq'].nunique():,}")
    print(f"  protein rows merged (with genome data):  {before:,} (100.0% matched)")
    print(f"  unique dna_hash:      {n_unique_dna:,} "
          f"({n_unique_dna / before * 100:.1f}% of protein rows -- many proteins share a contig)")
    return merged


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


def build_cooccurrence_set(df: pd.DataFrame) -> tuple[set, dict]:
    """Build a set of all sequence pairs that co-occur in any isolate.

    Two sequences "co-occur" if they appear together in the same isolate (same
    assembly_id). Used to prevent contradictory labels: if (A, B) co-occur in
    isolate X (positive), they cannot also be sampled as a negative.

    Returns:
        - cooccur_pairs: Set of canonical pair keys (`canonical_pair_key`
          format) for all sequence pairs that co-occur in at least one isolate.
        - cooccur_stats: dict with `total_cooccur_pairs`,
          `max_isolates_per_pair`, `pairs_in_multiple_isolates`,
          `isolate_pair_counts` (full pair_key -> count mapping).
    """
    cooccur_pairs = set()
    isolate_pair_counts = {}
    for aid, grp in df.groupby('assembly_id'):
        if len(grp) < 2:
            continue

        # Get unique protein sequence hashes for this isolate
        seq_hashes = grp['seq_hash'].unique().tolist()

        for i in range(len(seq_hashes)):
            for j in range(i + 1, len(seq_hashes)):
                seq_pair_key = canonical_pair_key(seq_hashes[i], seq_hashes[j])
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


def bipartite_components(pos_df: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Connected components of the bipartite (HA-DNA, NA-DNA) graph.

    Nodes: ('a', dna_hash_a) and ('b', dna_hash_b) values from `pos_df`.
    Edges: each unique `(dna_hash_a, dna_hash_b)` tuple. Computed by
    iterative-path-compression union-find (no networkx dependency).

    Returns:
        (component_id, summary)
        component_id: pd.Series aligned with pos_df.index, dtype int.
            Each row is labelled with its component's representative id
            (a contiguous integer 0..n_components-1).
        summary: dict with `n_components`, `n_pairs`, `n_dnas_a`,
            `n_dnas_b`, `largest_component_pairs`, `top_10_sizes`,
            `singleton_components` (count of size-1 components).

    The component label is stable under reordering of pos_df rows but
    not under reordering of (a, b) sides. v2 schema-mode positives are
    always (func_left, func_right) so this is fine.
    """
    if not {'dna_hash_a', 'dna_hash_b'}.issubset(pos_df.columns):
        raise ValueError(
            "bipartite_components: pos_df must contain dna_hash_a and dna_hash_b "
            "columns (added by attach_dna_to_prot_df upstream)."
        )

    # Union-find on string node ids ('a:'+hash, 'b:'+hash). String prefix
    # avoids the (rare) chance of an HA-DNA hash colliding with an NA-DNA
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

    edges = pos_df[['dna_hash_a', 'dna_hash_b']].drop_duplicates()
    for h in pos_df['dna_hash_a'].unique():
        parent[f'a:{h}'] = f'a:{h}'
    for h in pos_df['dna_hash_b'].unique():
        parent[f'b:{h}'] = f'b:{h}'
    for h_a, h_b in zip(edges['dna_hash_a'].values, edges['dna_hash_b'].values):
        union(f'a:{h_a}', f'b:{h_b}')

    # Map each row to its component's root, then to a contiguous int id.
    row_roots = [find(f'a:{h}') for h in pos_df['dna_hash_a'].values]
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
        'n_dnas_a': int(pos_df['dna_hash_a'].nunique()),
        'n_dnas_b': int(pos_df['dna_hash_b'].nunique()),
        'largest_component_pairs': int(sizes.iloc[0]) if len(sizes) else 0,
        'top_10_sizes': [int(s) for s in sizes.head(10).tolist()],
        'singleton_components': int((sizes == 1).sum()),
    }
    return component_id, summary


def seq_disjoint_route_pos_df(
    pos_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Route `pos_df` rows into train/val/test by bipartite-component LPT-greedy.

    Each connected component in the (HA-DNA, NA-DNA) bipartite graph is
    indivisible: the whole component lands in one split. A component's
    pairs always have both DNAs (slot a + slot b) present in that
    split's positives, so cross-split DNA leakage from positives is
    impossible by construction.

    LPT-greedy bin-packing: sort components by size desc (then by
    component-id for deterministic tie-breaking), assign each to the bin
    whose remaining-capacity deficit (target - current) is largest.
    With small components and three target ratios, this is within a
    rounding-rounding error of the targets in practice.

    `seed` is reserved for future tie-break shuffling but is not consumed
    in this implementation (the algorithm is fully deterministic without
    it). Accepted in the signature so the call site does not need a
    special-case.

    Returns:
        (train_pos, val_pos, test_pos, audit) -- the three DataFrames are
        row-disjoint partitions of pos_df preserving original column
        order; `audit` is a JSON-serializable dict (see code for keys).
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

    component_id, cc_summary = bipartite_components(pos_df)

    # Per-component pair counts, sorted (size desc, comp-id asc).
    sizes = component_id.value_counts().sort_index()  # comp-id asc
    sorted_comps = sorted(sizes.index, key=lambda c: (-int(sizes.loc[c]), int(c)))

    n_pairs = int(len(pos_df))
    targets = {
        'train': train_ratio * n_pairs,
        'val':   val_ratio * n_pairs,
        'test':  test_ratio * n_pairs,
    }
    bin_count = {'train': 0, 'val': 0, 'test': 0}
    comp_to_split: dict = {}

    for c in sorted_comps:
        s = int(sizes.loc[c])
        # Largest remaining-deficit bin wins. Ties broken in
        # train > val > test order for determinism (max() is stable).
        order = ['train', 'val', 'test']
        deficits = {k: targets[k] - bin_count[k] for k in order}
        winner = max(order, key=lambda k: deficits[k])
        comp_to_split[c] = winner
        bin_count[winner] += s

    split_for_row = component_id.map(comp_to_split)

    train_pos = pos_df[split_for_row == 'train'].reset_index(drop=True)
    val_pos = pos_df[split_for_row == 'val'].reset_index(drop=True)
    test_pos = pos_df[split_for_row == 'test'].reset_index(drop=True)

    achieved = {'train': len(train_pos), 'val': len(val_pos), 'test': len(test_pos)}

    # Cross-split DNA-hash overlap should be 0 by construction. Compute
    # anyway so the audit doubles as a regression check; the saver
    # raises if any overlap is non-zero.
    def _set(df: pd.DataFrame, side: str) -> set:
        return set(df[f'dna_hash_{side}'].dropna())
    overlaps: dict = {}
    for side in ('a', 'b'):
        sets = {sp: _set(d, side) for sp, d in
                (('train', train_pos), ('val', val_pos), ('test', test_pos))}
        overlaps[side] = {
            'train_val':   len(sets['train'] & sets['val']),
            'train_test':  len(sets['train'] & sets['test']),
            'val_test':    len(sets['val'] & sets['test']),
        }

    target_pcts = {k: 100.0 * v / n_pairs for k, v in targets.items()}
    achieved_pcts = {k: 100.0 * v / n_pairs for k, v in achieved.items()}
    audit = {
        'mode': 'seq_disjoint',
        'algorithm': 'bipartite_cc_lpt_greedy',
        'seed': int(seed),
        'cc_summary': cc_summary,
        'targets_pairs': {k: int(round(v)) for k, v in targets.items()},
        'targets_pct':   {k: round(v, 4) for k, v in target_pcts.items()},
        'achieved_pairs': achieved,
        'achieved_pct':   {k: round(v, 4) for k, v in achieved_pcts.items()},
        'max_target_deviation_pct': round(
            max(abs(achieved_pcts[k] - target_pcts[k]) for k in achieved_pcts), 4
        ),
        'pairs_dropped': 0,  # CC-bin-packing never splits a component.
        'dna_hash_overlap': overlaps,
    }
    return train_pos, val_pos, test_pos, audit


def filter_by_metadata(
    prot_df: pd.DataFrame,
    hn_subtype: str = None,
    host: str = None,
    year=None,
    geo_location: str = None,
    passage: str = None,
    ) -> pd.DataFrame:
    """Filter protein records by isolate-level metadata criteria.

    Filtering is performed at the isolate level: isolates matching ALL specified
    criteria are identified, then ALL protein records from those isolates are
    kept. This ensures we don't lose proteins due to metadata merge issues
    (e.g., an isolate has host metadata on one protein but not another).
    """
    any_filter = any(f is not None for f in [hn_subtype, host, year, geo_location, passage])
    if not any_filter:
        return prot_df

    print('\nMetadata filtering enabled.')
    print(f'HN subtype filter: {hn_subtype}')
    print(f'Host filter: {host}')
    print(f'Year filter: {year}')
    print(f'Geographic location filter: {geo_location}')
    print(f'Passage filter: {passage}')

    meta_cols = ['assembly_id', 'hn_subtype', 'host', 'year']
    if 'geo_location_clean' in prot_df.columns:
        meta_cols.append('geo_location_clean')
    if 'passage' in prot_df.columns:
        meta_cols.append('passage')
    aid_meta = prot_df.groupby('assembly_id')[meta_cols].first().reset_index(drop=True)

    print(f"\n   Available metadata columns: {meta_cols}")
    if 'geo_location_clean' in aid_meta.columns:
        unique_locations = aid_meta['geo_location_clean'].dropna().unique()
        print(f"   Unique geo_location_clean values (first 20): {sorted(unique_locations)[:20]}")
        if geo_location:
            matching_locs = [loc for loc in unique_locations if geo_location.lower() in str(loc).lower()]
            print(f"   Locations matching '{geo_location}' (case-insensitive): {matching_locs[:10]}")
    else:
        print(f"   WARNING: geo_location_clean column NOT found in prot_df!")
        print(f"   Available columns with 'location' in name: {[c for c in prot_df.columns if 'location' in c.lower()]}")

    aid_mask = pd.Series([True] * len(aid_meta))
    if host is not None:
        before = aid_mask.sum()
        aid_mask = aid_mask & aid_meta['host'].isin([host])
        after = aid_mask.sum()
        print(f"   Host filter '{host}': {before:,} -> {after:,} isolates")

    if year is not None:
        try:
            year = int(year)
        except (ValueError, TypeError):
            pass
        before = aid_mask.sum()
        aid_mask = aid_mask & aid_meta['year'].isin([year])
        after = aid_mask.sum()
        print(f"   Year filter '{year}': {before:,} -> {after:,} isolates")

    if hn_subtype is not None:
        before = aid_mask.sum()
        aid_mask = aid_mask & aid_meta['hn_subtype'].isin([hn_subtype])
        after = aid_mask.sum()
        print(f"   HN subtype filter '{hn_subtype}': {before:,} -> {after:,} isolates")

    if geo_location is not None:
        before = aid_mask.sum()
        aid_mask = aid_mask & aid_meta['geo_location_clean'].isin([geo_location])
        after = aid_mask.sum()
        print(f"   Geographic location filter '{geo_location}': {before:,} -> {after:,} isolates")

    if passage is not None and 'passage' in aid_meta.columns:
        before = aid_mask.sum()
        aid_mask = aid_mask & aid_meta['passage'].isin([passage])
        after = aid_mask.sum()
        print(f"   Passage filter '{passage}': {before:,} -> {after:,} isolates")

    matching_isolates = aid_meta[aid_mask]['assembly_id'].tolist()
    n_before = len(prot_df)
    prot_df = prot_df[prot_df['assembly_id'].isin(matching_isolates)].reset_index(drop=True)
    n_after = len(prot_df)

    print(f"   Filtered to {len(matching_isolates):,} isolates matching criteria")
    print(f"   Protein records: {n_before:,} -> {n_after:,} ({100*n_after/n_before:.1f}%)")

    return prot_df
