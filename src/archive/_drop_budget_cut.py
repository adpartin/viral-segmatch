"""Drop-budget 2D-CD holdout cut -- retired 2026-08-05.

Recovered an 80/10/10 holdout on the 2D mega-CC: bisect the heaviest component, drop the
straddling pairs, stop once the kept components LPT bin-pack into 80/10/10 within
`drift_pp`, or raise once the next cut would exceed `max_drop_frac`.

Retired because it was the hardcoded-80/10/10 form of `_megacc_cut.fragment_to_targets`,
which takes arbitrary `targets`; because it served the 2D-CD holdout, superseded by the
K-fold builder `dataset_pairs_cc.py`; and because no bundle or config group ever declared
the `split_strategy.drop_budget` knob that reached it.

The drop-budget MECHANISM is not retired: `_megacc_cut.fragment_until` caps its cuts with
`max_drop_frac`, wired as `split_strategy.edge_cut.max_drop_frac`, and that is what the
production 2D-CD path uses.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd

from src.datasets._bigraph import build_pair_bigraph, edges_to_row_index
from src.datasets._megacc_cut import (
    _live_atom_count,
    _lpt_max_drift,
    fragment_largest_cc,
)


class DropBudgetExceeded(RuntimeError):
    """Reaching an 80/10/10-feasible split would drop more than `max_drop_frac` of pairs."""


def apply_drop_budget_cut(
    pos_with_ids: pd.DataFrame,
    *,
    col_a: str = 'cluster_id_a',
    col_b: str = 'cluster_id_b',
    pair_key_col: str = 'pair_key',
    cut_method: str = 'spectral',
    target_frac: float = 0.80,
    drift_pp: float = 0.05,
    max_drop_frac: float = 0.20,
    seed: int = 1,
    max_cuts: int = 1000,
    ) -> tuple[pd.DataFrame, dict]:
    """Shrink the 2D mega-CC by edge min-cut so the kept components fit an 80/10/10 split.

    2D-CD routes each whole connected component (an atom) to one split, so a single very
    large component (the mega-CC) makes an 80/10/10 split impossible. This builds the
    pair-weighted simple bigraph and repeatedly bisects the largest component
    (`cut_method`), dropping each cut's straddling pairs, until the kept components LPT
    bin-pack into the 80/10/10 targets within `drift_pp` -- or it raises `DropBudgetExceeded`
    once dropping would exceed `max_drop_frac`. Clusters (the nodes) are never split, so the
    cost is counted in pairs.

    Stops on holdout feasibility; `fragment_until` is the sibling that stops on an atom count.
    Neither assigns pairs to splits -- they only shrink the graph, and the caller does the
    routing afterwards (`route_holdout` for a holdout, `groupkfold_by_atom` for K-fold). That
    is why both live here with the cut primitives rather than with the routers.

    Args:
        pos_with_ids: positive-pair rows with `col_a`/`col_b` cluster ids (+ `pair_key_col`);
            the pairs being routed (its index identifies each pair).
        col_a / col_b: slot-a / slot-b cluster-id column names.
        pair_key_col: per-pair key column; dropped keys are recorded in the audit.
        cut_method: how to bisect the largest CC -- `'spectral'` (Fiedler vector),
            `'kl'` (Kernighan-Lin), or `'none'` (no cut; return `pos_with_ids` unchanged).
        target_frac: nominal train fraction (0.80); audit-only -- the LPT bin targets are
            the module `_TARGETS` (80/10/10).
        drift_pp: stop once the LPT pack's worst-bin deviation from target is <= this
            (a fraction; 0.05 = 5 percentage points).
        max_drop_frac: cap on the dropped-pair fraction. Checked against what the NEXT cut
            would reach, so the cap holds on return: a cut that would break it is never
            applied, and `DropBudgetExceeded` is raised instead.
        seed: RNG seed for the seeded spectral / KL bisection.
        max_cuts: safety cap on the number of bisection iterations.

    Returns:
        `(kept_pos, audit)`: `kept_pos` is `pos_with_ids` minus the dropped straddling pairs
        (the caller re-derives components on it); `audit` holds the per-cut accounting and the
        dropped pair_keys, keyed like `fragment_until`'s where the two overlap
        (`cut_method`, `seed`, `n_cuts`, `n_atoms`, `pairs_dropped`, `dropped_frac`,
        `max_drop_frac`, `per_cut`).

    Raises:
        DropBudgetExceeded: if reaching 80/10/10 feasibility would need dropping more than
            `max_drop_frac` of pairs (the message lists the config knobs to relax).
    """
    n_total = int(len(pos_with_ids))
    if cut_method == 'none':
        return pos_with_ids, {'cut_method': 'none', 'pairs_dropped': 0,
                              'dropped_frac': 0.0, 'n_cuts': 0, 'per_cut': []}

    H, edge_rows = build_pair_bigraph(pos_with_ids, col_a=col_a, col_b=col_b)

    cross_edges: list[tuple] = []   # straddling edges dropped so far, in cut order
    pairs_dropped = 0               # straddling pairs dropped so far (edge weight)
    per_cut: list[dict] = []
    cut = 0

    while True:
        comps = list(nx.connected_components(H))
        cc_sizes = [int(H.subgraph(c).size(weight='weight')) for c in comps] # per-CC pair count
        retained = n_total - pairs_dropped
        largest = max(cc_sizes) if cc_sizes else 0
        drift = _lpt_max_drift(cc_sizes)
        per_cut.append({
            'cut': cut,
            'pairs_dropped': pairs_dropped,
            'dropped_frac': round(pairs_dropped / n_total, 6) if n_total else 0.0,
            'n_pieces': len(comps),  # raw CC count (may include a node stranded by a cut); cf. live n_atoms
            'largest_frac_of_retained': round(largest / retained, 6) if retained else 0.0,
            'lpt_drift': round(drift, 6),
        })
        if drift <= drift_pp:
            break
        # Empty input: nothing to cut. Checked after the row is recorded, so the audit below
        # still has a first and last entry.
        if not comps:
            break

        # Cost the next cut before applying it, so the budget caps what is dropped.
        step = fragment_largest_cc(H, cut_method=cut_method, seed=seed)
        would_drop_frac = (pairs_dropped + step.pairs_dropped) / n_total if n_total else 0.0
        if would_drop_frac > max_drop_frac or cut >= max_cuts:
            largest_frac = largest / retained if retained else 0.0   # guard: raising must not raise
            raise DropBudgetExceeded(
                f"drop-budget 2D-CD: recovering 80/10/10 needs dropping "
                f">{max_drop_frac:.0%} of pairs (would reach {would_drop_frac:.1%} at cut "
                f"{cut + 1}; largest CC still {largest_frac:.1%} of retained). "
                f"Options (require an explicit config change):\n"
                f"  - raise cluster_id_threshold (looser cut, smaller mega-CC),\n"
                f"  - raise split_strategy.drop_budget.max_drop_frac to accept the loss,\n"
                f"  - or use single_slot 1D-CD for this pair (no pairs dropped)."
            )
        cross_edges.extend(step.cross_edges)
        pairs_dropped += step.pairs_dropped
        # drop the crossing edges -> the largest CC splits into >=2 CCs (the 2 bisection
        # sides; a side splits further if its internal links ran through the other side).
        # The next loop's connected_components picks up the new pieces.
        H.remove_edges_from(step.cross_edges)
        cut += 1

    drop_idx = edges_to_row_index(cross_edges, edge_rows)
    kept_pos = pos_with_ids.drop(index=drop_idx)
    dropped_pair_keys = (
        pos_with_ids.loc[drop_idx, pair_key_col].tolist()
        if pair_key_col in pos_with_ids.columns else []
    )

    audit = {
        'cut_method': cut_method,
        'seed': seed,
        'target_frac': target_frac,
        'drift_pp': drift_pp,
        'max_drop_frac': max_drop_frac,
        'n_cuts': cut,
        'pairs_dropped': pairs_dropped,
        'dropped_frac': round(pairs_dropped / n_total, 6) if n_total else 0.0,
        'largest_cc_frac_before': per_cut[0]['largest_frac_of_retained'],
        'largest_cc_frac_after': per_cut[-1]['largest_frac_of_retained'],
        'lpt_drift_after': per_cut[-1]['lpt_drift'],
        # Routable atoms, not raw components: a stranded node is absent from the kept rows.
        'n_atoms': _live_atom_count(H),
        'per_cut': per_cut,
        'dropped_pair_keys': dropped_pair_keys,
    }
    return kept_pos, audit
