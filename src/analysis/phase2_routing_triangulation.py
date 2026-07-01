"""Phase 2 pair_key migration routing-triangulation check.

For each protein-pair (canonical prot_hash_a/b pair) found in BOTH a
pre-Phase-2 dataset and a post-Phase-2 dataset built from the same
bundle config, check whether it lands in the same split (train/val/test).

If >99% land in the same split: routing is biologically equivalent
between pre and post; only the cluster_id namespace changed (pre =
protein-hash-keyed, post = DNA-hash-keyed lookup) -- no routing bug.

If split assignment differs substantially: actual routing change.
Either an intentional downstream effect of pair-key universe inflation
(LPT-greedy bin-packer assigns CCs differently when per-CC pair counts
shift) or a codebase bug.

Used 2026-06-03 to triangulate the Section B before/after metric drops
in `docs/results/2026-06-03_phase2_postmigration_metrics.md`. Result:
~20-22% of common positive protein-pairs land in different splits
pre vs post, explaining the metric drops as routing-shift confounding
rather than codebase regression.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

PAIRS = {
    "HA-NA": {
        "pre": Path("data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_nt_id099_20260515_115012"),
        "post": Path("data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_nt_t099_20260602_231059"),
    },
    "PB2-PB1": {
        "pre": Path("data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_cluster_nt_id099_20260515_115014"),
        "post": Path("data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_cluster_nt_t099_20260602_233011"),
    },
}
SPLITS = ["train", "val", "test"]


def load_pairs(d: Path, split: str) -> pd.DataFrame:
    return pd.read_csv(d / f"{split}_pairs.csv",
                       keep_default_na=False, na_values=[""],
                       low_memory=False)


def protein_pair_key_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [f"{min(a, b)}__{max(a, b)}"
         for a, b in zip(df["prot_hash_a"], df["prot_hash_b"])],
        index=df.index,
    )


def load_protein_pair_split_map(d: Path, label: int) -> dict[str, str]:
    """For each protein-pair (canonicalized), return the split it appears in.
    If a protein-pair appears in multiple splits (post may collapse multiple
    DNA-pairs to same protein-pair, all in same split by construction), keep
    the first encountered split (and flag if not unique)."""
    mapping = {}
    duplicates = []
    for split in SPLITS:
        df = load_pairs(d, split)
        df = df[df["label"] == label]
        pp = protein_pair_key_series(df)
        for k in pp.values:
            if k in mapping and mapping[k] != split:
                duplicates.append((k, mapping[k], split))
            elif k not in mapping:
                mapping[k] = split
    return mapping, duplicates


def main():
    pd.options.display.width = 200
    pd.options.display.max_columns = None
    pd.options.display.float_format = "{:.4f}".format

    print("=" * 100)
    print("Namespace-independent routing check: same protein-pair → same split pre vs post?")
    print("=" * 100)

    rows = []
    for pair_name, dirs in PAIRS.items():
        for label, label_name in [(1, "positive"), (0, "negative")]:
            pre_map, pre_dups = load_protein_pair_split_map(dirs["pre"], label)
            post_map, post_dups = load_protein_pair_split_map(dirs["post"], label)

            common = set(pre_map) & set(post_map)
            same_split = sum(1 for k in common if pre_map[k] == post_map[k])
            only_pre = len(set(pre_map) - set(post_map))
            only_post = len(set(post_map) - set(pre_map))

            # cross-split breakdown for the common set
            cross = {}
            for k in common:
                cross[(pre_map[k], post_map[k])] = cross.get((pre_map[k], post_map[k]), 0) + 1

            rows.append({
                "pair": pair_name,
                "label": label_name,
                "pre_pairs": len(pre_map),
                "post_pairs_collapsed": len(post_map),
                "common": len(common),
                "same_split_on_common": same_split,
                "same_split_pct": (same_split / len(common) * 100) if common else float("nan"),
                "only_in_pre": only_pre,
                "only_in_post": only_post,
                "pre_dups_multi_split": len(pre_dups),
                "post_dups_multi_split": len(post_dups),
            })

            if common and same_split < len(common):
                print(f"\n  {pair_name} {label_name}: cross-split transitions (pre→post)")
                for (a, b), n in sorted(cross.items()):
                    marker = "  " if a == b else " *"
                    print(f"   {marker} {a:5} → {b:5}: {n:>7d} ({n / len(common) * 100:.2f}%)")

    print()
    print(pd.DataFrame(rows).to_string(index=False))

    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print("  same_split_pct >= 99: routing is biologically stable.")
    print("    Cluster atom namespace changed (pre uses protein hash, post uses DNA hash)")
    print("    but downstream routing decisions match — no bug.")
    print()
    print("  same_split_pct < 99: routing CHANGED. Could be:")
    print("    (#4) downstream effect of pair-count inflation in LPT bin-packing.")
    print("    (#3) a code bug.")
    print("    Distinguishing requires inspecting per-CC assignment + pair counts.")


if __name__ == "__main__":
    main()
