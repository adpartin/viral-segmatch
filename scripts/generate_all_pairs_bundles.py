#!/usr/bin/env python3
"""Generate 28 child bundle YAML files for all C(8,2) protein-pair combinations.

Each child inherits from flu_28_major_protein_pairs_master and only sets dataset.schema_pair.
Run once. Re-run is safe (overwrites existing files).

Usage:
    python scripts/generate_all_pairs_bundles.py
    python scripts/generate_all_pairs_bundles.py --dry_run
"""

import argparse
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLES_DIR = PROJECT_ROOT / "conf" / "bundles"
MASTER_BUNDLE = "flu_28_major_protein_pairs_master"

# 8 major proteins: one per segment, ordered S1-S8.
# Full function strings are defined in conf/virus/flu.yaml (selected_functions).
PROTEINS = [
    ("pb2", "RNA-dependent RNA polymerase PB2 subunit"),
    ("pb1", "RNA-dependent RNA polymerase catalytic core PB1 subunit"),
    ("pa",  "RNA-dependent RNA polymerase PA subunit"),
    ("ha",  "Hemagglutinin precursor"),
    ("np",  "Nucleocapsid protein"),
    ("na",  "Neuraminidase protein"),
    ("m1",  "Matrix protein 1"),
    ("ns1", "Non-structural protein 1, interferon antagonist and host mRNA processing inhibitor"),
]

CHILD_TEMPLATE = """\
# STATUS: active -- {short_a}/{short_b} pair (Task 11: all protein-pair combinations)
defaults:
  - {master}
  - _self_

dataset:
  schema_pair:
    - "{func_a}"
    - "{func_b}"
"""


def main():
    parser = argparse.ArgumentParser(description="Generate 28 protein-pair child bundles")
    parser.add_argument("--dry_run", action="store_true", help="Print filenames but do not write")
    args = parser.parse_args()

    pairs = list(combinations(PROTEINS, 2))
    print(f"Generating {len(pairs)} child bundles from master: {MASTER_BUNDLE}")

    for (short_a, func_a), (short_b, func_b) in pairs:
        filename = f"flu_28p_{short_a}_{short_b}.yaml"
        filepath = BUNDLES_DIR / filename
        content = CHILD_TEMPLATE.format(
            short_a=short_a.upper(),
            short_b=short_b.upper(),
            master=MASTER_BUNDLE,
            func_a=func_a,
            func_b=func_b,
        )
        if args.dry_run:
            print(f"  [dry-run] {filename}")
        else:
            filepath.write_text(content)
            print(f"  wrote {filename}")

    print(f"Done. {len(pairs)} bundles in {BUNDLES_DIR}/")


if __name__ == "__main__":
    main()
