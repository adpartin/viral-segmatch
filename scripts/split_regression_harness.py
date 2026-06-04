#!/usr/bin/env python
"""Split-router regression harness — P0 of the dataset-split refactor.

Captures a deterministic digest of the split router's output (the positive
`pair_key -> split` assignment, plus a full digest including negatives) for each
holdout guard-set bundle, so any refactor can be verified bit-exact.

  extract --dir RUN_DIR        # print the digest of an existing dataset run dir
  capture [--only NAME ...]    # build each guard bundle at fixed N and write its golden
  check   [--only NAME ...]    # rebuild and diff against goldens (exit != 0 on mismatch)

Scope: HOLDOUT-mode bit-exact guard only (plan 3.1). CV, single_slot=b, and
nt_cds-single-slot are fresh-validated separately (plan 5), NOT regression-guarded here.
Builds run at a small fixed isolate count (the split code path is scale-invariant, so a
path regression shows at any N) and land under results/ (gitignored); goldens are small
JSON under tests/golden/ (committed). See
docs/plans/2026-06-03_dataset_split_refactor_plan.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden" / "split_regression"
BUILD_ROOT = PROJECT_ROOT / "results" / "flu" / "July_2025" / "runs" / "split_regression" / "builds"
SPLITS = ("train", "val", "test")

# Holdout bit-exact guard set (plan 3.1, threshold-realigned). One entry per distinct
# split code path. CV / single_slot=b / nt_cds-single-slot are intentionally absent.
GUARD_SET = [
    {"name": "random_holdout",      "bundle": "flu_ha_na_random",                 "n": 2000,  "overrides": []},
    {"name": "seq_disjoint_seq",    "bundle": "flu_ha_na",                        "n": 2000,  "overrides": []},
    {"name": "seq_disjoint_seq_pb", "bundle": "flu_pb2_pb1",                      "n": 2000,  "overrides": []},
    {"name": "seq_disjoint_dna",    "bundle": "flu_ha_na",                        "n": 2000,  "overrides": ["dataset.split_strategy.hash_key=dna"]},
    {"name": "cluster_2d_aa",       "bundle": "flu_ha_na_cluster_t099",           "n": 2000,  "overrides": []},
    {"name": "cluster_2d_nt",       "bundle": "flu_ha_na_cluster_nt_t099",        "n": 2000,  "overrides": []},
    {"name": "cluster_1d_aa_slotA", "bundle": "flu_ha_na_cluster_aa_t095_HAonly", "n": 2000,  "overrides": []},
    # metadata_holdout filters isolates by axis, so it needs enough corpus to fill
    # train/val/test pools; a 2k subsample can starve a split. Tune N if the build errors.
    {"name": "metadata_holdout",    "bundle": "flu_ha_na_holdout_year",           "n": 20000, "overrides": []},
]


def _read_pairs(run_dir: Path, split: str):
    """DataFrame[pair_key, label] for one split; parquet preferred (no NaN-parse trap)."""
    import pandas as pd

    pq = run_dir / f"{split}_pairs.parquet"
    csv = run_dir / f"{split}_pairs.csv"
    if pq.exists():
        return pd.read_parquet(pq, columns=["pair_key", "label"])
    if csv.exists():
        return pd.read_csv(csv, usecols=["pair_key", "label"],
                           keep_default_na=False, na_values=[""], dtype={"pair_key": str})
    raise FileNotFoundError(f"no {split}_pairs.(parquet|csv) in {run_dir}")


def digest_run_dir(run_dir: Path) -> dict:
    """Deterministic digest of the split assignment in a dataset run dir.

    pos_digest  = sha256 over sorted (pair_key, split) for label==1 (pure router output).
    full_digest = sha256 over sorted (pair_key, label, split) for all rows (incl negatives).
    """
    run_dir = Path(run_dir)
    pos_rows, full_rows, counts = [], [], {}
    for split in SPLITS:
        df = _read_pairs(run_dir, split)
        lab = df["label"].astype(int)
        pk = df["pair_key"].astype(str)
        counts[split] = {"total": int(len(df)), "pos": int((lab == 1).sum())}
        pos_rows.extend((k, split) for k in pk[lab == 1])
        full_rows.extend((k, int(l), split) for k, l in zip(pk, lab))
    pos_rows.sort()
    full_rows.sort()

    def _h(rows):
        return hashlib.sha256(repr(rows).encode()).hexdigest()

    n_pos = sum(c["pos"] for c in counts.values())
    n_tot = sum(c["total"] for c in counts.values())
    return {
        "pos_digest": _h(pos_rows),
        "full_digest": _h(full_rows),
        "counts": counts,
        "n_pos": n_pos,
        "n_total": n_tot,
        "pos_split_pct": {s: round(100 * counts[s]["pos"] / n_pos, 3) if n_pos else 0.0 for s in SPLITS},
    }


def build(bundle: str, n: int, overrides: list, out_dir: Path) -> Path:
    """Run Stage 3 for one bundle at fixed N into out_dir (same interpreter -> same env)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # viz off: plots don't affect the split tables and pull in embedding/k-mer deps.
    ov = [f"dataset.max_isolates_to_process={n}", "dataset.generate_visualizations=false", *overrides]
    cmd = [sys.executable, str(PROJECT_ROOT / "src/datasets/dataset_segment_pairs.py"),
           "--config_bundle", bundle, "--output_dir", str(out_dir), "--override", *ov]
    print(f"  $ {' '.join(cmd)}", flush=True)
    if subprocess.run(cmd, cwd=PROJECT_ROOT).returncode != 0:
        raise SystemExit(f"build FAILED for {bundle} (see Stage-3 output above)")
    return out_dir


def _selected(only):
    if not only:
        return GUARD_SET
    sel = [g for g in GUARD_SET if g["name"] in set(only)]
    missing = set(only) - {g["name"] for g in sel}
    if missing:
        raise SystemExit(f"unknown guard names: {sorted(missing)} "
                         f"(known: {[g['name'] for g in GUARD_SET]})")
    return sel


def cmd_extract(args):
    print(json.dumps(digest_run_dir(Path(args.dir)), indent=2))


def cmd_capture(args):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for g in _selected(args.only):
        print(f"[capture] {g['name']}  ({g['bundle']}, N={g['n']}, overrides={g['overrides']})", flush=True)
        d = digest_run_dir(build(g["bundle"], g["n"], g["overrides"], BUILD_ROOT / f"capture_{g['name']}"))
        rec = {"name": g["name"], "bundle": g["bundle"], "n_isolates": g["n"], "overrides": g["overrides"], **d}
        (GOLDEN_DIR / f"{g['name']}.json").write_text(json.dumps(rec, indent=2))
        print(f"  -> tests/golden/split_regression/{g['name']}.json  "
              f"pos_digest={d['pos_digest'][:12]}  n_pos={d['n_pos']:,}  split%={d['pos_split_pct']}", flush=True)


def cmd_check(args):
    fails = []
    for g in _selected(args.only):
        gp = GOLDEN_DIR / f"{g['name']}.json"
        if not gp.exists():
            print(f"[check] {g['name']}: NO GOLDEN — run `capture --only {g['name']}` first")
            fails.append(g["name"])
            continue
        golden = json.loads(gp.read_text())
        d = digest_run_dir(build(g["bundle"], g["n"], g["overrides"], BUILD_ROOT / f"check_{g['name']}"))
        ok = d["pos_digest"] == golden["pos_digest"] and d["full_digest"] == golden["full_digest"]
        print(f"[check] {g['name']}: {'PASS' if ok else 'FAIL'}  "
              f"pos {d['pos_digest'][:12]} vs {golden['pos_digest'][:12]}  "
              f"full {'=' if d['full_digest'] == golden['full_digest'] else '!='}", flush=True)
        if not ok:
            fails.append(g["name"])
    if fails:
        raise SystemExit(f"\nREGRESSION: {len(fails)} mismatch(es): {fails}")
    print("\nAll selected guard bundles are bit-exact vs goldens.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    pe = sub.add_parser("extract", help="digest an existing run dir")
    pe.add_argument("--dir", required=True)
    pe.set_defaults(fn=cmd_extract)
    pc = sub.add_parser("capture", help="build guard bundles and write goldens")
    pc.add_argument("--only", nargs="+")
    pc.set_defaults(fn=cmd_capture)
    pk = sub.add_parser("check", help="rebuild and diff vs goldens")
    pk.add_argument("--only", nargs="+")
    pk.set_defaults(fn=cmd_check)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
