"""One-off data migration to the nt_cds/nt_ctg + hash schema (refactor Phase A2).

Derives the new-schema Stage-1/1.5 data files from the existing ones using
value-preserving transforms (md5 + column renames) -- NO re-run from GTOs. Produces
exactly what the updated preprocess_flu.py / extract_cds_dna.py would write.

Non-destructive:
  - genome_final, cds_final are LEFT UNTOUCHED (new files written alongside).
  - protein_final is rewritten in place ADDITIVELY (gains prot_hash) via
    .new -> verify -> atomic swap; the original is kept as *.bak_pre_prot_hash.

Outputs under data/processed/flu/July_2025/:
  cds_dna_final.parquet        <- cds_final    (cds_dna->cds_dna_seq, seq_hash->prot_hash)
  ctg_dna_final.{csv,parquet}  <- genome_final (dna_seq->ctg_dna_seq, +ctg_dna_hash)
  protein_final.{csv,parquet}  += prot_hash    (original -> *.bak_pre_prot_hash.*)

Run:  python scripts/migrate_to_nt_schema.py
"""
from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PROJ = Path(__file__).resolve().parents[1]
BASE = PROJ / "data" / "processed" / "flu" / "July_2025"


def _md5col(s: pd.Series) -> pd.Series:
    return s.map(lambda x: hashlib.md5(str(x).encode()).hexdigest())


def make_cds_dna_final() -> None:
    """cds_final -> cds_dna_final (rename cds_dna->cds_dna_seq, seq_hash->prot_hash)."""
    src = BASE / "cds_final.parquet"
    out = BASE / "cds_dna_final.parquet"
    print(f"\n[cds_dna_final] reading {src.name} ...", flush=True)
    df = pd.read_parquet(src)
    df = df.rename(columns={"cds_dna": "cds_dna_seq", "seq_hash": "prot_hash"})
    print(f"  writing {out.name} ({len(df):,} rows); cols={list(df.columns)}", flush=True)
    df.to_parquet(out, index=False)
    del df
    gc.collect()
    print(f"  OK -> {out.name} (cds_final untouched)", flush=True)


def make_ctg_dna_final() -> None:
    """genome_final -> ctg_dna_final (rename dna_seq->ctg_dna_seq, +ctg_dna_hash)."""
    src = BASE / "genome_final.parquet"
    print(f"\n[ctg_dna_final] reading {src.name} ...", flush=True)
    df = pd.read_parquet(src)
    df = df.rename(columns={"dna_seq": "ctg_dna_seq"})
    df["ctg_dna_hash"] = _md5col(df["ctg_dna_seq"])
    print(f"  writing ctg_dna_final.parquet ({len(df):,} rows); cols={list(df.columns)}", flush=True)
    df.to_parquet(BASE / "ctg_dna_final.parquet", index=False)
    print(f"  writing ctg_dna_final.csv ...", flush=True)
    df.to_csv(BASE / "ctg_dna_final.csv", sep=",", index=False)
    del df
    gc.collect()
    print(f"  OK -> ctg_dna_final.{{parquet,csv}} (genome_final untouched)", flush=True)


def add_prot_hash_inplace(ext: str) -> None:
    """protein_final.<ext> += prot_hash, via .new -> verify -> atomic swap (keeps .bak)."""
    src = BASE / f"protein_final.{ext}"
    print(f"\n[protein_final.{ext}] reading {src.name} ...", flush=True)
    if ext == "parquet":
        df = pd.read_parquet(src)
    else:
        df = pd.read_csv(src, dtype={"assembly_id": str, "genbank_ctg_id": str})
    n0, cols0 = len(df), list(df.columns)
    if "prot_hash" in cols0:
        print(f"  already has prot_hash; skip", flush=True)
        del df
        gc.collect()
        return
    df["prot_hash"] = _md5col(df["prot_seq"])
    assert len(df) == n0, "row count changed"
    assert set(cols0).issubset(df.columns), "lost a column"
    new = BASE / f"protein_final.new.{ext}"
    print(f"  writing {new.name} ({len(df):,} rows, +prot_hash) ...", flush=True)
    if ext == "parquet":
        df.to_parquet(new, index=False)
    else:
        df.to_csv(new, sep=",", index=False)
    del df
    gc.collect()
    if ext == "parquet":  # cheap structural verify before swapping
        nn = pq.read_metadata(new).num_rows
        assert nn == n0, f"new parquet rowcount {nn} != {n0}"
        assert "prot_hash" in pq.read_schema(new).names, "prot_hash missing in new parquet"
    bak = BASE / f"protein_final.bak_pre_prot_hash.{ext}"
    src.rename(bak)
    new.rename(src)
    print(f"  OK: original -> {bak.name}; new in place (rows={n0})", flush=True)


if __name__ == "__main__":
    print("=== Stage-1/1.5 schema migration (nt_cds/nt_ctg + hashes) ===", flush=True)
    make_cds_dna_final()          # new file (non-destructive)
    make_ctg_dna_final()          # new files (non-destructive)
    add_prot_hash_inplace("parquet")
    add_prot_hash_inplace("csv")  # heaviest; last
    print("\nDone.", flush=True)
