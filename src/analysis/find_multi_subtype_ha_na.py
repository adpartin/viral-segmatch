"""QC script: find HA/NA protein sequences associated with multiple H/N subtypes.

A single HA protein sequence (`seq_hash`) should map to exactly one H subtype
(e.g., H3) and a single NA protein sequence to exactly one N subtype, because
the HA/NA proteins *define* those subtype components. Internal proteins
(PB2, PB1, PA, NP, M1, NS1) routinely span multiple subtypes via reassortment;
HA and NA should not.

This script extracts the (small) set of HA and NA `seq_hash`es that violate
that expectation, summarizes the offending isolates, and traces them back to
the source GTO files for manual inspection.

Output: data/processed/flu/{data_version}/multi_subtype_ha_na_qc.csv
        (one row per offending seq_hash)
"""

import hashlib
from pathlib import Path

import pandas as pd

from src.utils.metadata_enrichment import enrich_prot_data_with_metadata


HA_FUNCTION = "Hemagglutinin precursor"
NA_FUNCTION = "Neuraminidase protein"


def find_multi_subtype_seqs(
    prot_df: pd.DataFrame,
    function_label: str,
    component: str,  # "h_part" or "n_part"
) -> pd.DataFrame:
    """Return one row per seq_hash whose isolates carry >1 distinct H or N component.

    Columns: seq_hash, function, n_isolates, n_distinct_components,
             distinct_components, distinct_hn_subtypes, assembly_ids, gto_files.
    """
    sub = prot_df[prot_df["function"].eq(function_label)].copy()
    if sub.empty:
        return pd.DataFrame()

    multi_mask = sub.groupby("seq_hash")[component].nunique(dropna=True).gt(1)
    multi_hashes = multi_mask[multi_mask].index
    if len(multi_hashes) == 0:
        return pd.DataFrame()

    offending = sub[sub["seq_hash"].isin(multi_hashes)]
    summary = offending.groupby("seq_hash").agg(
        function=("function", "first"),
        n_isolates=("assembly_id", "nunique"),
        n_distinct_components=(component, lambda s: s.dropna().nunique()),
        distinct_components=(
            component,
            lambda s: ",".join(sorted(s.dropna().unique().tolist())),
        ),
        distinct_hn_subtypes=(
            "hn_subtype",
            lambda s: ",".join(sorted(s.dropna().unique().tolist())),
        ),
        assembly_ids=(
            "assembly_id",
            lambda s: ";".join(sorted(s.dropna().unique().tolist())),
        ),
        gto_files=(
            "file",
            lambda s: ";".join(sorted(s.dropna().unique().tolist())),
        ),
    )
    return summary.reset_index()


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_version = "July_2025"
    parquet_path = (
        project_root / "data" / "processed" / "flu" / data_version / "protein_final.parquet"
    )
    out_path = parquet_path.parent / "multi_subtype_ha_na_qc.csv"

    print(f"Loading: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if "seq_hash" not in df.columns:
        print("Computing seq_hash (md5 of prot_seq)...")
        df["seq_hash"] = df["prot_seq"].apply(
            lambda x: hashlib.md5(str(x).encode()).hexdigest()
        )

    print("Enriching with hn_subtype metadata...")
    df = enrich_prot_data_with_metadata(df, project_root=project_root)

    df["h_part"] = df["hn_subtype"].str.extract(r"(H\d+)", expand=False)
    df["n_part"] = df["hn_subtype"].str.extract(r"(N\d+)", expand=False)

    ha = find_multi_subtype_seqs(df, HA_FUNCTION, "h_part")
    na = find_multi_subtype_seqs(df, NA_FUNCTION, "n_part")

    print(f"HA seqs with >1 H subtype: {len(ha)}")
    print(f"NA seqs with >1 N subtype: {len(na)}")

    out = pd.concat([ha, na], ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}  ({len(out)} rows)")

    if len(out) > 0:
        print("\nPreview:")
        print(
            out[
                [
                    "seq_hash",
                    "function",
                    "n_isolates",
                    "distinct_components",
                    "distinct_hn_subtypes",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
