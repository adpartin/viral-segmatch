"""Check whether short NS1 CDS continue in their source contigs.

This is a positional check for the Human-H3N2-2024 protein alignment pilot in
docs/plans/2026-09-14_codon_site_features_plan.md. It compares the annotated
CDS and the bases after its stop with the most frequent 693-nt CDS. This does
not turn post-stop contig bases into CDS features.

Example:
    python -m src.analysis.audit_ns1_downstream_contig
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.cds_utils import extract_cds_dna, parse_location  # noqa: E402
from src.utils.experiment_utils import file_sha256, get_git_info  # noqa: E402

DEFAULT_DIR = PROJ / 'results/flu/July_2025/cds_alignment_pilot/Human-H3N2-2024/NS1'


def audit(selected: pd.DataFrame, protein: pd.DataFrame,
          contigs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compare each short NS1 CDS and its downstream contig with a long reference.

    Args:
        selected: Complete pilot CDS records, one per isolate.
        protein: Source annotations for the selected isolates.
        contigs: Source contigs for the selected isolates.

    Returns:
        Per-isolate comparison table and reference information.
    """
    if selected.empty or selected.function.nunique() != 1:
        raise ValueError('Expected one nonempty NS1 population in selected CDS.')
    if not (selected.host.eq('Human').all() and selected.hn_subtype.eq('H3N2').all()
            and selected.year.eq(2024).all() and selected.is_complete_cds.eq(True).all()):
        raise ValueError('Selected CDS are not all complete Human-H3N2-2024 records.')
    if selected.duplicated('assembly_id').any():
        raise ValueError('Selected NS1 has more than one row per isolate.')
    long = selected.loc[selected.cds_length == 693]
    if long.empty:
        raise ValueError('No 693-nt reference CDS in the selected population.')
    frequency = (long.groupby(['cds_dna_hash', 'cds_dna_seq']).size()
                 .reset_index(name='isolate_count')
                 .sort_values(['isolate_count', 'cds_dna_hash'], ascending=[False, True]))
    reference = frequency.iloc[0]
    reference_seq = reference.cds_dna_seq.lower()

    short = selected.loc[selected.cds_length < 693].copy()
    if short.empty:
        raise ValueError('No shorter CDS to audit.')
    if not short.cds_length.isin([654, 660]).all():
        raise ValueError('This pilot only covers the observed 654- and 660-nt short forms.')
    if protein.duplicated(['assembly_id', 'function']).any():
        raise ValueError('Protein annotation has duplicate isolate/function rows.')
    if contigs.duplicated(['assembly_id', 'genbank_ctg_id']).any():
        raise ValueError('Contig table has duplicate isolate/contig rows.')
    joined = short.merge(protein, on=['assembly_id', 'function'], validate='one_to_one')
    joined = joined.merge(contigs, on=['assembly_id', 'genbank_ctg_id'], validate='one_to_one')
    if len(joined) != len(short):
        raise ValueError('Could not link every short CDS to its annotation and contig.')

    rows = []
    for row in joined.itertuples(index=False):
        exons = parse_location(row.location)
        if len(exons) != 1 or exons[0][2] != '+':
            raise ValueError(f'{row.assembly_id}: expected one positive-strand NS1 exon.')
        contig_id, start_1based, _, annotated_length = exons[0]
        if contig_id != row.genbank_ctg_id or annotated_length != row.cds_length:
            raise ValueError(f'{row.assembly_id}: CDS annotation and selected length differ.')
        contig = row.ctg_dna_seq.lower()
        cds = row.cds_dna_seq.lower()
        if extract_cds_dna(contig, row.location).lower() != cds:
            raise ValueError(f'{row.assembly_id}: source contig does not recover the CDS.')
        end = start_1based - 1 + row.cds_length
        downstream = contig[end:]
        comparison_length = 693 - row.cds_length
        if len(downstream) < comparison_length:
            raise ValueError(f'{row.assembly_id}: contig ends before comparison window.')
        reference_tail = reference_seq[row.cds_length:693]
        observed_tail = downstream[:comparison_length]
        matches = sum(a == b for a, b in zip(observed_tail, reference_tail))
        stop_codon_number = row.cds_length // 3
        reference_at_stop = reference_seq[row.cds_length - 3:row.cds_length]
        rows.append({
            'assembly_id': row.assembly_id,
            'genbank_ctg_id': row.genbank_ctg_id,
            'cds_dna_hash': row.cds_dna_hash,
            'cds_length_nt': int(row.cds_length),
            'stop_codon_number': stop_codon_number,
            'short_stop_codon': cds[-3:],
            'reference_codon_at_stop': reference_at_stop,
            'downstream_available_nt': len(downstream),
            'downstream_compared_nt': comparison_length,
            'downstream_match_nt': matches,
            'downstream_reference_terminal_codon': observed_tail[-3:],
            'reference_terminal_codon': reference_tail[-3:],
        })
    result = pd.DataFrame(rows).sort_values(['cds_length_nt', 'assembly_id']).reset_index(drop=True)
    ref_info = {
        'reference_cds_dna_hash': reference.cds_dna_hash,
        'reference_length_nt': 693,
        'reference_isolates': int(reference.isolate_count),
        'reference_selection': 'most frequent 693-nt CDS by isolate count; hash breaks ties',
    }
    return result, ref_info


def main() -> None:
    """Write the short NS1 downstream comparison and provenance.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--pilot_dir', type=Path, default=DEFAULT_DIR)
    parser.add_argument('--protein_final', type=Path,
                        default=PROJ / 'data/processed/flu/July_2025/protein_final.parquet')
    parser.add_argument('--contig_final', type=Path,
                        default=PROJ / 'data/processed/flu/July_2025/ctg_dna_final.parquet')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    output_names = ('ns1_downstream_contig.csv', 'ns1_downstream_summary.csv',
                    'ns1_downstream_manifest.json')
    existing = [name for name in output_names if (args.pilot_dir / name).exists()]
    if existing and not args.overwrite:
        parser.error(f'Outputs already exist: {existing}. Pass --overwrite to replace them.')

    selected_path = args.pilot_dir / 'selected_cds.parquet'
    selected = pd.read_parquet(selected_path)
    ids = selected.loc[selected.cds_length < 693, 'assembly_id'].tolist()
    protein = pd.read_parquet(
        args.protein_final,
        columns=['assembly_id', 'function', 'genbank_ctg_id', 'location'],
        filters=[('assembly_id', 'in', ids)],
    )
    protein = protein.loc[protein.function == selected.function.iloc[0]].copy()
    contigs = pd.read_parquet(
        args.contig_final,
        columns=['assembly_id', 'genbank_ctg_id', 'ctg_dna_seq'],
        filters=[('assembly_id', 'in', ids)],
    )
    result, reference = audit(selected, protein, contigs)
    summary = (result.groupby('cds_length_nt', sort=True)
               .agg(isolates=('assembly_id', 'size'),
                    unique_cds=('cds_dna_hash', 'nunique'),
                    downstream_available_min_nt=('downstream_available_nt', 'min'),
                    downstream_available_max_nt=('downstream_available_nt', 'max'),
                    downstream_compared_nt=('downstream_compared_nt', 'first'),
                    match_min_nt=('downstream_match_nt', 'min'),
                    match_max_nt=('downstream_match_nt', 'max'))
               .reset_index())
    args.pilot_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.pilot_dir / output_names[0], index=False)
    summary.to_csv(args.pilot_dir / output_names[1], index=False)
    manifest = {
        'scope': 'complete Human-H3N2-2024 NS1 CDS shorter than 693 nt',
        'reference': reference,
        'script': str(Path(__file__).resolve()),
        'script_sha256': file_sha256(Path(__file__)),
        'inputs': {str(path): file_sha256(path) for path in
                   (selected_path, args.protein_final, args.contig_final)},
        'outputs': {name: file_sha256(args.pilot_dir / name) for name in output_names[:2]},
        'git': get_git_info(),
    }
    (args.pilot_dir / output_names[2]).write_text(json.dumps(manifest, indent=2) + '\n')
    print(summary.to_string(index=False))
    print(f'Wrote {len(result)} per-isolate comparisons to {args.pilot_dir}')


if __name__ == '__main__':
    main()
