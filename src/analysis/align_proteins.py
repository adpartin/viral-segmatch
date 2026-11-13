"""Prepare complete CDS proteins and align them with MAFFT.

The procedure follows docs/plans/2026-09-14_codon_site_features_plan.md.
FASTA headers use source prot_hash values, which hash prot_seq including its terminal stop.

`--length_share_above` keeps only the CDS lengths whose share of the unique complete CDS
is above the given fraction. It is step 2 of the retention rule in
docs/results/2026-11-09_protein_alignment.md.

`--gap_pattern_share_above`, given with `--length_share_above` once that run is inspected,
keeps only the CDS whose protein's gap pattern holds more than the given fraction of the
run's unique CDS, and prepares them for realignment. It reads the run's outputs, not the
corpus. These are steps 4 and 5 of the same retention rule.

Outputs under <out_dir>/<population>/<protein>/, under
<out_dir>/<population>/<protein>/length_share_above_<value>/ with `--length_share_above`,
or under its subdirectory gap_pattern_share_above_<value>/ with both:
    selected_cds.parquet      selected complete records, retaining isolate and CDS mappings
    length_summary.csv        isolate, unique-CDS and unique-protein counts for every CDS length
                              of the complete records, its share of their unique CDS and whether
                              it was kept; with `--gap_pattern_share_above`, of the kept records
    proteins_for_alignment.fasta  unique proteins without terminal stop markers
    x_summary.csv           isolate, unique-CDS and unique-protein X counts by CDS length
    selection_manifest.json   input paths, filters, checks and counts; with
                              `--gap_pattern_share_above`, also the parent's hashes and the
                              kept and dropped gap patterns
    aligned_proteins.fasta  MAFFT protein alignment, rows in guide-tree order
    mafft.log               MAFFT progress and selected strategy
    protein_input_hashes.csv  hash of each stop-free input protein
    protein_cds_map.csv     protein-to-CDS mapping with isolate counts
    alignment_manifest.json  tool version, command, strategy, runtime and file hashes
    alignment_validation.json  input recovery check for every aligned protein
    gap_locations.csv      gap positions and gap-pattern name for every aligned protein
    gap_pattern_shares.csv  unique proteins, unique CDS, isolates and share of unique CDS
                            for each gap pattern
    aligned_proteins_labelled.fasta  the alignment, labels starting with prot_hash
    aligned_proteins_jalview.fasta  the alignment, labels starting with the protein
    inspection_manifest.json  input and output hashes for the gap inspection

Example:
    python -m src.analysis.align_proteins --protein NS1 --host Human --hn_subtype H3N2 --year 2024
    python -m src.analysis.align_proteins --protein NS1 --host Human --hn_subtype H3N2 --year 2024 --align_only --mafft_bin /path/to/mafft
    python -m src.analysis.align_proteins --protein NS1 --host Human --hn_subtype H3N2 --year 2024 --validate_only
    python -m src.analysis.align_proteins --protein NS1 --host Human --hn_subtype H3N2 --year 2024 --inspect_only
    python -m src.analysis.align_proteins --protein PB2 --host Human --hn_subtype H3N2 --year 2024 --length_share_above 0.02
    python -m src.analysis.align_proteins --protein PB2 --host Human --hn_subtype H3N2 --year 2024 --length_share_above 0.02 --gap_pattern_share_above 0.02
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.cds_utils import strip_terminal_stop, translate_dna  # noqa: E402
from src.utils.config_hydra import load_function_metadata  # noqa: E402
from src.utils.experiment_utils import file_sha256, get_git_info  # noqa: E402
from src.utils.fasta_utils import read_fasta  # noqa: E402
from src.utils.metadata_enrichment import (  # noqa: E402
    attach_isolate_metadata,
    filter_by_metadata,
    population_label,
)
from src.utils.schema import sequence_md5  # noqa: E402
from src.utils.timer_utils import Timer  # noqa: E402

CDS_COLUMNS = [
    'assembly_id', 'function', 'canonical_segment', 'cds_dna_hash',
    'cds_dna_seq', 'cds_length', 'prot_hash', 'prot_seq', 'is_complete_cds',
]
OUTPUT_NAMES = (
    'selected_cds.parquet', 'length_summary.csv', 'proteins_for_alignment.fasta',
    'x_summary.csv', 'selection_manifest.json',
)
ALIGNMENT_OUTPUT_NAMES = (
    'aligned_proteins.fasta', 'mafft.log', 'protein_input_hashes.csv',
    'protein_cds_map.csv', 'alignment_manifest.json',
)
INSPECTION_OUTPUT_NAMES = (
    'gap_locations.csv', 'gap_pattern_shares.csv', 'aligned_proteins_labelled.fasta',
    'aligned_proteins_jalview.fasta', 'inspection_manifest.json',
)


def summarize_selected_lengths(selected: pd.DataFrame) -> pd.DataFrame:
    """Count complete input records by length without collapsing synonymous CDS.

    Isolates sharing a CDS remain in the input. Each sequence count uses its
    own key: CDS DNA hashes for unique CDS and protein strings for unique proteins.
    Protein strings retain the terminal stop marker at this selection step.

    Args:
        selected: Complete CDS records, one per isolate.

    Returns:
        Counts of isolates, unique CDS and unique proteins at each CDS length.
    """
    summary = selected.groupby('cds_length', sort=True).agg(
        isolates=('assembly_id', 'nunique'),
        unique_cds=('cds_dna_hash', 'nunique'),
        unique_proteins=('prot_seq', 'nunique')
    )
    summary = summary.reset_index()
    return summary


def keep_cds_lengths_above_share(complete: pd.DataFrame,
                                 share_above: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the CDS lengths whose share of the unique complete CDS is above `share_above`.

    A length's share is its unique CDS count divided by the unique CDS count over all
    lengths, so a CDS carried by many isolates counts once.

    Args:
        complete: Complete CDS records, one per isolate.
        share_above: The share a length must exceed to be kept, as a fraction. 0.0 keeps
            every length, because every length holds at least one unique CDS.

    Returns:
        The records at the kept lengths, and the `summarize_selected_lengths` table of all
        lengths with two more columns, `unique_cds_share` and `kept`.

    Raises:
        ValueError: A CDS hash has more than one length, or no length is above `share_above`.
    """
    lengths_per_cds = complete.groupby('cds_dna_hash')['cds_length'].nunique()
    if lengths_per_cds.gt(1).any():
        raise ValueError(f'CDS hash {lengths_per_cds.idxmax()} has more than one CDS length.')

    lengths = summarize_selected_lengths(complete)
    lengths['unique_cds_share'] = lengths['unique_cds'] / complete['cds_dna_hash'].nunique()
    lengths['kept'] = lengths['unique_cds_share'] > share_above
    if not lengths['kept'].any():
        raise ValueError(f'No CDS length holds more than {share_above:g} of the unique CDS.')

    kept_lengths = lengths.loc[lengths['kept'], 'cds_length']
    kept = complete.loc[complete['cds_length'].isin(kept_lengths)].reset_index(drop=True)
    return kept, lengths


def prepare_proteins(selected: pd.DataFrame) -> pd.DataFrame:
    """Validate each unique CDS and make protein sequences without terminal stops.

    Args:
        selected: Complete CDS records, including the original DNA and protein hashes.

    Returns:
        One row per protein hash, with its sequence for protein alignment.

    Raises:
        ValueError: A required field is absent or a CDS, protein, length or hash disagrees.
    """
    required = ['cds_dna_hash', 'cds_dna_seq', 'cds_length', 'prot_hash', 'prot_seq']
    if selected[required].isna().any().any():
        raise ValueError('Selected complete CDS contain a missing sequence, length or hash.')
    consistency = selected.groupby('cds_dna_hash')[required[1:]].nunique(dropna=False)
    inconsistent = consistency.index[consistency.ne(1).any(axis=1)]
    if len(inconsistent):
        raise ValueError(f'CDS hash {inconsistent[0]} has conflicting source records.')

    unique_cds = selected.drop_duplicates('cds_dna_hash')
    for row in unique_cds.itertuples(index=False):
        if len(row.cds_dna_seq) != row.cds_length:
            raise ValueError(f'CDS {row.cds_dna_hash}: stored length differs from DNA length.')
        if len(row.cds_dna_seq) != 3 * len(row.prot_seq):
            raise ValueError(f'CDS {row.cds_dna_hash}: DNA length is not 3 x protein length.')
        if not row.prot_seq.endswith('*') or row.prot_seq.count('*') != 1:
            raise ValueError(f'CDS {row.cds_dna_hash}: expected one terminal stop marker.')

        try:
            translated = translate_dna(row.cds_dna_seq)
        except ValueError as exc:
            raise ValueError(f'CDS {row.cds_dna_hash}: translation failed: {exc}') from exc

        if translated != row.prot_seq:
            raise ValueError(f'CDS {row.cds_dna_hash}: stored protein differs from translation.')
        expected_hash = sequence_md5(row.prot_seq)
        if row.prot_hash != expected_hash:
            raise ValueError(f'CDS {row.cds_dna_hash}: stored protein hash differs from sequence.')

    proteins = unique_cds[['prot_hash', 'prot_seq']].drop_duplicates('prot_hash').copy()
    if proteins['prot_hash'].nunique() != unique_cds['prot_seq'].nunique():
        raise ValueError('One protein hash names more than one protein sequence.')
    proteins['alignment_prot_seq'] = proteins['prot_seq'].map(strip_terminal_stop)
    proteins = proteins[['prot_hash', 'alignment_prot_seq']].sort_values('prot_hash')
    return proteins.reset_index(drop=True)


def summarize_unknown_residues(selected: pd.DataFrame) -> pd.DataFrame:
    """Count records containing X by CDS length.

    Args:
        selected: Validated complete CDS records, one per isolate.

    Returns:
        Isolate, unique-CDS and unique-protein counts containing X at each length.
    """
    with_x = selected.loc[selected['prot_seq'].str.contains('X', regex=False)]
    counts = (with_x.groupby('cds_length', sort=True)
              .agg(isolates_with_x=('assembly_id', 'nunique'),
                   unique_cds_with_x=('cds_dna_hash', 'nunique'),
                   unique_proteins_with_x=('prot_hash', 'nunique')))
    lengths = pd.Index(sorted(selected['cds_length'].unique()), name='cds_length')
    summary = counts.reindex(lengths, fill_value=0).reset_index()
    return summary


# MAFFT flags for each --mafft_strategy choice.
#   fft_ns_2  `--retree 2 --maxiterate 0`, which is what MAFFT runs with no options at all.
#             It gives FFT-NS-2 at any input size, so the strategy does not depend on how
#             many sequences a protein has.
#   auto      `--auto`, which lets MAFFT choose from the input size. A protein with more
#             sequences could get a different strategy.
# The manifest records the command and the strategy MAFFT reports, so the choice is
# checkable after the run.
MAFFT_STRATEGY_OPTIONS = {
    'auto': ['--auto'],
    'fft_ns_2': ['--retree', '2', '--maxiterate', '0'],
}


def align_prepared_proteins(out_dir: Path, mafft_bin: str, overwrite: bool,
                            strategy_name: str) -> None:
    """Align the prepared protein FASTA and record MAFFT provenance.

    Args:
        out_dir: Directory containing selection and protein-preparation outputs.
        mafft_bin: MAFFT executable path or command on PATH.
        overwrite: Whether existing alignment outputs may be replaced.
        strategy_name: Key of `MAFFT_STRATEGY_OPTIONS` giving the options to pass.

    Returns:
        None.

    Raises:
        FileNotFoundError: A required input or MAFFT executable is absent.
        ValueError: Prepared FASTA and selected CDS disagree or MAFFT reports no strategy.
        RuntimeError: MAFFT fails or produces no alignment.
    """
    selected_path = out_dir / 'selected_cds.parquet'
    fasta_path = out_dir / 'proteins_for_alignment.fasta'
    selection_manifest_path = out_dir / 'selection_manifest.json'

    for path in (selected_path, fasta_path, selection_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f'Required preparation output is missing: {path}')
    existing = [name for name in ALIGNMENT_OUTPUT_NAMES if (out_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Alignment outputs already exist in {out_dir}: {existing}; '
                              'pass --overwrite to replace them.')

    selected = pd.read_parquet(selected_path)
    selection_manifest = json.loads(selection_manifest_path.read_text())
    if selection_manifest.get('counts', {}).get('unique_complete_proteins') != selected['prot_hash'].nunique():
        raise ValueError('Selection manifest and selected CDS disagree on protein count.')
    expected = selected[['prot_hash', 'prot_seq']].drop_duplicates('prot_hash')
    if expected['prot_hash'].nunique() != selected['prot_seq'].nunique():
        raise ValueError('A protein hash names more than one selected protein sequence.')
    expected_sequences = dict(zip(expected['prot_hash'],
                                  expected['prot_seq'].map(strip_terminal_stop)))
    fasta_lines = fasta_path.read_text().splitlines()
    if len(fasta_lines) != 2 * len(expected_sequences):
        raise ValueError('Prepared FASTA does not have one two-line record per selected protein.')
    input_sequences = {}
    for header, sequence in zip(fasta_lines[0::2], fasta_lines[1::2]):
        if not header.startswith('>') or not sequence or header[1:] in input_sequences:
            raise ValueError('Prepared FASTA has an invalid or repeated protein record.')
        input_sequences[header[1:]] = sequence
    if input_sequences != expected_sequences:
        raise ValueError('Prepared FASTA sequences disagree with the selected CDS proteins.')

    resolved_bin = shutil.which(mafft_bin)
    if resolved_bin is None:
        raise FileNotFoundError(f'MAFFT executable not found: {mafft_bin}')
    resolved_bin = str(Path(resolved_bin).resolve())
    version_result = subprocess.run([resolved_bin, '--version'], capture_output=True, text=True,
                                    check=True)
    version = (version_result.stdout or version_result.stderr).strip()
    # --reorder writes rows in MAFFT's guide-tree order instead of input order. The MAFFT
    # wrapper applies it to the finished alignment, so it changes the row order and nothing
    # else; later steps look rows up by prot_hash.
    command = [resolved_bin, '--amino', *MAFFT_STRATEGY_OPTIONS[strategy_name],
               '--reorder', '--thread', '1', str(fasta_path.resolve())]
    # Timed here because MAFFT reports no timing of its own, and the file timestamps only
    # bound it, since selection and alignment run separately.
    mafft_timer = Timer()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    mafft_timer.stop_timer()
    if result.returncode != 0:
        raise RuntimeError(f'MAFFT failed with exit code {result.returncode}:\n{result.stderr}')
    if not result.stdout.startswith('>'):
        raise RuntimeError('MAFFT produced no FASTA alignment.')
    strategy_match = re.search(r'^\s*Strategy:\s*(.+?)\s*$', result.stderr, flags=re.MULTILINE)
    if strategy_match is None:
        raise ValueError('MAFFT log does not report the selected alignment strategy.')
    strategy = strategy_match.group(1)

    input_hashes = pd.DataFrame(
        [(protein_hash, hashlib.sha256(sequence.encode('utf-8')).hexdigest())
         for protein_hash, sequence in sorted(input_sequences.items())],
        columns=['prot_hash', 'alignment_seq_sha256'],
    )
    cds_map = (selected.groupby(['prot_hash', 'cds_dna_hash', 'cds_length'], sort=True)
               .agg(isolates=('assembly_id', 'nunique')).reset_index())
    aligned_path = out_dir / 'aligned_proteins.fasta'
    log_path = out_dir / 'mafft.log'
    aligned_path.write_text(result.stdout)
    log_path.write_text(result.stderr)
    input_hashes.to_csv(out_dir / 'protein_input_hashes.csv', index=False)
    cds_map.to_csv(out_dir / 'protein_cds_map.csv', index=False)

    manifest = {
        'stage': 'protein_alignment',
        'selection_manifest': str(selection_manifest_path.resolve()),
        'selected_cds_sha256': file_sha256(selected_path),
        'input_fasta': str(fasta_path.resolve()),
        'input_fasta_sha256': file_sha256(fasta_path),
        'input_sequence_hashes': 'protein_input_hashes.csv',
        'protein_cds_map': 'protein_cds_map.csv',
        'mafft_version': version,
        'mafft_command': shlex.join(command) + f' > {shlex.quote(str(aligned_path.resolve()))}',
        'mafft_strategy': strategy,
        'mafft_seconds': round(mafft_timer.elapsed, 1),
        'aligned_fasta_sha256': file_sha256(aligned_path),
        'mafft_log_sha256': file_sha256(log_path),
        'protein_count': len(input_sequences),
        'unique_cds_count': int(selected['cds_dna_hash'].nunique()),
        'code': get_git_info(),
        'outputs': list(ALIGNMENT_OUTPUT_NAMES),
    }
    alignment_manifest = out_dir / 'alignment_manifest.json'
    alignment_manifest.write_text(json.dumps(manifest, indent=2) + '\n')

    print(f'MAFFT {version}: aligned {len(input_sequences):,} proteins with {strategy}.')
    print(f'Wrote alignment outputs to {out_dir}')


def validate_alignment(out_dir: Path, overwrite: bool) -> None:
    """Check that every aligned protein recovers its exact stop-free input.

    Args:
        out_dir: Directory containing the prepared and aligned protein FASTA files.
        overwrite: Whether an existing validation report may be replaced.

    Returns:
        None.

    Raises:
        FileNotFoundError: An alignment input or output is missing.
        ValueError: File hashes, protein IDs, widths or recovered sequences disagree.
    """
    input_path = out_dir / 'proteins_for_alignment.fasta'
    aligned_path = out_dir / 'aligned_proteins.fasta'
    manifest_path = out_dir / 'alignment_manifest.json'
    report_path = out_dir / 'alignment_validation.json'

    for path in (input_path, aligned_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f'Required alignment file is missing: {path}')
    if report_path.exists() and not overwrite:
        raise FileExistsError(f'Validation report already exists: {report_path}; '
                              'pass --overwrite to replace it.')

    manifest = json.loads(manifest_path.read_text())
    for path, hash_key in ((input_path, 'input_fasta_sha256'),
                           (aligned_path, 'aligned_fasta_sha256')):
        if file_sha256(path) != manifest[hash_key]:
            raise ValueError(f'{path} does not match the alignment manifest hash.')

    inputs = read_fasta(input_path)
    aligned = read_fasta(aligned_path)
    if set(inputs) != set(aligned):
        missing = sorted(set(inputs) - set(aligned))
        extra = sorted(set(aligned) - set(inputs))
        raise ValueError(f'Alignment protein IDs differ from inputs: missing={missing[:3]}, '
                         f'extra={extra[:3]}.')

    if len(inputs) != manifest['protein_count']:
        raise ValueError('Protein count differs from the alignment manifest.')

    widths = {len(sequence) for sequence in aligned.values()}
    if len(widths) != 1:
        raise ValueError(f'Aligned protein widths differ: {sorted(widths)}.')

    for protein_hash, sequence in aligned.items():
        if sequence.replace('-', '') != inputs[protein_hash]:
            raise ValueError(f'Aligned protein {protein_hash} does not recover its input.')

    report = {
        'stage': 'alignment_validation',
        'input_fasta': str(input_path.resolve()),
        'input_fasta_sha256': manifest['input_fasta_sha256'],
        'aligned_fasta': str(aligned_path.resolve()),
        'aligned_fasta_sha256': manifest['aligned_fasta_sha256'],
        'proteins_checked': len(inputs),
        'proteins_recovered': len(inputs),
        'aligned_width': next(iter(widths)),
        'all_passed': True,
        'code': get_git_info(),
    }
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Validated {len(inputs):,} aligned proteins; every input sequence was recovered.')
    print(f'Wrote validation report to {report_path}')


def number_gap_patterns(gap_locations: pd.DataFrame, isolates: pd.Series) -> pd.Series:
    """Name each protein's gap pattern by rank, `gap_pat001` being held by the most proteins.

    Patterns are ranked by unique proteins, then by isolates, then by the
    `gap_columns` string, so patterns with equal counts still get a fixed order.
    The rank is zero-padded to three digits so the names sort as text in rank order.

    Args:
        gap_locations: One row per unique protein, with `prot_hash` and `gap_columns`.
        isolates: Isolate count per protein, indexed by `prot_hash`.

    Returns:
        The gap-pattern name of each row of `gap_locations`, indexed like it.
    """
    per_protein = gap_locations[['prot_hash', 'gap_columns']].copy()
    per_protein['isolates'] = per_protein['prot_hash'].map(isolates)
    if per_protein['isolates'].isna().any():
        raise ValueError('An aligned protein has no isolate count.')
    patterns = (per_protein.groupby('gap_columns')
                .agg(unique_proteins=('prot_hash', 'size'), isolates=('isolates', 'sum'))
                .reset_index()
                .sort_values(['unique_proteins', 'isolates', 'gap_columns'],
                             ascending=[False, False, True]))
    if len(patterns) > 999:
        raise ValueError(f'{len(patterns)} gap patterns do not fit a three-digit name.')
    patterns['gap_pattern'] = [f'gap_pat{rank:03d}' for rank in range(1, len(patterns) + 1)]
    names = per_protein['gap_columns'].map(patterns.set_index('gap_columns')['gap_pattern'])
    return names


def summarize_gap_patterns(gap_locations: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Count each gap pattern's unique proteins, unique CDS and isolates, and its share of the unique CDS.

    The share divides a pattern's unique CDS by all unique CDS in `selected`. With
    `--length_share_above`, those are the CDS at the kept lengths.

    Args:
        gap_locations: One row per aligned protein, with `prot_hash`, `gap_pattern`,
            `gap_columns`, `cds_length` and `amino_acids`.
        selected: The selected CDS records, with `prot_hash`, `cds_dna_hash` and `assembly_id`.

    Returns:
        One row per gap pattern, in name order, with its CDS length, aa residue count, gap
        columns, unique proteins, unique CDS, isolates and `unique_cds_share`.

    Raises:
        ValueError: A selected record has no aligned protein, or a gap pattern spans more than
            one CDS length.
    """
    pattern_columns = ['gap_pattern', 'cds_length', 'amino_acids', 'gap_columns']
    records = selected[['prot_hash', 'cds_dna_hash', 'assembly_id']].merge(
        gap_locations[['prot_hash', *pattern_columns]], on='prot_hash', how='left',
        validate='many_to_one')
    if records['gap_pattern'].isna().any():
        raise ValueError('A selected CDS record has no aligned protein.')

    patterns = (records.groupby(pattern_columns, sort=True)
                .agg(unique_proteins=('prot_hash', 'nunique'),
                     unique_cds=('cds_dna_hash', 'nunique'),
                     isolates=('assembly_id', 'nunique'))
                .reset_index())
    if not patterns['gap_pattern'].is_unique:
        raise ValueError('A gap pattern spans more than one CDS length.')
    patterns['unique_cds_share'] = patterns['unique_cds'] / selected['cds_dna_hash'].nunique()
    return patterns


def describe_proteins(proteins: pd.DataFrame) -> pd.Series:
    """Join each protein's gap pattern, CDS length, residue count and isolate count.

    Args:
        proteins: Rows with `gap_pattern`, `cds_length`, `amino_acids` and `isolates`.

    Returns:
        Strings such as `gap_pat001|2280nt|759aa|5046iso`, indexed like `proteins`.
    """
    fields = (proteins['gap_pattern']
              + '|' + proteins['cds_length'].astype(str) + 'nt'
              + '|' + proteins['amino_acids'].astype(str) + 'aa'
              + '|' + proteins['isolates'].astype(str) + 'iso')
    return fields


def build_hash_first_labels(proteins: pd.DataFrame, protein: str) -> pd.Series:
    """Build FASTA labels that start with `prot_hash`, so `label.split('|')[0]` recovers it.

    The labels hold no whitespace, so `read_fasta` keys each record by the whole label.

    Args:
        proteins: Rows with `prot_hash`, `gap_pattern`, `cds_length`, `amino_acids`
            and `isolates`.
        protein: Short protein name, e.g. PB2.

    Returns:
        Labels such as `<prot_hash>|PB2|gap_pat001|2280nt|759aa|5046iso`, indexed like
        `proteins`.
    """
    labels = proteins['prot_hash'] + '|' + protein + '|' + describe_proteins(proteins)
    return labels


def build_protein_first_labels(proteins: pd.DataFrame, protein: str) -> pd.Series:
    """Build FASTA labels that start with the protein, so sorting them groups the gap patterns.

    The labels hold no whitespace, so `read_fasta` keys each record by the whole label.

    Args:
        proteins: Rows with `prot_hash`, `gap_pattern`, `cds_length`, `amino_acids`
            and `isolates`.
        protein: Short protein name, e.g. PB2.

    Returns:
        Labels such as `PB2|gap_pat001|2280nt|759aa|5046iso|<prot_hash>`, indexed like
        `proteins`.
    """
    labels = protein + '|' + describe_proteins(proteins) + '|' + proteins['prot_hash']
    return labels


def inspect_alignment(out_dir: Path, protein: str, overwrite: bool) -> None:
    """Record gap locations for every validated aligned protein and write two labelled copies.

    Also writes each gap pattern's counts and its share of the unique CDS.

    Args:
        out_dir: Directory containing selected CDS and validated alignment outputs.
        protein: Short protein name, used in the labels.
        overwrite: Whether existing inspection outputs may be replaced.

    Returns:
        None.

    Raises:
        FileNotFoundError: A required selection, alignment or validation file is absent.
        ValueError: Input hashes or protein IDs disagree, the selected CDS changed after the
            alignment was made, or validation did not pass.
    """
    selected_path = out_dir / 'selected_cds.parquet'
    aligned_path = out_dir / 'aligned_proteins.fasta'
    alignment_manifest_path = out_dir / 'alignment_manifest.json'
    validation_path = out_dir / 'alignment_validation.json'
    for path in (selected_path, aligned_path, alignment_manifest_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(f'Required inspection input is missing: {path}')
    existing = [name for name in INSPECTION_OUTPUT_NAMES if (out_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Inspection outputs already exist in {out_dir}: {existing}; '
                              'pass --overwrite to replace them.')

    validation = json.loads(validation_path.read_text())
    if not validation.get('all_passed') or validation['aligned_fasta_sha256'] != file_sha256(aligned_path):
        raise ValueError('The alignment has no passing validation for its current FASTA.')
    # The protein IDs checked below can survive a change to the CDS records or their isolates,
    # so the selection itself must be the one the alignment was made from.
    alignment_manifest = json.loads(alignment_manifest_path.read_text())
    if alignment_manifest['selected_cds_sha256'] != file_sha256(selected_path):
        raise ValueError(f'{selected_path} changed after the alignment was made; rerun the '
                         'alignment and its validation.')
    selected = pd.read_parquet(selected_path,
                               columns=['prot_hash', 'cds_dna_hash', 'cds_length', 'assembly_id'])
    lengths_by_hash = selected.groupby('prot_hash')['cds_length'].nunique()
    if lengths_by_hash.ne(1).any():
        raise ValueError('A protein hash has more than one CDS length.')
    cds_lengths = selected.drop_duplicates('prot_hash').set_index('prot_hash')['cds_length']
    aligned = read_fasta(aligned_path)
    if set(aligned) != set(cds_lengths.index):
        raise ValueError('Aligned protein IDs disagree with selected CDS records.')

    rows = []
    for protein_hash, sequence in sorted(aligned.items()):
        leading = len(sequence) - len(sequence.lstrip('-'))
        trailing = len(sequence) - len(sequence.rstrip('-'))
        gap_count = sequence.count('-')
        gaps = ';'.join(f'{match.start() + 1}-{match.end()}'
                        for match in re.finditer(r'-+', sequence))
        unknowns = ';'.join(str(i + 1) for i, residue in enumerate(sequence) if residue == 'X')
        rows.append({
            'prot_hash': protein_hash,
            'cds_length': int(cds_lengths[protein_hash]),
            'alignment_width': len(sequence),
            'amino_acids': len(sequence) - gap_count,
            'gap_count': gap_count,
            'leading_gaps': leading,
            'internal_gaps': gap_count - leading - trailing,
            'trailing_gaps': trailing,
            'gap_columns': gaps,
            'x_columns': unknowns,
        })

    gap_locations = pd.DataFrame(rows)
    if gap_locations['alignment_width'].nunique() != 1:
        raise ValueError('Aligned proteins have different widths during inspection.')
    isolates = selected.groupby('prot_hash')['assembly_id'].nunique()
    gap_locations['gap_pattern'] = number_gap_patterns(gap_locations, isolates)
    gap_path = out_dir / 'gap_locations.csv'
    gap_locations.to_csv(gap_path, index=False)
    shares = summarize_gap_patterns(gap_locations, selected)
    shares_path = out_dir / 'gap_pattern_shares.csv'
    shares.to_csv(shares_path, index=False)

    # Two copies of aligned_proteins.fasta, same rows in the same order, with readable
    # labels: one starting with prot_hash, one starting with the protein so that sorting
    # by label in a viewer groups the rows by gap pattern. aligned_proteins.fasta itself
    # stays keyed by prot_hash for the later steps.
    proteins = gap_locations.set_index('prot_hash', drop=False)
    proteins['isolates'] = proteins['prot_hash'].map(isolates)
    labelled_path = out_dir / 'aligned_proteins_labelled.fasta'
    jalview_path = out_dir / 'aligned_proteins_jalview.fasta'
    for path, labels in ((labelled_path, build_hash_first_labels(proteins, protein)),
                         (jalview_path, build_protein_first_labels(proteins, protein))):
        records = [f'>{labels[protein_hash]}\n{sequence}'
                   for protein_hash, sequence in aligned.items()]
        path.write_text('\n'.join(records) + '\n')

    manifest = {
        'stage': 'alignment_gap_inspection',
        'selected_cds_sha256': file_sha256(selected_path),
        'aligned_fasta_sha256': file_sha256(aligned_path),
        'validation_report_sha256': file_sha256(validation_path),
        'gap_locations_sha256': file_sha256(gap_path),
        'gap_pattern_shares_sha256': file_sha256(shares_path),
        'labelled_fasta_sha256': file_sha256(labelled_path),
        'jalview_fasta_sha256': file_sha256(jalview_path),
        'proteins_inspected': len(gap_locations),
        'alignment_width': int(gap_locations['alignment_width'].iloc[0]),
        'code': get_git_info(),
        'outputs': list(INSPECTION_OUTPUT_NAMES),
    }
    inspec_manifest = out_dir / 'inspection_manifest.json'
    inspec_manifest.write_text(json.dumps(manifest, indent=2) + '\n')

    print(f'Inspected gap locations for {len(gap_locations):,} aligned proteins.')
    print(f'Wrote inspection outputs to {out_dir}')


def check_inspected_alignment(alignment_dir: Path) -> dict:
    """Check that an inspected alignment directory's files all describe its current alignment.

    The alignment must have passed validation. Every hash the alignment, validation and
    inspection manifests recorded must match the file it names, so a file rewritten after
    the step that recorded it is caught.

    Args:
        alignment_dir: A directory that `--inspect_only` has completed.

    Returns:
        The sha256 of each file checked, keyed by file name.

    Raises:
        FileNotFoundError: A required file is missing.
        ValueError: Validation did not pass, or a file does not match the hash a manifest
            recorded for it.
    """
    names = ('selection_manifest.json', 'selected_cds.parquet', 'aligned_proteins.fasta',
             'alignment_manifest.json', 'alignment_validation.json', 'inspection_manifest.json',
             'gap_locations.csv', 'gap_pattern_shares.csv')
    missing = [name for name in names if not (alignment_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f'{alignment_dir} is missing {missing}; run the selection, '
                                'alignment, validation and inspection first.')
    hashes = {name: file_sha256(alignment_dir / name) for name in names}

    validation = json.loads((alignment_dir / 'alignment_validation.json').read_text())
    if not validation.get('all_passed'):
        raise ValueError(f'The alignment in {alignment_dir} did not pass validation.')
    alignment = json.loads((alignment_dir / 'alignment_manifest.json').read_text())
    inspection = json.loads((alignment_dir / 'inspection_manifest.json').read_text())
    # Each row: the manifest, the hash it recorded, and the file that hash must describe.
    recorded = (
        ('alignment_validation.json', validation.get('aligned_fasta_sha256'), 'aligned_proteins.fasta'),
        ('alignment_manifest.json', alignment.get('aligned_fasta_sha256'), 'aligned_proteins.fasta'),
        ('alignment_manifest.json', alignment.get('selected_cds_sha256'), 'selected_cds.parquet'),
        ('inspection_manifest.json', inspection.get('aligned_fasta_sha256'), 'aligned_proteins.fasta'),
        ('inspection_manifest.json', inspection.get('selected_cds_sha256'), 'selected_cds.parquet'),
        ('inspection_manifest.json', inspection.get('validation_report_sha256'),
         'alignment_validation.json'),
        ('inspection_manifest.json', inspection.get('gap_locations_sha256'), 'gap_locations.csv'),
        ('inspection_manifest.json', inspection.get('gap_pattern_shares_sha256'),
         'gap_pattern_shares.csv'),
    )
    for manifest_name, recorded_hash, file_name in recorded:
        if recorded_hash != hashes[file_name]:
            raise ValueError(f'{alignment_dir}: {file_name} does not match the hash in '
                             f'{manifest_name}; rerun the steps from the one that wrote it.')
    return hashes


def keep_gap_patterns_above_share(selected: pd.DataFrame, gap_locations: pd.DataFrame,
                                  shares: pd.DataFrame,
                                  share_above: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the CDS records whose protein's gap pattern holds more than `share_above` of the unique CDS.

    Every record of a kept protein is kept, including its synonymous CDS. A pattern's share is
    the `unique_cds_share` that inspection wrote, so a pattern exactly at `share_above` is
    dropped.

    Args:
        selected: The CDS records of an inspected alignment, with `prot_hash`.
        gap_locations: Its `gap_locations.csv`, one row per aligned protein.
        shares: Its `gap_pattern_shares.csv`, one row per gap pattern.
        share_above: The share a gap pattern must exceed to be kept, as a fraction.

    Returns:
        The kept records, and `shares` with a `kept` column added.

    Raises:
        ValueError: A protein appears twice in `gap_locations`, a selected protein has no gap
            pattern, `gap_locations` and `shares` name different patterns, or no pattern is
            dropped or none is kept.
    """
    if not gap_locations['prot_hash'].is_unique:
        raise ValueError('A protein has more than one row in gap_locations.csv.')
    pattern_by_protein = gap_locations.set_index('prot_hash')['gap_pattern']
    without_pattern = sorted(set(selected['prot_hash']) - set(pattern_by_protein.index))
    if without_pattern:
        raise ValueError(f'{len(without_pattern)} selected proteins have no gap pattern, for '
                         f'example {without_pattern[0]}.')
    if set(pattern_by_protein) != set(shares['gap_pattern']) or not shares['gap_pattern'].is_unique:
        raise ValueError('gap_locations.csv and gap_pattern_shares.csv name different gap patterns.')

    shares = shares.copy()
    shares['kept'] = shares['unique_cds_share'] > share_above
    if shares['kept'].all():
        raise ValueError(f'Every gap pattern holds more than {share_above:g} of the unique CDS, '
                         'so no record is dropped and there is nothing to realign.')
    if not shares['kept'].any():
        raise ValueError(f'No gap pattern holds more than {share_above:g} of the unique CDS, '
                         'so no record would be kept.')

    kept_patterns = set(shares.loc[shares['kept'], 'gap_pattern'])
    is_kept = selected['prot_hash'].map(pattern_by_protein).isin(kept_patterns)
    kept = selected.loc[is_kept].reset_index(drop=True)
    return kept, shares


def count_records(records: pd.DataFrame) -> dict:
    """Count records, isolates, unique CDS and unique proteins.

    Args:
        records: CDS records with `assembly_id`, `cds_dna_hash` and `prot_hash`.

    Returns:
        The four counts as plain ints.
    """
    counts = {
        'records': len(records),
        'isolates': int(records['assembly_id'].nunique()),
        'unique_cds': int(records['cds_dna_hash'].nunique()),
        'unique_proteins': int(records['prot_hash'].nunique()),
    }
    return counts


def write_selection(out_dir: Path, selected: pd.DataFrame, summary: pd.DataFrame,
                    proteins: pd.DataFrame, x_summary: pd.DataFrame, manifest: dict,
                    overwrite: bool) -> None:
    """Write the five selection outputs that `--align_only` reads.

    Args:
        out_dir: Directory to write to; created if absent.
        selected: The selected CDS records.
        summary: Counts by CDS length, for `length_summary.csv`.
        proteins: Output of `prepare_proteins`, written as the MAFFT input FASTA.
        x_summary: Output of `summarize_unknown_residues`.
        manifest: Contents of `selection_manifest.json`.
        overwrite: Whether existing selection outputs may be replaced.

    Returns:
        None.

    Raises:
        FileExistsError: A selection output exists and `overwrite` is False.
    """
    existing = [name for name in OUTPUT_NAMES if (out_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Output files already exist in {out_dir}: {existing}; '
                              'pass --overwrite to replace them.')

    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(out_dir / 'selected_cds.parquet', index=False)
    summary.to_csv(out_dir / 'length_summary.csv', index=False)
    fasta = ''.join(f'>{row.prot_hash}\n{row.alignment_prot_seq}\n'
                    for row in proteins.itertuples(index=False))
    (out_dir / 'proteins_for_alignment.fasta').write_text(fasta)
    x_summary.to_csv(out_dir / 'x_summary.csv', index=False)
    (out_dir / 'selection_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def select_kept_gap_patterns(parent_dir: Path, out_dir: Path, share_above: float,
                             overwrite: bool) -> None:
    """Select the parent's records whose gap pattern holds more than `share_above` of its unique CDS.

    The records come from the parent's `selected_cds.parquet`, not the corpus, so they are the
    exact records the parent alignment was made from. The selection is written to `out_dir`
    for `--align_only` to realign. Nothing is written when no pattern is dropped or none is
    kept.

    Args:
        parent_dir: An inspected alignment directory, here the `--length_share_above` one.
        out_dir: Directory for the new selection.
        share_above: The share a gap pattern must exceed to be kept, as a fraction.
        overwrite: Whether existing selection outputs may be replaced.

    Returns:
        None.

    Raises:
        FileNotFoundError: A parent file is missing.
        ValueError: A parent file is out of date, the gap-pattern tables disagree, the parent
            selection has no length filter, or no pattern is dropped or none is kept.
    """
    parent_hashes = check_inspected_alignment(parent_dir)
    parent_selection = json.loads((parent_dir / 'selection_manifest.json').read_text())
    if 'length_filter' not in parent_selection:
        raise ValueError(f'{parent_dir / "selection_manifest.json"} has no length_filter; '
                         'select the parent with --length_share_above.')
    parent_records = pd.read_parquet(parent_dir / 'selected_cds.parquet')
    gap_locations = pd.read_csv(parent_dir / 'gap_locations.csv', keep_default_na=False)
    shares = pd.read_csv(parent_dir / 'gap_pattern_shares.csv', keep_default_na=False)

    selected, patterns = keep_gap_patterns_above_share(parent_records, gap_locations, shares,
                                                       share_above)
    selected = selected.sort_values(['cds_length', 'cds_dna_hash', 'assembly_id']).reset_index(drop=True)
    dropped = parent_records.loc[~parent_records['prot_hash'].isin(selected['prot_hash'])]

    # The length table describes the kept records, so its shares are out of their unique CDS.
    summary = summarize_selected_lengths(selected)
    summary['unique_cds_share'] = summary['unique_cds'] / selected['cds_dna_hash'].nunique()
    summary['kept'] = True
    proteins = prepare_proteins(selected)
    x_summary = summarize_unknown_residues(selected)

    # One plain-JSON record per parent gap pattern, split by whether it is kept.
    kept_rows, dropped_rows = [], []
    for row in patterns.itertuples(index=False):
        record = {
            'gap_pattern': row.gap_pattern,
            'cds_length': int(row.cds_length),
            'amino_acids': int(row.amino_acids),
            'gap_columns': row.gap_columns,
            'unique_proteins': int(row.unique_proteins),
            'unique_cds': int(row.unique_cds),
            'isolates': int(row.isolates),
            'unique_cds_share': float(row.unique_cds_share),
        }
        if row.kept:
            kept_rows.append(record)
        else:
            dropped_rows.append(record)

    parent_counts = parent_selection['counts']
    manifest = {
        'stage': 'protein_realignment_preparation',
        'protein': parent_selection['protein'],
        'function': parent_selection['function'],
        'population': parent_selection['population'],
        'inputs': parent_selection['inputs'],
        'filters': {**parent_selection['filters'], 'gap_pattern_share_above': share_above},
        # Copied unchanged: it records the parent's CDS-length decision, not this selection.
        'length_filter': parent_selection['length_filter'],
        'gap_pattern_filter': {
            'share_above': share_above,
            'parent_dir': str(parent_dir.resolve()),
            'parent_selection_manifest_sha256': parent_hashes['selection_manifest.json'],
            'parent_selected_cds_sha256': parent_hashes['selected_cds.parquet'],
            'parent_inspection_manifest_sha256': parent_hashes['inspection_manifest.json'],
            'parent_gap_pattern_shares_sha256': parent_hashes['gap_pattern_shares.csv'],
            'share_denominator_unique_cds': int(parent_records['cds_dna_hash'].nunique()),
            # Gap-pattern names are local to the parent alignment; realignment names its own.
            'kept_patterns': kept_rows,
            'dropped_patterns': dropped_rows,
            'parent_selection': count_records(parent_records),
            'kept': count_records(selected),
            'dropped': count_records(dropped),
        },
        # The population counts are the parent's: this selection cannot recount them. The
        # rest describe the records written.
        'counts': {
            'population_records': parent_counts['population_records'],
            'population_isolates': parent_counts['population_isolates'],
            'complete_records': len(selected),
            'complete_isolates': int(selected['assembly_id'].nunique()),
            'excluded_incomplete_records': parent_counts['excluded_incomplete_records'],
            'unknown_completeness_records': parent_counts['unknown_completeness_records'],
            'unique_complete_cds': int(selected['cds_dna_hash'].nunique()),
            'unique_complete_proteins': int(selected['prot_seq'].nunique()),
            'unique_proteins_for_alignment': len(proteins),
            'isolates_with_x': int(x_summary['isolates_with_x'].sum()),
            'unique_cds_with_x': int(x_summary['unique_cds_with_x'].sum()),
            'unique_proteins_with_x': int(x_summary['unique_proteins_with_x'].sum()),
        },
        'checks': {
            'unique_cds_checked': int(selected['cds_dna_hash'].nunique()),
            'translation_and_terminal_stop_passed': True,
            'dna_length_and_protein_hash_passed': True,
        },
        'code': get_git_info(),
        'outputs': list(OUTPUT_NAMES),
    }
    write_selection(out_dir, selected, summary, proteins, x_summary, manifest, overwrite)

    kept_counts = manifest['gap_pattern_filter']['kept']
    dropped_counts = manifest['gap_pattern_filter']['dropped']
    print(f'{parent_selection["protein"]}, {parent_selection["population"]}: gap patterns kept '
          f'{[row["gap_pattern"] for row in kept_rows]}; dropped {[row["gap_pattern"] for row in dropped_rows]}')
    print(f'Kept {kept_counts["isolates"]:,} isolates, {kept_counts["unique_cds"]:,} unique CDS and '
          f'{kept_counts["unique_proteins"]:,} unique proteins; dropped {dropped_counts["isolates"]:,} '
          f'isolates, {dropped_counts["unique_cds"]:,} unique CDS and '
          f'{dropped_counts["unique_proteins"]:,} unique proteins.')
    print(f'Wrote selection and protein-preparation outputs to {out_dir}')


def main() -> None:
    """Prepare CDS inputs, align proteins, validate them, inspect their gaps or select kept gap patterns.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--protein', required=True, help='protein short name, e.g. NS1')
    parser.add_argument('--cds_final', type=Path,
        default=PROJ / 'data/processed/flu/July_2025/cds_dna_final.parquet')
    parser.add_argument('--virus_config', type=Path, default=PROJ / 'conf/virus/flu.yaml')
    parser.add_argument('--metadata_file', type=Path,
        default=PROJ / 'data/processed/flu/metadata_eda/flu_genomes_metadata_parsed.csv')
    parser.add_argument('--host', nargs='+', default=None)
    parser.add_argument('--hn_subtype', nargs='+', default=None)
    years = parser.add_mutually_exclusive_group()
    years.add_argument('--year', nargs='+', type=int, default=None)
    years.add_argument('--year_range', nargs=2, type=int, default=None, metavar=('MIN', 'MAX'))
    parser.add_argument('--out_dir', type=Path,
        default=PROJ / 'results/flu/July_2025/cds_alignment_pilot')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--align_only', action='store_true',
        help='run MAFFT on existing preparation outputs without selecting CDS again')
    actions.add_argument('--validate_only', action='store_true',
        help='validate an existing alignment without rerunning MAFFT')
    actions.add_argument('--inspect_only', action='store_true',
        help='record gap positions in an existing validated alignment')
    parser.add_argument('--mafft_bin', default='mafft',
        help='MAFFT executable path or command on PATH (with --align_only)')
    parser.add_argument(
        '--mafft_strategy', choices=sorted(MAFFT_STRATEGY_OPTIONS),
        default='fft_ns_2',
        help="which MAFFT options to pass (with --align_only). 'fft_ns_2' passes "
             "--retree 2 --maxiterate 0, which give FFT-NS-2 at any input size, "
             "so every protein gets the same strategy. 'auto' passes --auto and "
             "lets MAFFT choose the strategy from the input size")
    parser.add_argument('--length_share_above', type=float, default=None,
        help='keep only the CDS lengths whose share of the unique complete CDS is above this '
             'fraction, e.g. 0.02. Outputs go to a subdirectory named '
             'length_share_above_<value>, so pass the same value with --align_only, '
             '--validate_only and --inspect_only')
    parser.add_argument('--gap_pattern_share_above', type=float, default=None,
        help='with --length_share_above, after that run is inspected: keep only the CDS whose '
             "protein's gap pattern in its alignment holds more than this fraction of its "
             'unique CDS, e.g. 0.02, and prepare them for realignment. Reads that run\'s '
             'outputs, not the corpus. Outputs go to its subdirectory '
             'gap_pattern_share_above_<value>, so pass both values with --align_only, '
             '--validate_only and --inspect_only')
    parser.add_argument('--overwrite', action='store_true',
        help='replace existing outputs for this population and protein')
    args = parser.parse_args()
    if args.length_share_above is not None and not 0 <= args.length_share_above < 1:
        parser.error('--length_share_above must be a fraction, at least 0 and below 1.')
    if args.gap_pattern_share_above is not None:
        if args.length_share_above is None:
            parser.error('--gap_pattern_share_above needs --length_share_above: it filters the '
                         'inspected alignment of that run.')
        if not 0 <= args.gap_pattern_share_above < 1:
            parser.error('--gap_pattern_share_above must be a fraction, at least 0 and below 1.')

    names = load_function_metadata(args.virus_config)
    if args.protein not in names.short_to_function:
        parser.error(f'Unknown protein {args.protein!r}; choose from {sorted(names.short_to_function)}')
    label = population_label(
        host=args.host,
        hn_subtype=args.hn_subtype,
        year=args.year,
        year_range=args.year_range)
    out_dir = args.out_dir / label / args.protein
    if args.length_share_above is not None:
        # A length-filtered selection is a different alignment input, so it gets its own
        # directory and cannot replace the alignment of all lengths.
        out_dir = out_dir / f'length_share_above_{args.length_share_above:g}'
    if args.gap_pattern_share_above is not None:
        # The records kept from that alignment are a new alignment input, so they get their
        # own directory inside the one they were selected from.
        out_dir = out_dir / f'gap_pattern_share_above_{args.gap_pattern_share_above:g}'

    if args.align_only:
        align_prepared_proteins(out_dir, args.mafft_bin, args.overwrite, args.mafft_strategy)
        return

    if args.validate_only:
        validate_alignment(out_dir, args.overwrite)
        return

    if args.inspect_only:
        inspect_alignment(out_dir, args.protein, args.overwrite)
        return

    if args.gap_pattern_share_above is not None:
        select_kept_gap_patterns(out_dir.parent, out_dir, args.gap_pattern_share_above,
                                 args.overwrite)
        return

    function = names.short_to_function[args.protein]
    cds = pd.read_parquet(args.cds_final, columns=CDS_COLUMNS,
                          filters=[('function', '==', function)])

    population = filter_by_metadata(
        attach_isolate_metadata(cds, metadata_file=args.metadata_file),
        host=args.host, hn_subtype=args.hn_subtype,
        year=args.year, year_range=args.year_range)
    if population.empty:
        raise ValueError('No protein records match the requested population.')

    complete = population.loc[population['is_complete_cds'].eq(True)].copy()
    if complete.empty:
        raise ValueError('The requested protein and population have no complete CDS.')

    share_above = 0.0 if args.length_share_above is None else args.length_share_above
    selected, summary = keep_cds_lengths_above_share(complete, share_above)
    selected = selected.sort_values(['cds_length', 'cds_dna_hash', 'assembly_id']).reset_index(drop=True)
    kept_lengths = summary.loc[summary['kept'], 'cds_length'].tolist()
    dropped_lengths = summary.loc[~summary['kept'], 'cds_length'].tolist()

    proteins = prepare_proteins(selected)
    x_summary = summarize_unknown_residues(selected)

    manifest = {
        'stage': 'protein_alignment_preparation',
        'protein': args.protein,
        'function': function,
        'population': label,
        'inputs': {
            'cds_final': str(args.cds_final.resolve()),
            'metadata_file': str(args.metadata_file.resolve()),
            'virus_config': str(args.virus_config.resolve()),
        },
        'filters': {
            'host': args.host, 'hn_subtype': args.hn_subtype,
            'year': args.year, 'year_range': args.year_range,
            'is_complete_cds': True,
            'length_share_above': args.length_share_above,
        },
        # `counts` below describe the records at the kept lengths, which are the ones written.
        'length_filter': {
            'share_above': share_above,
            'kept_lengths': [int(length) for length in kept_lengths],
            'dropped_lengths': [int(length) for length in dropped_lengths],
            'complete_records_all_lengths': len(complete),
            'complete_isolates_all_lengths': int(complete['assembly_id'].nunique()),
            'unique_complete_cds_all_lengths': int(complete['cds_dna_hash'].nunique()),
            'unique_complete_proteins_all_lengths': int(complete['prot_seq'].nunique()),
        },
        'counts': {
            'population_records': len(population),
            'population_isolates': int(population['assembly_id'].nunique()),
            'complete_records': len(selected),
            'complete_isolates': int(selected['assembly_id'].nunique()),
            'excluded_incomplete_records': int(population['is_complete_cds'].eq(False).sum()),
            'unknown_completeness_records': int(population['is_complete_cds'].isna().sum()),
            'unique_complete_cds': int(selected['cds_dna_hash'].nunique()),
            'unique_complete_proteins': int(selected['prot_seq'].nunique()),
            'unique_proteins_for_alignment': len(proteins),
            'isolates_with_x': int(x_summary['isolates_with_x'].sum()),
            'unique_cds_with_x': int(x_summary['unique_cds_with_x'].sum()),
            'unique_proteins_with_x': int(x_summary['unique_proteins_with_x'].sum()),
        },
        'checks': {
            'unique_cds_checked': int(selected['cds_dna_hash'].nunique()),
            'translation_and_terminal_stop_passed': True,
            'dna_length_and_protein_hash_passed': True,
        },
        'code': get_git_info(),
        'outputs': list(OUTPUT_NAMES),
    }
    write_selection(out_dir, selected, summary, proteins, x_summary, manifest, args.overwrite)

    print(f'{args.protein}, {label}: {manifest["counts"]["population_isolates"]:,} isolates before selection')
    print(f'CDS lengths kept: {kept_lengths}; dropped: {dropped_lengths}')
    print(f'{manifest["counts"]["complete_isolates"]:,} isolates with complete CDS at kept lengths; '
          f'{manifest["counts"]["unique_complete_cds"]:,} unique CDS; '
          f'{manifest["counts"]["unique_complete_proteins"]:,} unique proteins')
    print(summary.to_string(index=False))
    print(x_summary.to_string(index=False))
    print(f'Wrote selection and protein-preparation outputs to {out_dir}')


if __name__ == '__main__':
    main()
