"""Tests for the corpus-wide co-occurrence blocking set.

The builders read the set from the Stage-1/1.5 tables rather than from the filtered frame,
so a candidate negative is rejected when its two sequences co-occur in any corpus isolate.
These tests write small `*_final` tables, check what the loader reads for each alphabet, and
check that a CV run draws no negative from the set it was handed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.datasets._pair_helpers import (
    build_cooccurrence_set,
    build_corpus_cooccurrence_set,
    canonical_pair_key,
    load_corpus_hash_frame,
)
from src.datasets.dataset_segment_pairs_v2 import generate_all_cv_folds_v2

# Two functions in the toy corpus, plus one that is never requested.
FUNC_A, FUNC_B, FUNC_OTHER = 'Protein A', 'Protein B', 'Protein C'

# One row per (isolate, function): (assembly_id, function, sequence token).
# i0 and i1 are the population; i2 co-occurs a0 with b1, which the population never
# observes together although both sequences are in it; i3 holds an incomplete CDS at a
# length the pinned-length filter would drop.
CORPUS_RECORDS = [
    ('i0', FUNC_A, 'a0'), ('i0', FUNC_B, 'b0'),
    ('i1', FUNC_A, 'a1'), ('i1', FUNC_B, 'b1'),
    ('i2', FUNC_A, 'a0'), ('i2', FUNC_B, 'b1'),
    ('i3', FUNC_A, 'a2'), ('i3', FUNC_B, 'b2'),
    ('i0', FUNC_OTHER, 'c0'), ('i1', FUNC_OTHER, 'c1'),
]
POPULATION_ISOLATES = ('i0', 'i1')


def _write_corpus(tmp_path: Path) -> Path:
    """Write protein_final, cds_dna_final and ctg_dna_final for CORPUS_RECORDS.

    Returns the protein path in the shape the config resolves (a `.csv` suffix beside the
    parquet that exists), so the test covers that resolution too.
    """
    prot_rows, cds_rows, ctg_rows = [], [], []
    for assembly_id, function, token in CORPUS_RECORDS:
        ctg_id = f'{assembly_id}_{token}_ctg'
        prot_rows.append({
            'assembly_id': assembly_id,
            'function': function,
            'genbank_ctg_id': ctg_id,
            'prot_hash': f'prot_{token}',
            'prot_seq': f'M{token}',  # never read by the loader
        })
        cds_rows.append({
            'assembly_id': assembly_id,
            'function': function,
            'cds_dna_hash': f'cds_{token}',
            'cds_length': 300 if assembly_id != 'i3' else 297,
            'is_complete_cds': assembly_id != 'i3',
        })
        ctg_rows.append({
            'assembly_id': assembly_id,
            'genbank_ctg_id': ctg_id,
            'ctg_dna_hash': f'ctg_{token}',
        })
    pd.DataFrame(prot_rows).to_parquet(tmp_path / 'protein_final.parquet', index=False)
    pd.DataFrame(cds_rows).to_parquet(tmp_path / 'cds_dna_final.parquet', index=False)
    pd.DataFrame(ctg_rows).to_parquet(tmp_path / 'ctg_dna_final.parquet', index=False)
    return tmp_path / 'protein_final.csv'


def _population_frame(hash_col: str, prefix: str) -> pd.DataFrame:
    """The same records as CORPUS_RECORDS, narrowed to POPULATION_ISOLATES and two functions.

    Stands in for the frame the orchestrator hands the splitter after its metadata filter.
    """
    rows = [
        {'assembly_id': assembly_id, 'function': function, hash_col: f'{prefix}_{token}'}
        for assembly_id, function, token in CORPUS_RECORDS
        if assembly_id in POPULATION_ISOLATES and function in (FUNC_A, FUNC_B)
    ]
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ('alphabet', 'hash_col', 'prefix'),
    [('aa', 'prot_hash', 'prot'),
     ('nt_cds', 'cds_dna_hash', 'cds'),
     ('nt_ctg', 'ctg_dna_hash', 'ctg')],
)
def test_loader_reads_the_alphabet_hash(tmp_path, alphabet, hash_col, prefix):
    protein_path = _write_corpus(tmp_path)
    obs = load_corpus_hash_frame(protein_path, (FUNC_A, FUNC_B), alphabet=alphabet)

    assert list(obs.columns) == ['assembly_id', 'function', hash_col]
    assert set(obs['function']) == {FUNC_A, FUNC_B}       # FUNC_OTHER is not requested
    assert set(obs['assembly_id']) == {'i0', 'i1', 'i2', 'i3'}
    assert len(obs) == 8
    expected = {f'{prefix}_{token}' for _, function, token in CORPUS_RECORDS
                if function in (FUNC_A, FUNC_B)}
    assert set(obs[hash_col]) == expected


def test_corpus_set_holds_a_pair_the_population_never_observes(tmp_path):
    protein_path = _write_corpus(tmp_path)
    corpus_pairs, corpus_stats = build_corpus_cooccurrence_set(
        protein_path, (FUNC_A, FUNC_B), alphabet='nt_cds')
    population_pairs, _ = build_cooccurrence_set(
        _population_frame('cds_dna_hash', 'cds'), hash_col='cds_dna_hash')

    outside = canonical_pair_key('cds_a0', 'cds_b1')      # observed in i2 only
    assert outside in corpus_pairs
    assert outside not in population_pairs
    assert population_pairs < corpus_pairs                 # strict superset
    assert corpus_stats['scope'] == 'corpus'
    assert corpus_stats['hash_col'] == 'cds_dna_hash'


def test_corpus_set_keeps_records_the_pinned_length_filter_drops(tmp_path):
    """i3's CDS is incomplete and off the modal length, and still blocks its pair."""
    protein_path = _write_corpus(tmp_path)
    corpus_pairs, _ = build_corpus_cooccurrence_set(
        protein_path, (FUNC_A, FUNC_B), alphabet='nt_cds')

    assert canonical_pair_key('cds_a2', 'cds_b2') in corpus_pairs


def test_unknown_function_raises(tmp_path):
    protein_path = _write_corpus(tmp_path)
    with pytest.raises(ValueError, match='no corpus records'):
        load_corpus_hash_frame(protein_path, (FUNC_A, 'Protein Z'), alphabet='nt_cds')


def test_unknown_alphabet_raises(tmp_path):
    protein_path = _write_corpus(tmp_path)
    with pytest.raises(ValueError):
        load_corpus_hash_frame(protein_path, (FUNC_A, FUNC_B), alphabet='codon')


def test_contig_without_a_hash_raises(tmp_path):
    """nt_ctg joins protein_final to ctg_dna_final, so a missing contig row is fatal."""
    protein_path = _write_corpus(tmp_path)
    ctg = pd.read_parquet(tmp_path / 'ctg_dna_final.parquet')
    ctg.iloc[1:].to_parquet(tmp_path / 'ctg_dna_final.parquet', index=False)

    with pytest.raises(RuntimeError, match='no ctg_dna_hash'):
        load_corpus_hash_frame(protein_path, (FUNC_A, FUNC_B), alphabet='nt_ctg')


def _cv_frame(n_isolates: int = 12) -> pd.DataFrame:
    """A population frame of `n_isolates`, one FUNC_A and one FUNC_B record each."""
    rows = []
    for i in range(n_isolates):
        assembly_id = f'iso{i:02d}'
        for function, token in ((FUNC_A, f'a{i}'), (FUNC_B, f'b{i}')):
            rows.append({
                'assembly_id': assembly_id,
                'brc_fea_id': f'{assembly_id}_{token}',
                'genbank_ctg_id': f'{assembly_id}_{token}_ctg',
                'prot_seq': f'M{token}',
                'ctg_dna_seq': f'ATG{token}',
                'canonical_segment': function,
                'function': function,
                'prot_hash': f'prot_{token}',
                'ctg_dna_hash': f'ctg_{token}',
                'cds_dna_hash': f'cds_{token}',
            })
    return pd.DataFrame(rows)


def test_cv_draws_no_negative_from_the_blocking_set():
    """Every negative of every split avoids the set the generator was handed.

    The set carries the 12 observed pairs plus 11 pairs that no population isolate
    observes, each of which the sampler could otherwise draw, so the assertion fails if the
    generator blocks against anything narrower than the set it is given.
    """
    frame = _cv_frame()
    cooccur_pairs, cooccur_stats = build_cooccurrence_set(frame, hash_col='cds_dna_hash')
    blocked_elsewhere = {canonical_pair_key(f'cds_a{i}', f'cds_b{i + 1}') for i in range(11)}
    cooccur_pairs |= blocked_elsewhere
    cooccur_stats['total_cooccur_pairs'] = len(cooccur_pairs)
    cooccur_stats['scope'] = 'corpus'

    folds = list(generate_all_cv_folds_v2(
        df=frame,
        cooccur_pairs=cooccur_pairs,
        cooccur_stats=cooccur_stats,
        n_folds=3,
        seed=42,
        neg_to_pos_ratio=1.0,
        val_ratio=0.2,
        schema_pair=(FUNC_A, FUNC_B),
        axes_for_flags=[],
        pair_key_alphabet='nt_cds',
        negative_scope='within_fold',
    ))

    assert len(folds) == 3
    n_negatives = 0
    for fold in folds:
        for split_name in ('train_pairs', 'val_pairs', 'test_pairs'):
            pairs = fold[split_name]
            negatives = set(pairs.loc[pairs['label'] == 0, 'pair_key'].astype(str))
            n_negatives += len(negatives)
            assert not negatives & cooccur_pairs
    assert n_negatives > 0
