"""Regression tests for Stage 1 and Stage 1.5 CDS-completeness handling."""

import pandas as pd
import pytest

from src.preprocess.extract_cds_dna import (
    _build_output_frame,
    _coerce_boolean_flag,
)
from src.utils.protein_utils import analyze_protein_ambiguities


def test_starts_with_m_uses_normalized_protein_sequence():
    source = pd.DataFrame({'prot_seq': ['MPEP*', ' apep* ', ' mqq ', None]})

    result = analyze_protein_ambiguities(source)

    assert result['starts_with_m'].tolist() == [True, False, True, False]


def test_boolean_flag_accepts_booleans_and_exact_text_values():
    source = pd.Series([True, False, 'True', 'False'], dtype=object)

    result = _coerce_boolean_flag(source, 'starts_with_m')

    assert result.dtype == bool
    assert result.tolist() == [True, False, True, False]


@pytest.mark.parametrize('invalid', [None, 'unexpected', 1])
def test_boolean_flag_rejects_missing_or_unrecognized_values(invalid):
    with pytest.raises(ValueError, match='starts_with_m'):
        _coerce_boolean_flag(pd.Series([invalid], dtype=object), 'starts_with_m')


def test_build_output_frame_propagates_flags_and_derives_completeness():
    prot = pd.DataFrame({
        'assembly_id': ['a0', 'a1', 'a2'],
        'genbank_ctg_id': ['c0', 'c1', 'c2'],
        'brc_fea_id': ['f0', 'f1', 'f2'],
        'function': ['HA', 'HA', 'NA'],
        'canonical_segment': ['S4', 'S4', 'S6'],
        'prot_hash': ['p0', 'p1', 'p2'],
        'prot_seq': ['M*', 'A*', 'M**'],
        'length': [2, 2, 3],
        'starts_with_m': [True, False, True],
        'has_terminal_stop': [True, True, True],
        'has_internal_stop': [False, False, True],
    })

    result = _build_output_frame(
        prot,
        keep_idx=[0, 2],
        cds_list=['ATGTAA', 'ATGTAGTAA'],
        cds_hashes=['d0', 'd2'],
    )

    assert result['brc_fea_id'].tolist() == ['f0', 'f2']
    assert result['starts_with_m'].tolist() == [True, True]
    assert result['has_terminal_stop'].tolist() == [True, True]
    assert result['has_internal_stop'].tolist() == [False, True]
    assert result['is_complete_cds'].tolist() == [True, False]


def test_build_output_frame_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match='equal lengths'):
        _build_output_frame(pd.DataFrame(), [0], [], [])
