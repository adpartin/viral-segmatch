"""Regression tests for GTO duplicate handling."""

import pandas as pd
import pytest

from src.utils.gto_utils import handle_assembly_duplicates


def _row(feature_id: str, function: str, file_name: str = 'a.gto') -> dict:
    return {
        'prot_seq': 'MPEPTIDE*',
        'assembly_id': 'assembly-a',
        'function': function,
        'file': file_name,
        'brc_fea_id': feature_id,
    }


def test_same_sequence_with_different_functions_is_retained():
    source = pd.DataFrame([
        _row('feature-1', 'Nuclear export protein'),
        _row('feature-2', 'Hypothetical host adaptation protein NS3'),
    ])

    result, summary = handle_assembly_duplicates(source)

    assert result['brc_fea_id'].tolist() == ['feature-1', 'feature-2']
    assert summary.empty


@pytest.mark.parametrize(
    ('strategy', 'expected_feature'),
    [('keep_first', 'feature-1'), ('keep_last', 'feature-2')],
)
def test_same_function_duplicate_uses_requested_strategy(strategy, expected_feature):
    source = pd.DataFrame([
        _row('feature-1', 'Nuclear export protein'),
        _row('feature-2', 'Nuclear export protein'),
    ])

    result, summary = handle_assembly_duplicates(source, strategy=strategy)

    assert result['brc_fea_id'].tolist() == [expected_feature]
    assert set(summary['action_taken']) == {'kept', 'removed'}


def test_invalid_duplicate_strategy_is_rejected():
    source = pd.DataFrame([_row('feature-1', 'Nuclear export protein')])

    with pytest.raises(ValueError, match='Invalid duplicate strategy'):
        handle_assembly_duplicates(source, strategy='first')
