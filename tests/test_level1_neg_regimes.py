"""Sanity test for the new Level 1 plots in analyze_stage4_train.py.

Constructs a hand-built predictions DataFrame with known regimes / labels /
predictions, then runs:

  * analyze_level1_neg_regimes (Plot A: 9-regime view)
  * analyze_level1_neg_regimes_agg (Plot B: aggregated by match-count)

twice -- once with the v2-written `neg_regime` / `metadata_match_count`
columns present, once stripped -- and asserts:

  1. Per-bucket n_samples / TPR / TNR match expected hand-computed values.
  2. The two paths (sampler-written column vs derive-from-metadata) agree
     on regime + match_count.

Run: python tests/test_level1_neg_regimes.py
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

import matplotlib
matplotlib.use('Agg')  # headless

from src.analysis.analyze_stage4_train import (
    analyze_level1_neg_regimes,
    analyze_level1_neg_regimes_agg,
    _resolve_neg_regime_column,
    _resolve_match_count_column,
    _derive_neg_regime,
)


# ---------------------------------------------------------------------------
# Synthetic predictions DataFrame.
#
# Hand-built so each regime has a known count and the per-bar TPR / TNR are
# easy to compute on paper:
#
#   - 4 positives (label=1), 3 correctly predicted positive -> TPR = 0.75
#   - 1 negative per regime in {none_match, host_only, subtype_only,
#     year_only, host_subtype_only, host_year_only, subtype_year_only},
#     each correctly predicted negative -> TNR = 1.0
#   - 4 negatives in host_subtype_year, 1 correctly predicted neg ->
#     TNR = 0.25 (the hardest regime, model fails)
#   - 1 negative with a null axis -> unknown_metadata_neg, predicted neg ->
#     TNR = 1.0
#
# Match-count rollup (Plot B):
#   match_count_0 = 1 neg (none_match), TNR = 1.0
#   match_count_1 = 3 negs (host_only, subtype_only, year_only), TNR = 1.0
#   match_count_2 = 3 negs (host_subtype_only, host_year_only,
#                           subtype_year_only), TNR = 1.0
#   match_count_3 = 4 negs (host_subtype_year), TNR = 0.25
#   unknown_metadata_neg = 1 neg, TNR = 1.0
# ---------------------------------------------------------------------------

# Known regime axis values for the 8 metadata-defined buckets. Each row is
# a (host_a, host_b, hn_subtype_a, hn_subtype_b, year_a, year_b) sextuple
# chosen so that the (host_match, subtype_match, year_bin_match) tuple
# uniquely yields the named regime.
# Year bins: 2010 -> '<=2015', 2018 -> '2016-2020', 2024 -> '2021+'.
_NEG_ROWS = [
    # (regime, host_a, host_b, sub_a, sub_b, year_a, year_b, pred_label)
    ('none_match',         'Pig',   'Human', 'H1N1', 'H3N2', 2010, 2024, 0),
    ('host_only',          'Human', 'Human', 'H1N1', 'H3N2', 2010, 2024, 0),
    ('subtype_only',       'Pig',   'Human', 'H3N2', 'H3N2', 2010, 2024, 0),
    ('year_only',          'Pig',   'Human', 'H1N1', 'H3N2', 2024, 2024, 0),
    ('host_subtype_only',  'Human', 'Human', 'H3N2', 'H3N2', 2010, 2024, 0),
    ('host_year_only',     'Human', 'Human', 'H1N1', 'H3N2', 2024, 2024, 0),
    ('subtype_year_only',  'Pig',   'Human', 'H3N2', 'H3N2', 2024, 2024, 0),
    # 4 host_subtype_year negatives, only 1 predicted correctly -> TNR=0.25
    ('host_subtype_year',  'Human', 'Human', 'H3N2', 'H3N2', 2024, 2024, 0),
    ('host_subtype_year',  'Human', 'Human', 'H3N2', 'H3N2', 2024, 2024, 1),
    ('host_subtype_year',  'Human', 'Human', 'H3N2', 'H3N2', 2024, 2024, 1),
    ('host_subtype_year',  'Human', 'Human', 'H3N2', 'H3N2', 2024, 2024, 1),
    # one unknown-metadata-neg (null host on side B)
    ('unknown_metadata_neg', 'Human', None, 'H3N2', 'H3N2', 2024, 2024, 0),
]
# 4 positives, 3 predicted correctly -> TPR=0.75
_POS_ROWS = [
    # (host, subtype, year, pred_label)
    ('Human', 'H3N2', 2024, 1),
    ('Human', 'H3N2', 2024, 1),
    ('Human', 'H3N2', 2024, 1),
    ('Pig',   'H1N1', 2018, 0),
]


def _make_predictions_df(with_sampler_columns: bool) -> pd.DataFrame:
    """Build the predictions DataFrame.

    `with_sampler_columns=True`  -> include `neg_regime` + `metadata_match_count`
                                    (regime-aware path).
    `with_sampler_columns=False` -> drop them; the analyzer must derive both.
    """
    regime_to_count = {
        'none_match': 0,
        'host_only': 1, 'subtype_only': 1, 'year_only': 1,
        'host_subtype_only': 2, 'host_year_only': 2, 'subtype_year_only': 2,
        'host_subtype_year': 3,
        'unknown_metadata_neg': pd.NA,
    }

    rows = []
    for regime, ha, hb, sa, sb, ya, yb, pl in _NEG_ROWS:
        rows.append({
            'label': 0,
            'pred_label': pl,
            'pred_prob': 0.9 if pl == 1 else 0.1,
            'host_a': ha, 'host_b': hb,
            'hn_subtype_a': sa, 'hn_subtype_b': sb,
            'year_a': ya, 'year_b': yb,
            'neg_regime': regime,
            'metadata_match_count': regime_to_count[regime],
        })
    for h, s, y, pl in _POS_ROWS:
        rows.append({
            'label': 1,
            'pred_label': pl,
            'pred_prob': 0.9 if pl == 1 else 0.1,
            'host_a': h, 'host_b': h,
            'hn_subtype_a': s, 'hn_subtype_b': s,
            'year_a': y, 'year_b': y,
            'neg_regime': pd.NA,
            'metadata_match_count': pd.NA,
        })

    df = pd.DataFrame(rows)
    if not with_sampler_columns:
        df = df.drop(columns=['neg_regime', 'metadata_match_count'])
    return df


def test_resolve_helpers_agree_across_paths():
    """Both column-resolvers must produce identical results on the two
    paths (sampler-written vs derived). Tests _resolve_neg_regime_column
    and _resolve_match_count_column directly."""
    with_cols = _make_predictions_df(with_sampler_columns=True)
    no_cols = _make_predictions_df(with_sampler_columns=False)

    r_with = _resolve_neg_regime_column(with_cols)
    r_no = _resolve_neg_regime_column(no_cols)
    assert r_with.tolist() == r_no.tolist(), (
        f"neg_regime resolution differs across paths.\n"
        f"with_sampler:  {r_with.tolist()}\n"
        f"derived:       {r_no.tolist()}"
    )

    mc_with = _resolve_match_count_column(with_cols).tolist()
    mc_no = _resolve_match_count_column(no_cols).tolist()
    # tolist() returns pd.NA for nullable Int64 nulls; comparison via repr
    # avoids the "NA != NA" trap when both are NA.
    assert [repr(v) for v in mc_with] == [repr(v) for v in mc_no], (
        f"metadata_match_count resolution differs across paths.\n"
        f"with_sampler:  {mc_with}\n"
        f"derived:       {mc_no}"
    )


def _check_level1_regimes(stats_df: pd.DataFrame, label: str):
    """Assert per-regime n_samples / TPR / TNR match the hand-built dataset."""
    expected = {
        'positive':            (4, 0.75, None),
        'none_match':          (1, None, 1.0),
        'host_only':           (1, None, 1.0),
        'subtype_only':        (1, None, 1.0),
        'year_only':           (1, None, 1.0),
        'host_subtype_only':   (1, None, 1.0),
        'host_year_only':      (1, None, 1.0),
        'subtype_year_only':   (1, None, 1.0),
        'host_subtype_year':   (4, None, 0.25),
        'unknown_metadata_neg': (1, None, 1.0),
    }
    by_regime = stats_df.set_index('regime')
    for regime, (n_exp, tpr_exp, tnr_exp) in expected.items():
        assert regime in by_regime.index, f"[{label}] regime {regime!r} missing from output"
        row = by_regime.loc[regime]
        assert int(row['n_samples']) == n_exp, (
            f"[{label}] regime {regime}: n_samples expected {n_exp}, got {int(row['n_samples'])}"
        )
        if tpr_exp is not None:
            assert abs(row['tpr'] - tpr_exp) < 1e-9, (
                f"[{label}] regime {regime}: tpr expected {tpr_exp}, got {row['tpr']}"
            )
        if tnr_exp is not None:
            assert abs(row['tnr'] - tnr_exp) < 1e-9, (
                f"[{label}] regime {regime}: tnr expected {tnr_exp}, got {row['tnr']}"
            )


def _check_level1_agg(stats_df: pd.DataFrame, label: str):
    """Assert per-bucket n_samples / TPR / TNR for the aggregated view."""
    expected = {
        'positive':              (4, 0.75, None),
        'match_count_0':         (1, None, 1.0),
        'match_count_1':         (3, None, 1.0),
        'match_count_2':         (3, None, 1.0),
        'match_count_3':         (4, None, 0.25),
        'unknown_metadata_neg':  (1, None, 1.0),
    }
    by_bucket = stats_df.set_index('bucket')
    for bucket, (n_exp, tpr_exp, tnr_exp) in expected.items():
        assert bucket in by_bucket.index, f"[{label}] bucket {bucket!r} missing from output"
        row = by_bucket.loc[bucket]
        assert int(row['n_samples']) == n_exp, (
            f"[{label}] bucket {bucket}: n_samples expected {n_exp}, got {int(row['n_samples'])}"
        )
        if tpr_exp is not None:
            assert abs(row['tpr'] - tpr_exp) < 1e-9, (
                f"[{label}] bucket {bucket}: tpr expected {tpr_exp}, got {row['tpr']}"
            )
        if tnr_exp is not None:
            assert abs(row['tnr'] - tnr_exp) < 1e-9, (
                f"[{label}] bucket {bucket}: tnr expected {tnr_exp}, got {row['tnr']}"
            )


def test_level1_neg_regimes_with_sampler_columns():
    df = _make_predictions_df(with_sampler_columns=True)
    with tempfile.TemporaryDirectory() as td:
        out = analyze_level1_neg_regimes(df, Path(td))
        _check_level1_regimes(out, label='with_sampler')
        # PNG and CSV should both land
        assert (Path(td) / 'level1_neg_regimes.csv').exists()
        assert (Path(td) / 'level1_neg_regimes.png').exists()


def test_level1_neg_regimes_derived_from_metadata():
    df = _make_predictions_df(with_sampler_columns=False)
    with tempfile.TemporaryDirectory() as td:
        out = analyze_level1_neg_regimes(df, Path(td))
        _check_level1_regimes(out, label='derived')


def test_level1_agg_with_sampler_columns():
    df = _make_predictions_df(with_sampler_columns=True)
    with tempfile.TemporaryDirectory() as td:
        out = analyze_level1_neg_regimes_agg(df, Path(td))
        _check_level1_agg(out, label='with_sampler')
        assert (Path(td) / 'level1_neg_regimes_agg.csv').exists()
        assert (Path(td) / 'level1_neg_regimes_agg.png').exists()


def test_level1_agg_derived_from_metadata():
    df = _make_predictions_df(with_sampler_columns=False)
    with tempfile.TemporaryDirectory() as td:
        out = analyze_level1_neg_regimes_agg(df, Path(td))
        _check_level1_agg(out, label='derived')


def test_derive_neg_regime_handles_each_class():
    """Direct test of _derive_neg_regime on one row per regime."""
    df = _make_predictions_df(with_sampler_columns=False)
    neg_only = df[df['label'] == 0].copy()
    derived = _derive_neg_regime(neg_only)
    expected_per_row = [r[0] for r in _NEG_ROWS]
    assert derived.tolist() == expected_per_row, (
        f"_derive_neg_regime mismatch.\n"
        f"expected: {expected_per_row}\n"
        f"got:      {derived.tolist()}"
    )


if __name__ == '__main__':
    test_resolve_helpers_agree_across_paths()
    test_derive_neg_regime_handles_each_class()
    test_level1_neg_regimes_with_sampler_columns()
    test_level1_neg_regimes_derived_from_metadata()
    test_level1_agg_with_sampler_columns()
    test_level1_agg_derived_from_metadata()
    print('Done. All tests passed.')
