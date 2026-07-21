"""Tests for the P2 edge-cut fragmentation wiring in `dataset_pairs_cc.assign_atoms_prod`.

Builds the OOD nt_cds t095 positives once and runs `assign_atoms_prod` natural vs edge-cut,
asserting the invariants (no-op natural path, atom growth, cluster-disjointness, natural_cc_id).
SKIPs when the production OOD clusters are absent.

Run: pytest tests/test_dataset_pairs_cc_cut.py   (or: python tests/test_dataset_pairs_cc_cut.py)
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from omegaconf import OmegaConf

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.datasets._split_helpers import load_cluster_lookup
from src.datasets.dataset_pairs_cc import _POS_HASH, _resolve_spec, assign_atoms_prod, build_frontend
from src.datasets.dataset_segment_pairs_v2 import create_positive_pairs_v2
from src.utils.config_hydra import get_virus_config_hydra

OOD_CLUSTERS = PROJ / 'data' / 'processed' / 'flu' / 'July_2025' / 'clusters_nt_cds_ood'
_CACHE: dict = {}


def _ood_absent(threshold='t095'):
    return not (OOD_CLUSTERS / threshold / 'combined_cluster.parquet').exists()


def _build_pos_lookup(threshold='t095'):
    """(pos, cluster_lookup, pos_hash_col) for OOD nt_cds HA-NA at `threshold` -- the exact inputs
    `dataset_pairs_cc._build_positives` feeds `assign_atoms_prod`. Cached across tests (front-end
    load is the slow part)."""
    if threshold in _CACHE:
        return _CACHE[threshold]
    bundle = 'flu_ha_na_cc_nt_cds_ood'
    cpath = f'data/processed/flu/July_2025/clusters_nt_cds_ood/{threshold}/combined_cluster.parquet'
    cfg = get_virus_config_hydra(bundle, config_path=str(PROJ / 'conf'))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f'dataset.split_strategy.cluster_id_path={cpath}']))
    args = SimpleNamespace(config_bundle=bundle, protein_final=None, override=None, out_dir=None)
    spec = _resolve_spec(args, cfg)
    input_file = spec.cluster_id_path.parents[2] / 'protein_final.parquet'
    df = build_frontend(cfg, input_file, (spec.fa, spec.fb),
                        cds_final_path=input_file.parent / 'cds_dna_final.parquet')
    pos, _ = create_positive_pairs_v2(df, schema_pair=(spec.fa, spec.fb),
                                      pair_key_alphabet=spec.pair_key_alphabet)
    lookup = load_cluster_lookup(spec.cluster_id_path)
    _CACHE[threshold] = (pos, lookup, _POS_HASH[spec.alphabet])
    return _CACHE[threshold]


def _max_atoms_per_cluster(pos_ids):
    """Cluster-disjointness metric: the most distinct atom_ids any single-side cluster maps to.
    1 == every cluster (both slots) lands in exactly one atom (so no cluster spans folds)."""
    a = pos_ids[['cluster_id_a', 'atom_id']].rename(columns={'cluster_id_a': 'cid'})
    b = pos_ids[['cluster_id_b', 'atom_id']].rename(columns={'cluster_id_b': 'cid'})
    return int(pd.concat([a, b]).groupby('cid')['atom_id'].nunique().max())


def test_edge_cut_disabled_is_natural():
    """edge_cut=None: the pre-existing natural path -- 108 atoms, atom==cc, no natural_cc_id."""
    if _ood_absent():
        print('SKIP test_edge_cut_disabled_is_natural: OOD clusters absent')
        return
    pos, lookup, hcol = _build_pos_lookup()
    nat, summ = assign_atoms_prod(pos, lookup, hcol, edge_cut=None)
    assert nat['atom_id'].nunique() == 108                      # natural bipartite CCs
    assert (nat['atom_id'] == nat['cc_id']).all()               # atom == cc
    assert 'natural_cc_id' not in nat.columns                   # no cut -> no snapshot column
    assert 'edge_cut' not in summ                               # no cut audit
    assert _max_atoms_per_cluster(nat) == 1                     # cluster-disjoint
    assert len(nat) == len(pos)                                 # no pairs dropped


def test_edge_cut_grows_atoms_and_stays_cluster_disjoint():
    """edge_cut target 124: grow 108 -> ~124 atoms within a 2% drop budget, cluster-disjoint,
    with natural_cc_id retained. n_atoms is the stop target (stable); the exact dropped set isn't
    bit-locked (spectral eigensolver FP) -- so only construction/structural invariants here."""
    if _ood_absent():
        print('SKIP test_edge_cut_grows_atoms: OOD clusters absent')
        return
    pos, lookup, hcol = _build_pos_lookup()
    ec = {'enabled': True, 'cut_method': 'spectral', 'target_atoms': 124, 'max_drop_frac': 0.02, 'seed': 42}
    cut, summ = assign_atoms_prod(pos, lookup, hcol, edge_cut=ec)
    audit = summ['edge_cut']
    assert 124 <= cut['atom_id'].nunique() <= 128               # hit the target (may overshoot by a cut)
    assert cut['atom_id'].nunique() == audit['n_atoms']         # audit matches the routed atoms
    assert (cut['atom_id'] == cut['cc_id']).all()               # atom == cc (fragment)
    assert cut['natural_cc_id'].nunique() == 108                # pre-cut CCs retained for analysis
    assert _max_atoms_per_cluster(cut) == 1                     # cluster-disjoint after fragmentation
    assert audit['dropped_frac'] <= 0.02                        # within the budget (guaranteed)
    assert len(cut) == len(pos) - audit['pairs_dropped']        # kept + dropped partition pos
    assert audit['stopped_reason'] == 'stop_fn'                 # reached the target before the budget


def test_edge_cut_target_below_natural_does_zero_cuts():
    """A target <= the natural atom count needs no cut (already satisfied)."""
    if _ood_absent():
        print('SKIP test_edge_cut_target_below_natural: OOD clusters absent')
        return
    pos, lookup, hcol = _build_pos_lookup()
    ec = {'enabled': True, 'cut_method': 'spectral', 'target_atoms': 50, 'max_drop_frac': 0.02, 'seed': 42}
    cut, summ = assign_atoms_prod(pos, lookup, hcol, edge_cut=ec)
    assert summ['edge_cut']['n_cuts'] == 0 and summ['edge_cut']['pairs_dropped'] == 0
    assert cut['atom_id'].nunique() == 108                      # unchanged natural atom count
    assert len(cut) == len(pos)


if __name__ == '__main__':
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and callable(_fn):
            print(f'{_name} ...', flush=True)
            _fn()
    print('Done. All tests passed.')
