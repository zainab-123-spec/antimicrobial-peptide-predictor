import numpy as np
import pytest

from ampred.features import (
    PHYSCHEM_NAMES,
    aa_composition,
    dipeptide_composition,
    featurize,
    hydrophobic_moment,
    net_charge,
    physchem,
)


def test_aac_sums_to_one():
    v = aa_composition("KKLLA")
    assert v.shape == (20,)
    assert np.isclose(v.sum(), 1.0)


def test_aac_counts():
    v = aa_composition("AAK")
    assert np.isclose(v[0], 2 / 3)  # 'A' is first


def test_dpc_shape_and_sum():
    v = dipeptide_composition("ACDA")
    assert v.shape == (400,)
    assert np.isclose(v.sum(), 1.0)


def test_dpc_single_residue_is_zero():
    assert dipeptide_composition("A").sum() == 0


def test_net_charge():
    assert net_charge("KRKDE") == 1.0
    assert net_charge("GGG") == 0.0


def test_hydrophobic_moment_amphipathic_beats_uniform():
    uniform = hydrophobic_moment("LLLLLLLLLLLL")
    # one residue per 100 deg: alternating faces of a helix
    amphi = hydrophobic_moment("LKKLLKKLLKKL")
    assert amphi > uniform >= 0


def test_physchem_length_matches_names():
    assert physchem("GIGKFLHSAKKF").shape == (len(PHYSCHEM_NAMES),)


def test_featurize_combines_blocks():
    x = featurize(["KKLL", "ACDEF"], "aac+dpc+physchem")
    assert x.shape == (2, 20 + 400 + len(PHYSCHEM_NAMES))


def test_featurize_unknown_block():
    with pytest.raises(ValueError):
        featurize(["AA"], "nonsense")
