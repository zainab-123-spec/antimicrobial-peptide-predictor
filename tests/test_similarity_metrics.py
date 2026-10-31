import numpy as np

from ampred.metrics import binary_metrics, bootstrap_ci
from ampred.similarity import jaccard, kmers, max_similarity_to_reference


def test_kmers():
    assert kmers("ABCD", 3) == {"ABC", "BCD"}
    assert kmers("AB", 3) == set()


def test_jaccard_bounds():
    assert jaccard({"a"}, {"a"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(set(), set()) == 0.0


def test_identical_sequence_has_similarity_one():
    sim = max_similarity_to_reference(["GIGKFLHSAK"], ["MSTEDAAA", "GIGKFLHSAK"])
    assert np.isclose(sim[0], 1.0)


def test_unrelated_sequence_has_zero_similarity():
    sim = max_similarity_to_reference(["WWWWW"], ["AAAAA"])
    assert sim[0] == 0.0


def test_perfect_classifier_metrics():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["accuracy"] == 1.0 and m["mcc"] == 1.0 and m["auc"] == 1.0
    assert m["sensitivity"] == 1.0 and m["specificity"] == 1.0


def test_inverted_classifier():
    m = binary_metrics([0, 1], [0.9, 0.1])
    assert m["auc"] == 0.0 and m["accuracy"] == 0.0


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.25, 200), 0, 1)
    point = binary_metrics(y, p)["auc"]
    lo, hi = bootstrap_ci(y, p, "auc", n_boot=200)
    assert lo <= point <= hi
