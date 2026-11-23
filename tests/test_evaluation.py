"""Έλεγχοι της αξιολόγησης: οι μετρικές συμπεριφέρονται σωστά σε γνωστές περιπτώσεις."""

import numpy as np

from src.evaluation import bootstrap_ci, compute_metrics, grid_edge_rows, tune_threshold


def test_perfect_predictions():
    y = np.array([0, 0, 1, 1, 0, 1])
    m = compute_metrics(y, y.astype(float))
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert m[key] == 1.0
    assert m["fp"] == m["fn"] == 0


def test_random_predictions_auc_near_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    m = compute_metrics(y, rng.random(5000))
    assert 0.45 <= m["roc_auc"] <= 0.55


def test_tuned_threshold_in_range():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    prob = np.clip(y * 0.3 + rng.random(1000) * 0.7, 0, 1)
    t = tune_threshold(y, prob)
    assert 0.01 <= t <= 0.99


def test_confusion_counts_add_up():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    m = compute_metrics(y, rng.random(300), threshold=0.4)
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == 300


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 800)
    prob = np.clip(y * 0.4 + rng.random(800) * 0.6, 0, 1)
    point = compute_metrics(y, prob)["roc_auc"]
    low, high = bootstrap_ci(y, prob, 0.5, "roc_auc", n=200)
    assert low <= point <= high


def test_grid_edge_detection():
    space = {"model__C": [0.1, 1, 10], "model__kernel": ["a", "b", "c"], "model__depth": [None, 2, 4, 8]}
    rows = grid_edge_rows("m", space, {"C": 10, "kernel": "c", "depth": 4}, threshold=0.99)
    found = {(r["parameter"], r["edge"]) for r in rows}
    assert found == {("C", "max"), ("threshold", "max")}
