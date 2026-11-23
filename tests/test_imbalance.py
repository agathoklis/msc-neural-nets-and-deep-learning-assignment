"""Έλεγχοι της ανισορροπίας κλάσεων: η επαναδειγματοληψία είναι σωστή και ο κανόνας επιλογής δουλεύει όπως ορίστηκε."""

import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL
from src.data import load_splits
from src.imbalance import choose_method, class_weight_kwargs, keras_class_weight, resample


@pytest.fixture(scope="module")
def splits():
    return load_splits()


def test_undersampling_balances_classes(splits):
    X_train, y_train = splits[0], splits[1]
    Xr, yr = resample(X_train, y_train, "undersample")
    counts = np.bincount(yr)
    assert counts[0] == counts[1] == (y_train == 1).sum()


def test_smote_nc_outputs_valid_categories(splits):
    """Οι συνθετικοί πελάτες έχουν μόνο κατηγορίες που υπάρχουν στα πραγματικά δεδομένα."""
    X_train, y_train = splits[0], splits[1]
    Xr, yr = resample(X_train, y_train, "smote_nc")
    assert np.bincount(yr)[0] == np.bincount(yr)[1]
    for col in CATEGORICAL:
        assert set(Xr[col].unique()) <= set(X_train[col].unique())
    assert set(Xr["SeniorCitizen"].unique()) <= {0, 1}


def test_resampling_does_not_touch_other_splits(splits):
    """Η resample επιστρέφει νέα δεδομένα· τα validation/test δεν περνούν ποτέ από αυτή."""
    X_train, y_train, X_val, y_val = splits[:4]
    before = X_val.copy()
    resample(X_train, y_train, "smote_nc")
    pd.testing.assert_frame_equal(before, X_val)


def test_none_and_class_weight_leave_data_unchanged(splits):
    X_train, y_train = splits[0], splits[1]
    for method in ("none", "class_weight"):
        Xr, yr = resample(X_train, y_train, method)
        assert Xr is X_train and yr is y_train


def test_class_weight_kwargs_per_library():
    y = np.array([0, 0, 0, 1])
    assert class_weight_kwargs("xgb", "class_weight", y) == {"scale_pos_weight": 3.0}
    assert class_weight_kwargs("catboost", "class_weight", y) == {"auto_class_weights": "Balanced"}
    assert class_weight_kwargs("knn", "class_weight", y) == {}
    assert class_weight_kwargs("logreg", "none", y) == {}
    w = keras_class_weight(y, "class_weight")
    assert w[1] / w[0] == pytest.approx(3.0)


def test_selection_rule_prefers_simplest_when_close():
    results = pd.DataFrame(
        {
            "method": ["none", "class_weight", "smote_nc", "undersample"],
            "pr_auc_mean": [0.650, 0.655, 0.660, 0.600],
            "pr_auc_std": [0.02, 0.02, 0.02, 0.02],
        }
    )
    choice = choose_method(results)
    assert choice["best_by_pr_auc"] == "smote_nc"
    assert choice["method"] == "class_weight"


def test_selection_rule_keeps_clear_winner():
    results = pd.DataFrame(
        {
            "method": ["none", "class_weight", "smote_nc", "undersample"],
            "pr_auc_mean": [0.60, 0.61, 0.70, 0.59],
            "pr_auc_std": [0.01, 0.01, 0.01, 0.01],
        }
    )
    assert choose_method(results)["method"] == "smote_nc"
