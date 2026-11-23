"""Έλεγχοι της επεξήγησης: η επεξήγηση στα αρχικά χαρακτηριστικά δίνει ίδιες προβλέψεις και μόνο έγκυρους πελάτες."""

import numpy as np
import pytest
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression

from src.config import CATEGORICAL, FEATURES
from src.data import load_splits
from src.explain import OriginalSpace, predict_from_original
from src.imbalance import make_pipeline


@pytest.fixture(scope="module", params=["full", "merged_no_total"])
def setup(request):
    X_train, y_train, _, _, X_test, _ = load_splits()
    pipe = make_pipeline(LogisticRegression(max_iter=2000), "none", variant=request.param).fit(X_train, y_train)
    space = OriginalSpace(X_train, request.param)
    return X_train, X_test, pipe, space


def test_round_trip_gives_identical_predictions(setup):
    """Κωδικοί → πελάτες → pipeline δίνει ακριβώς ό,τι το pipeline στους αρχικούς πελάτες."""
    X_train, X_test, pipe, space = setup
    rows = X_test.head(50)
    direct = pipe.predict_proba(rows[FEATURES])[:, 1]
    via_codes = predict_from_original(space.to_codes(rows), space, lambda df: pipe.predict_proba(df)[:, 1])
    assert np.max(np.abs(direct - via_codes)) < 1e-6


def test_lime_perturbations_are_valid_customers(setup):
    """Κάθε παραλλαγή που φτιάχνει το LIME έχει μόνο κατηγορίες που υπάρχουν στα δεδομένα."""
    X_train, X_test, pipe, space = setup
    seen = []

    def predict(Z):
        seen.append(Z)
        p = predict_from_original(Z, space, lambda df: pipe.predict_proba(df)[:, 1])
        return np.column_stack([1 - p, p])

    Z_train = space.to_codes(X_train)
    explainer = LimeTabularExplainer(Z_train, feature_names=space.features, categorical_features=space.cat_idx,
                                     categorical_names=space.category_names(), discretize_continuous=True,
                                     random_state=0)
    explainer.explain_instance(space.to_codes(X_test.head(1))[0], predict, num_features=5, num_samples=100)
    decoded = space.to_frame(np.vstack(seen))
    for col in CATEGORICAL:
        assert set(decoded[col].unique()) <= set(X_train[col].unique())
    assert set(decoded["SeniorCitizen"].unique()) <= {0, 1}


def test_model_view_is_idempotent(setup):
    """Η παραλλαγή κωδικοποίησης εφαρμόζεται με ασφάλεια και σε πελάτες που την έχουν ήδη περάσει."""
    X_train, X_test, pipe, space = setup
    once = space.model_view(X_test.head(20))
    assert np.array_equal(space.to_codes(once), space.to_codes(X_test.head(20)))
