"""Έλεγχοι της προεπεξεργασίας: σωστές στήλες ανά παραλλαγή κωδικοποίησης, χωρίς διαρροή από το test."""

import numpy as np
import pytest

from src.config import FEATURES, NUMERIC
from src.data import clean, load_raw, make_splits
from src.encoding import ENCODING_VARIANTS, apply_encoding_variant
from src.preprocessing import ENCODED_FEATURE_COUNTS, build_preprocessor, feature_names, original_feature


@pytest.fixture(scope="module")
def train_val():
    df = clean(load_raw())
    s = make_splits(df, save=False)
    return df.loc[s["train"], FEATURES], df.loc[s["val"], FEATURES]


@pytest.fixture(scope="module")
def fitted(train_val):
    return build_preprocessor("full").fit(train_val[0])


@pytest.mark.parametrize("variant", ENCODING_VARIANTS)
def test_output_shape_and_no_nans(variant, train_val):
    prep = build_preprocessor(variant).fit(train_val[0])
    for X in train_val:
        out = prep.transform(X)
        assert out.shape == (len(X), ENCODED_FEATURE_COUNTS[variant])
        assert not np.isnan(out).any()


def test_numeric_columns_scaled_on_train(fitted, train_val):
    """Στο train οι αριθμητικές στήλες έχουν μέσο ~0 και τυπική απόκλιση ~1."""
    out = fitted.transform(train_val[0])
    names = feature_names(fitted)
    for col in NUMERIC:
        values = out[:, names.index(col)]
        assert abs(values.mean()) < 1e-6
        assert abs(values.std() - 1) < 1e-3


def test_transform_is_deterministic(fitted, train_val):
    X = train_val[1]
    assert np.array_equal(fitted.transform(X), fitted.transform(X))


def test_original_feature_mapping(fitted):
    """Κάθε μία από τις 40 στήλες αντιστοιχεί σε ένα από τα 19 αρχικά χαρακτηριστικά, και όλα καλύπτονται."""
    names = feature_names(fitted)
    assert len(names) == ENCODED_FEATURE_COUNTS["full"]
    mapped = {original_feature(n) for n in names}
    assert mapped == set(FEATURES)
    assert original_feature("Contract_Two year") == "Contract"
    assert original_feature("MultipleLines_No phone service") == "MultipleLines"


def test_merged_variant_removes_duplicates_without_losing_information(train_val):
    """Το `merged` δεν αφήνει «No internet/phone service», και τα InternetService/PhoneService δεν αλλάζουν."""
    X = train_val[0]
    merged = apply_encoding_variant(X, "merged")
    from src.encoding import INTERNET_ADDONS

    for col in [*INTERNET_ADDONS, "MultipleLines"]:
        assert X[col].isin(["No internet service", "No phone service"]).any()  # υπήρχαν πριν
        assert not merged[col].isin(["No internet service", "No phone service"]).any()
    assert merged["InternetService"].equals(X["InternetService"])
    assert merged["PhoneService"].equals(X["PhoneService"])
    assert "TotalCharges" not in apply_encoding_variant(X, "merged_no_total").columns
    assert apply_encoding_variant(X, "full").equals(X)
