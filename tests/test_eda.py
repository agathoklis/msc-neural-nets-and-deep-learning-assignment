"""Έλεγχοι της περιγραφικής ανάλυσης: περιγραφή του dataset και σχέσεις ανάμεσα στα χαρακτηριστικά."""

import numpy as np

from src.data import clean, load_raw
from src.eda import (
    categorical_associations,
    churn_rate_by_category,
    cramers_v,
    dataset_description,
    dataset_facts,
    redundancy_checks,
)


def test_description_has_all_21_columns_with_greek():
    raw = load_raw()
    desc = dataset_description(raw)
    assert len(desc) == 21
    assert desc["Περιγραφή"].str.len().gt(0).all()
    notes = dict(zip(desc["Μεταβλητή"], desc["Σημείωση"]))
    assert notes["customerID"] == "αφαιρείται"
    assert notes["Churn"] == "μεταβλητή-στόχος"


def test_facts_counts():
    raw = load_raw()
    facts = dataset_facts(raw, clean(raw))
    assert facts["n_rows"] == 7043 and facts["n_features"] == 19
    assert facts["n_numeric"] + facts["n_binary"] + facts["n_categorical"] == 19
    assert facts["missing_before_cleaning"] == {"TotalCharges": 11}


def test_churn_rates_cover_all_categories():
    df = clean(load_raw())
    table = churn_rate_by_category(df)
    assert table["Μεταβλητή"].nunique() == 16
    # Τα μερίδια κάθε μεταβλητής αθροίζουν σε 1.
    assert (table.groupby("Μεταβλητή")["Μερίδιο"].sum().round(3) == 1).all()


def test_cramers_v_hand_computed():
    """2×2 πίνακας [[30, 10], [10, 30]]: χ² = 20, n = 80, άρα V = sqrt(20 / 80) = 0,5."""
    a = np.array(["x"] * 40 + ["y"] * 40)
    b = np.array(["p"] * 30 + ["q"] * 10 + ["p"] * 10 + ["q"] * 30)
    assert abs(cramers_v(a, b) - 0.5) < 1e-9
    assert abs(cramers_v(a, a) - 1.0) < 1e-9


def test_known_redundancies():
    df = clean(load_raw())
    checks = redundancy_checks(df)
    assert len(checks) == 9 and checks["Αποτέλεσμα"].all()
    assoc = categorical_associations(df).set_index(["feature_a", "feature_b"])["cramers_v"]
    assert assoc.loc[("PhoneService", "MultipleLines")] > 0.999
