"""Έλεγχοι των δεδομένων: καθαρισμός και διαχωρισμός των δεδομένων."""

import pandas as pd
import pytest

from src.config import FEATURES, TARGET
from src.data import clean, load_raw, make_splits


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def cleaned(raw):
    return clean(raw)


def test_row_count_and_no_missing(cleaned):
    """Καμία γραμμή δεν χάνεται και δεν μένει κανένα κενό."""
    assert len(cleaned) == 7043
    assert not cleaned.isna().any().any()
    assert list(cleaned.columns) == FEATURES + [TARGET]


def test_blank_total_charges_are_new_customers(raw, cleaned):
    """Ακριβώς 11 κενά TotalCharges, όλα σε πελάτες με tenure 0, που έγιναν 0."""
    blank = pd.to_numeric(raw["TotalCharges"].str.strip(), errors="coerce").isna()
    assert blank.sum() == 11
    assert (raw.loc[blank, "tenure"] == 0).all()
    assert (cleaned.loc[blank, "TotalCharges"] == 0).all()


def test_target_is_binary(cleaned):
    assert set(cleaned[TARGET].unique()) == {0, 1}


def test_splits_are_disjoint_and_complete(cleaned):
    """Τα τρία σύνολα δεν επικαλύπτονται και μαζί καλύπτουν όλες τις γραμμές."""
    s = make_splits(cleaned, save=False)
    train, val, test = set(s["train"]), set(s["val"]), set(s["test"])
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(cleaned.index)


def test_splits_are_stratified(cleaned):
    """Το ποσοστό αποχώρησης κάθε συνόλου απέχει λιγότερο από 1 μονάδα από το συνολικό."""
    s = make_splits(cleaned, save=False)
    overall = cleaned[TARGET].mean()
    for name in ("train", "val", "test"):
        assert abs(cleaned.loc[s[name], TARGET].mean() - overall) < 0.01


def test_split_sizes(cleaned):
    s = make_splits(cleaned, save=False)
    n = len(cleaned)
    assert abs(len(s["test"]) / n - 0.15) < 0.005
    assert abs(len(s["val"]) / n - 0.15) < 0.005


def test_splits_are_reproducible(cleaned):
    """Με το ίδιο seed ο διαχωρισμός είναι πάντα ο ίδιος."""
    assert make_splits(cleaned, save=False) == make_splits(cleaned, save=False)
