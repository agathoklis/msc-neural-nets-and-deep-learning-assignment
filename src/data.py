"""Φόρτωση, καθαρισμός και διαχωρισμός των δεδομένων.

Ο διαχωρισμός σε train/validation/test γίνεται μία φορά και οι δείκτες αποθηκεύονται σε αρχείο.
Έτσι όλα τα μοντέλα, σε όποια στιγμή κι αν τρέξουν, εκπαιδεύονται και αξιολογούνται στις ίδιες ακριβώς
γραμμές, και η σύγκρισή τους είναι δίκαιη.
"""

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    FEATURES,
    ID_COLUMN,
    SEED,
    SPLITS_PATH,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
)


def load_raw() -> pd.DataFrame:
    """Διαβάζει το αρχικό CSV χωρίς καμία αλλαγή."""
    return pd.read_csv(DATA_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Καθαρίζει τα δεδομένα με τρεις αποφάσεις που ισχύουν για όλα τα μοντέλα:

    - Αφαιρεί το `customerID`: είναι αναγνωριστικό και δεν λέει τίποτα για τη συμπεριφορά του πελάτη.
    - Μετατρέπει το `TotalCharges` σε αριθμό. Στο CSV είναι κείμενο, επειδή 11 γραμμές έχουν κενό.
      Και οι 11 είναι πελάτες με `tenure == 0`, δηλαδή νέοι πελάτες που δεν έχουν χρεωθεί ακόμα,
      οπότε η σωστή τιμή είναι 0. Αν τις αφαιρούσαμε, θα χάναμε όλους τους νέους πελάτες.
    - Κωδικοποιεί τον στόχο `Churn` ως 1 (αποχώρησε) / 0 (παρέμεινε).
    """
    df = df.drop(columns=[ID_COLUMN]).copy()
    total = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    blank = total.isna()
    # Έλεγχος της υπόθεσης πριν τη χρησιμοποιήσουμε: αν κάποιο κενό δεν είχε tenure 0,
    # το «βάζουμε 0» δεν θα ήταν πια σωστό και πρέπει να το ξέρουμε.
    if not (df.loc[blank, "tenure"] == 0).all():
        raise ValueError("Υπάρχουν κενά TotalCharges σε πελάτες με tenure > 0.")
    df["TotalCharges"] = total.fillna(0.0)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df[FEATURES + [TARGET]]


def make_splits(df: pd.DataFrame, save: bool = True) -> dict[str, list[int]]:
    """Χωρίζει τα δεδομένα σε train/val/test (70/15/15) με διαστρωμάτωση ως προς τον στόχο.

    Η διαστρωμάτωση κρατά το ποσοστό αποχώρησης (~26,5%) ίδιο σε κάθε σύνολο. Γίνονται δύο διαδοχικοί
    διαχωρισμοί: πρώτα αφαιρείται το 15% για test, και μετά από το υπόλοιπο 85% κρατιέται το 15/85
    για validation, ώστε το validation να είναι κι αυτό 15% του συνόλου.
    """
    idx = df.index.to_numpy()
    rest, test = train_test_split(idx, test_size=TEST_SIZE, stratify=df[TARGET], random_state=SEED)
    val_fraction = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        rest, test_size=val_fraction, stratify=df.loc[rest, TARGET], random_state=SEED
    )
    splits = {"train": sorted(map(int, train)), "val": sorted(map(int, val)), "test": sorted(map(int, test))}
    if save:
        SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SPLITS_PATH.write_text(json.dumps(splits))
    return splits


def load_splits():
    """Επιστρέφει `(X_train, y_train, X_val, y_val, X_test, y_test)` από τους αποθηκευμένους δείκτες.

    Τα X είναι τα καθαρισμένα δεδομένα *πριν* από την κωδικοποίηση: κάθε μοντέλο περνά τα δικά του
    μέσα από τον preprocessor (ή, για το CatBoost, τα χρησιμοποιεί ως έχουν). Αν δεν υπάρχει ακόμα
    αρχείο διαχωρισμού, δημιουργείται.
    """
    df = clean(load_raw())
    if SPLITS_PATH.exists():
        splits = json.loads(SPLITS_PATH.read_text())
    else:
        splits = make_splits(df)
    parts = []
    for name in ("train", "val", "test"):
        part = df.loc[splits[name]]
        parts += [part[FEATURES], part[TARGET]]
    return tuple(parts)
