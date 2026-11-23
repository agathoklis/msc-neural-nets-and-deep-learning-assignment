"""Ανάλυση και χειρισμός της ανισορροπίας κλάσεων.

Μόνο το ~26,5% των πελατών αποχωρεί. Εδώ συγκρίνονται τέσσερις τρόποι χειρισμού και επιλέγεται ένας,
με κανόνα που ορίζεται από πριν. Η επιλογή αποθηκεύεται, και **όλα** τα μοντέλα την εφαρμόζουν μέσα
από τις συναρτήσεις αυτού του module, ώστε να εκπαιδεύονται με τον ίδιο τρόπο.

Η επαναδειγματοληψία (SMOTE-NC, undersampling) γίνεται πάντα μέσα σε imblearn Pipeline. Έτσι, στο
cross-validation εφαρμόζεται μόνο στα folds εκπαίδευσης, και οι συνθετικοί πελάτες δεν μπαίνουν ποτέ
στα δεδομένα αξιολόγησης.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import BINARY_PASSTHROUGH, CATEGORICAL, FEATURES, METRICS_DIR, NUMERIC, SEED, TABLES_DIR
from src.data import load_splits
from src.plotting import save_figure
from src.preprocessing import build_preprocessor

IMBALANCE_METHODS = ["none", "class_weight", "smote_nc", "undersample"]

# Σειρά «απλότητας»: αν οι καλύτερες μέθοδοι δεν διαφέρουν ουσιαστικά, κρατάμε την πιο απλή.
# Τα class weights δεν αλλάζουν τα δεδομένα· το undersampling αφαιρεί δεδομένα· το SMOTE-NC φτιάχνει
# συνθετικά, άρα είναι η πιο «παρεμβατική» μέθοδος.
SIMPLICITY_ORDER = ["class_weight", "none", "undersample", "smote_nc"]

METHOD_LABELS = {
    "none": "Κανένας χειρισμός",
    "class_weight": "Βάρη κλάσεων",
    "smote_nc": "SMOTE-NC",
    "undersample": "Undersampling",
}

METRIC_LABELS = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1", "recall": "Ανάκληση"}

CHOICE_PATH = METRICS_DIR / "imbalance_choice.json"

# Μοντέλα χωρίς παράμετρο βαρών: γι' αυτά τα class weights δεν εφαρμόζονται, και η ανισορροπία
# αντιμετωπίζεται μόνο με το threshold που επιλέγεται στο validation.
NO_WEIGHT_MODELS = {"knn", "nb"}


# --- Εφαρμογή της επιλογής σε όλα τα μοντέλα ------------------------------------------------------


def load_choice() -> str:
    """Η μέθοδος που επέλεξε η ανάλυση ανισορροπίας. Όλα τα μοντέλα τη διαβάζουν από εδώ."""
    return json.loads(CHOICE_PATH.read_text())["method"]


def sampler(method: str):
    """Ο sampler της μεθόδου, ή None αν η μέθοδος δεν αλλάζει τα δεδομένα."""
    if method == "smote_nc":
        # Το απλό SMOTE θα «ανακάτευε» one-hot τιμές (π.χ. 0,4 «Two year»). Το SMOTE-NC χειρίζεται
        # χωριστά τις κατηγορικές στήλες και δίνει μόνο έγκυρες κατηγορίες.
        categorical_idx = [FEATURES.index(c) for c in BINARY_PASSTHROUGH + CATEGORICAL]
        return SMOTENC(categorical_features=categorical_idx, random_state=SEED)
    if method == "undersample":
        return RandomUnderSampler(random_state=SEED)
    return None


def class_weight_kwargs(model_name: str, method: str, y) -> dict:
    """Οι παράμετροι βαρών κάθε μοντέλου για τη μέθοδο `class_weight` (κενό λεξικό για τις άλλες).

    Κάθε βιβλιοθήκη ονομάζει διαφορετικά το ίδιο πράγμα: «τα λάθη στους churners μετράνε περισσότερο».
    """
    if method != "class_weight" or model_name in NO_WEIGHT_MODELS:
        return {}
    if model_name == "xgb":
        y = np.asarray(y)
        return {"scale_pos_weight": float((y == 0).sum() / (y == 1).sum())}
    if model_name == "catboost":
        return {"auto_class_weights": "Balanced"}
    return {"class_weight": "balanced"}  # logreg, rf, svm, lgbm


def keras_class_weight(y, method: str) -> dict | None:
    """Τα βάρη κλάσεων για το Keras, με τον ίδιο τύπο που χρησιμοποιεί το «balanced» του scikit-learn."""
    if method != "class_weight":
        return None
    y = np.asarray(y)
    n, counts = len(y), np.bincount(y)
    return {cls: float(n / (2 * counts[cls])) for cls in (0, 1)}


def make_pipeline(estimator, method: str, preprocess: bool = True, variant: str | None = None) -> Pipeline:
    """Pipeline [sampler] → preprocessor (ή μόνο η παραλλαγή κωδικοποίησης) → μοντέλο.

    Ο sampler μπαίνει *πριν* από την προεπεξεργασία: το SMOTE-NC χρειάζεται όλες τις αρχικές στήλες στη
    σειρά του `FEATURES`. Τα μοντέλα χωρίς κοινό preprocessor (CatBoost) παίρνουν τουλάχιστον το βήμα της
    παραλλαγής κωδικοποίησης, ώστε όλα τα μοντέλα να βλέπουν τα ίδια δεδομένα.
    """
    from src.encoding import load_choice as load_encoding_choice
    from src.preprocessing import variant_step

    steps = []
    s = sampler(method)
    if s is not None:
        steps.append(("sampler", s))
    variant = variant or load_encoding_choice()
    steps.append(("variant", variant_step(variant)))
    if preprocess:
        # Το imblearn δεν δέχεται pipeline μέσα σε pipeline, οπότε τα δύο βήματα του preprocessor μπαίνουν
        # χωριστά: «variant» (παραλλαγή κωδικοποίησης) και «prep» (κανονικοποίηση και one-hot).
        steps.append(("prep", build_preprocessor(variant).named_steps["encode"]))
    steps.append(("model", estimator))
    return Pipeline(steps)


def resample(X, y, method: str):
    """Επαναδειγματοληψία των δεδομένων εκπαίδευσης, για μοντέλα εκτός Pipeline (το νευρωνικό).

    Καλείται μόνο με δεδομένα train. Τα validation και test δεν αλλάζουν ποτέ.
    """
    s = sampler(method)
    if s is None:
        return X, y
    return s.fit_resample(X, y)


# --- Μέρος Α: επιχείρημα από τα δεδομένα -----------------------------------------------------------


def imbalance_facts(y_train, X_train=None) -> dict:
    """Τα νούμερα που στηρίζουν το επιχείρημα της ενότητας 2.3, χωρίς κανένα μοντέλο."""
    y = np.asarray(y_train)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    n_cat = len(BINARY_PASSTHROUGH + CATEGORICAL)
    return {
        "train_rows": int(len(y)),
        "train_churn": n_pos,
        "train_stay": n_neg,
        "ratio_stay_per_churn": round(n_neg / n_pos, 3),
        "undersample_discards": n_neg - n_pos,
        "undersample_discards_share": round((n_neg - n_pos) / len(y), 4),
        "smote_nc_creates": n_neg - n_pos,
        "input_columns_categorical": n_cat,
        "input_columns_numeric": len(NUMERIC),
        # Μετά το one-hot, όλες οι στήλες εκτός από τις αριθμητικές είναι 0/1: εκεί το απλό SMOTE
        # θα έβγαζε μη έγκυρες τιμές.
        "encoded_columns_binary": _n_binary_encoded(X_train),
    }


def _n_binary_encoded(X_train) -> int | None:
    """Πλήθος στηλών 0/1 μετά την προεπεξεργασία της επιλεγμένης παραλλαγής κωδικοποίησης."""
    if X_train is None:
        return None
    prep = build_preprocessor().fit(X_train)
    n_numeric = len(prep.named_steps["encode"].transformers_[0][2])
    return int(prep.transform(X_train.head(1)).shape[1] - n_numeric)


# --- Μέρος Β: πείραμα -----------------------------------------------------------------------------


def compare_methods(X_train, y_train) -> pd.DataFrame:
    """Logistic Regression με τις 4 μεθόδους, με 5-fold stratified CV μόνο στο train.

    Το F1 και το recall μετρώνται στο threshold 0,5. Εκεί φαίνεται η επίδραση της ανισορροπίας, αφού
    χωρίς χειρισμό το μοντέλο τείνει να προβλέπει «παραμένει».
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1", "recall": "recall"}
    rows = []
    for method in IMBALANCE_METHODS:
        model = LogisticRegression(max_iter=5000, **class_weight_kwargs("logreg", method, y_train))
        scores = cross_validate(make_pipeline(model, method), X_train, y_train, cv=cv, scoring=scoring)
        row = {"method": method}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        rows.append(row)
    return pd.DataFrame(rows)


def choose_method(results: pd.DataFrame) -> dict:
    """Εφαρμόζει τον κανόνα επιλογής (ορίστηκε πριν από το πείραμα) και επιστρέφει τη μέθοδο με την αιτιολόγησή της.

    Κερδίζει το μεγαλύτερο μέσο PR-AUC. Αν όμως άλλες μέθοδοι απέχουν λιγότερο από μία τυπική απόκλιση
    από την καλύτερη, η διαφορά δεν είναι ουσιαστική και κρατάμε την απλούστερη από αυτές.
    """
    best = results.loc[results["pr_auc_mean"].idxmax()]
    tolerance = best["pr_auc_std"]
    close = results[results["pr_auc_mean"] >= best["pr_auc_mean"] - tolerance]["method"].tolist()
    chosen = next(m for m in SIMPLICITY_ORDER if m in close)
    if chosen == best["method"]:
        reason = f"Μεγαλύτερο μέσο PR-AUC ({best['pr_auc_mean']:.4f})."
    else:
        reason = (
            f"Το μεγαλύτερο μέσο PR-AUC το έδωσε η «{best['method']}» ({best['pr_auc_mean']:.4f}), αλλά οι μέθοδοι "
            f"{close} απέχουν λιγότερο από μία τυπική απόκλιση ({tolerance:.4f}). Επιλέχθηκε η απλούστερη από αυτές."
        )
    return {"method": chosen, "best_by_pr_auc": best["method"], "within_one_std": close, "reason": reason}


def plot_comparison(results: pd.DataFrame):
    """Ομαδοποιημένες μπάρες: μία ομάδα ανά μετρική, μία μπάρα ανά μέθοδο, με σφάλμα ±1 τυπική απόκλιση."""
    metrics = list(METRIC_LABELS)
    x = np.arange(len(metrics))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, method in enumerate(IMBALANCE_METHODS):
        row = results.set_index("method").loc[method]
        ax.bar(x + (i - 1.5) * width, [row[f"{m}_mean"] for m in metrics], width,
               yerr=[row[f"{m}_std"] for m in metrics], capsize=3, label=METHOD_LABELS[method])
    ax.set_xticks(x, [METRIC_LABELS[m] for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Τιμή μετρικής (μέσος όρος 5 πτυχών)")
    ax.set_title("Σύγκριση μεθόδων χειρισμού της ανισορροπίας\n(Logistic Regression, διασταυρωμένη επικύρωση στο σύνολο εκπαίδευσης)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    return save_figure(fig, "imbalance_comparison")


def run() -> None:
    """Τρέχει την ανάλυση ανισορροπίας: γεγονότα, πείραμα, επιλογή, πίνακας και σχήμα."""
    X_train, y_train, *_ = load_splits()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    facts = imbalance_facts(y_train, X_train)
    (TABLES_DIR / "imbalance_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2))

    results = compare_methods(X_train, y_train)
    results.round(4).to_csv(TABLES_DIR / "imbalance_comparison.csv", index=False)
    plot_comparison(results)

    choice = choose_method(results)
    CHOICE_PATH.write_text(json.dumps(choice, ensure_ascii=False, indent=2))

    print(results.round(4).to_string(index=False))
    print(f"Επιλογή: {choice['method']} — {choice['reason']}")
