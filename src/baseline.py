"""Baseline: Logistic Regression (ερώτημα 2).

Ένα απλό, ερμηνεύσιμο γραμμικό μοντέλο, με το οποίο συγκρίνεται το νευρωνικό. Οι συντελεστές του
δίνουν και μια πρώτη εικόνα για το ποια χαρακτηριστικά συνδέονται με την αποχώρηση.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src import labels
from src.config import MODELS_DIR, SEED, TABLES_DIR
from src.data import load_splits
from src.evaluation import (
    compute_metrics,
    grid_edge_rows,
    save_model_metrics,
    save_predictions,
    tune_threshold,
    update_grid_edge_report,
    search_flatness_row,
    update_search_flatness,
)
from src.imbalance import class_weight_kwargs, load_choice, make_pipeline
from src.plotting import COLOR_CHURN, COLOR_STAY, save_figure
from src.preprocessing import feature_names

MODEL_NAME = "logreg"

# Πλέγμα αναζήτησης: ένταση κανονικοποίησης C και είδος ποινής.
# Από το scikit-learn 1.8, η ποινή ορίζεται με l1_ratio: 0 = L2 (ridge), 1 = L1 (lasso).
# Σε πρώτη δοκιμή, με πλέγμα [0,01–10], το καλύτερο C ήταν 10, στην άκρη του πλέγματος· γι' αυτό επεκτάθηκε.
PARAM_GRID = {"model__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000], "model__l1_ratio": [0.0, 1.0]}


def fit_logreg(X_train, y_train, method: str) -> GridSearchCV:
    """Εκπαιδεύει τη Logistic Regression με grid search (5-fold CV στο train, κριτήριο ROC-AUC).

    Ο solver «saga» υποστηρίζει και L1 και L2 ποινή. Το max_iter είναι μεγάλο, ώστε να συγκλίνει πάντα.
    """
    model = LogisticRegression(solver="saga", max_iter=5000, random_state=SEED,
                               **class_weight_kwargs(MODEL_NAME, method, y_train))
    search = GridSearchCV(
        make_pipeline(model, method),
        PARAM_GRID,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        n_jobs=-1,
    )
    return search.fit(X_train, y_train)


def coefficient_table(pipeline) -> pd.DataFrame:
    """Οι συντελεστές του μοντέλου, ταξινομημένοι κατά απόλυτη τιμή.

    Το odds ratio = e^συντελεστής δείχνει πόσο πολλαπλασιάζονται οι «πιθανότητες» αποχώρησης όταν η
    μεταβλητή αυξάνεται κατά 1. Για τις αριθμητικές μεταβλητές, που είναι κανονικοποιημένες, αυτό
    σημαίνει αύξηση κατά μία τυπική απόκλιση· για τις one-hot, ότι ο πελάτης ανήκει σε αυτή την κατηγορία.
    """
    names = feature_names(pipeline.named_steps["prep"])
    coefs = pipeline.named_steps["model"].coef_.ravel()
    table = pd.DataFrame({"Χαρακτηριστικό": names, "Συντελεστής": coefs, "Λόγος σχετικών πιθανοτήτων": np.exp(coefs)})
    order = table["Συντελεστής"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True).round(4)


def plot_coefficients(table: pd.DataFrame, top: int = 15):
    """Οι `top` μεγαλύτεροι συντελεστές: πορτοκαλί αυξάνουν τον κίνδυνο αποχώρησης, μπλε τον μειώνουν."""
    data = table.head(top).iloc[::-1]
    colors = [COLOR_CHURN if c > 0 else COLOR_STAY for c in data["Συντελεστής"]]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([labels.encoded_name(f) for f in data["Χαρακτηριστικό"]], data["Συντελεστής"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Συντελεστής (θετικός = μεγαλύτερος κίνδυνος αποχώρησης)")
    ax.set_title(f"Logistic Regression: οι {top} μεγαλύτεροι συντελεστές")
    return save_figure(fig, "logreg_coefficients")


def run() -> None:
    """Εκπαίδευση, επιλογή threshold στο validation, αποθήκευση προβλέψεων, μετρικών και συντελεστών."""
    X_train, y_train, X_val, y_val, *_ = load_splits()
    method = load_choice()

    search = fit_logreg(X_train, y_train, method)
    pipeline = search.best_estimator_

    prob_train = pipeline.predict_proba(X_train)[:, 1]
    prob_val = pipeline.predict_proba(X_val)[:, 1]
    threshold = tune_threshold(y_val, prob_val)

    save_predictions(MODEL_NAME, "train", X_train.index, y_train, prob_train)
    save_predictions(MODEL_NAME, "val", X_val.index, y_val, prob_val)
    val_metrics = {
        "at_0.5": compute_metrics(y_val, prob_val, 0.5),
        "at_tuned": compute_metrics(y_val, prob_val, threshold),
    }
    params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    update_grid_edge_report(MODEL_NAME, grid_edge_rows(MODEL_NAME, PARAM_GRID, params, threshold))
    update_search_flatness(search_flatness_row(MODEL_NAME, search))
    save_model_metrics(MODEL_NAME, threshold, val_metrics, params,
                       extra={"cv_roc_auc": float(search.best_score_), "imbalance_method": method})

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / f"{MODEL_NAME}.joblib")

    table = coefficient_table(pipeline)
    table.to_csv(TABLES_DIR / "logreg_coefficients.csv", index=False)
    plot_coefficients(table)

    m = val_metrics["at_tuned"]
    print(f"Καλύτερες παράμετροι: {params} (CV ROC-AUC {search.best_score_:.4f})")
    print(f"Validation: ROC-AUC {m['roc_auc']:.4f}, F1 {m['f1']:.4f} στο threshold {threshold:.2f}")
    print(table.head(5).to_string(index=False))
