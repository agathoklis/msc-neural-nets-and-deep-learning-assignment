"""Πείραμα κωδικοποίησης των περιττών κατηγοριών (ερώτημα 1).

Η ανάλυση των σχέσεων ανάμεσα στα χαρακτηριστικά (EDA) έδειξε ότι:
- το «No internet service» στις 6 πρόσθετες υπηρεσίες είναι ακριβές αντίγραφο του `InternetService = No`·
- το «No phone service» στο `MultipleLines` είναι ακριβές αντίγραφο του `PhoneService = No`·
- το `TotalCharges` είναι σχεδόν ακριβώς `tenure × MonthlyCharges` (r = 0,9996).

Εδώ συγκρίνονται τρεις τρόποι κωδικοποίησης και επιλέγεται ένας, με κανόνα που ορίζεται από πριν:
- `full`: η αρχική κωδικοποίηση (40 στήλες μετά την κωδικοποίηση)·
- `merged`: οι περιττές κατηγορίες γίνονται «No»· η πληροφορία μένει στα `InternetService` / `PhoneService`·
- `merged_no_total`: ό,τι το `merged`, και επιπλέον αφαιρείται το `TotalCharges`.

Η μετατροπή γίνεται στις αρχικές στήλες, μέσα στο pipeline κάθε μοντέλου, οπότε ισχύει για όλα (και για το CatBoost).
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from src.config import METRICS_DIR, SEED, TABLES_DIR

ENCODING_VARIANTS = ["full", "merged", "merged_no_total"]
DEFAULT_VARIANT = "full"

# Από το λιγότερο προς το περισσότερο απλό (λιγότερα χαρακτηριστικά = απλούστερο μοντέλο).
SIMPLICITY_ORDER = ["merged_no_total", "merged", "full"]

VARIANT_LABELS = {
    "full": "Πλήρης",
    "merged": "Συγχώνευση",
    "merged_no_total": "Συγχώνευση χωρίς συνολικές χρεώσεις",
}

INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                   "StreamingMovies"]

CHOICE_PATH = METRICS_DIR / "encoding_choice.json"


def apply_encoding_variant(X: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Εφαρμόζει μια παραλλαγή κωδικοποίησης στις αρχικές (καθαρισμένες) στήλες. Επιστρέφει νέο DataFrame.

    Στο `merged` καμία πληροφορία δεν χάνεται: το «ο πελάτης δεν έχει internet» μένει στο
    `InternetService = No`, και το «δεν έχει τηλέφωνο» στο `PhoneService = No`.
    """
    if variant not in ENCODING_VARIANTS:
        raise ValueError(f"Άγνωστη παραλλαγή κωδικοποίησης: {variant}")
    X = X.copy()
    if variant in ("merged", "merged_no_total"):
        for col in INTERNET_ADDONS:
            X[col] = X[col].replace("No internet service", "No")
        X["MultipleLines"] = X["MultipleLines"].replace("No phone service", "No")
    if variant == "merged_no_total":
        # errors="ignore": η μετατροπή εφαρμόζεται με ασφάλεια και σε δεδομένα όπου έχει ήδη εφαρμοστεί.
        X = X.drop(columns=["TotalCharges"], errors="ignore")
    return X


def load_choice() -> str:
    """Η παραλλαγή που επέλεξε το πείραμα. Πριν τρέξει το πείραμα, ισχύει η αρχική `full`."""
    if CHOICE_PATH.exists():
        return json.loads(CHOICE_PATH.read_text())["variant"]
    return DEFAULT_VARIANT


# --- Πείραμα ------------------------------------------------------------------------------------------

# Το πείραμα τρέχει με class weights, τη μέθοδο ανισορροπίας της πρώτης ανάλυσης. Στο pipeline η
# ανάλυση ανισορροπίας (στάδιο `imbalance`) τρέχει αμέσως μετά, με την παραλλαγή που επιλέγεται εδώ,
# και επιβεβαιώνει ότι η επιλογή της δεν αλλάζει.
EXPERIMENT_IMBALANCE_METHOD = "class_weight"


def _models(y_train) -> dict:
    y = np.asarray(y_train)
    return {
        "logreg": LogisticRegression(max_iter=5000, class_weight="balanced"),
        "xgb": XGBClassifier(eval_metric="auc", random_state=SEED, n_jobs=1,
                             scale_pos_weight=float((y == 0).sum() / (y == 1).sum())),
    }


def compare_variants(X_train, y_train) -> pd.DataFrame:
    """Logistic Regression και XGBoost με τις 3 παραλλαγές, με 5-fold stratified CV μόνο στο train."""
    from src.imbalance import make_pipeline

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1", "recall": "recall"}
    rows = []
    for model_name, model in _models(y_train).items():
        for variant in ENCODING_VARIANTS:
            pipe = make_pipeline(model, EXPERIMENT_IMBALANCE_METHOD, variant=variant)
            scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            from src.preprocessing import build_preprocessor

            n_features = build_preprocessor(variant).fit(X_train).transform(X_train.head(1)).shape[1]
            row = {"model": model_name, "variant": variant, "Παραλλαγή": VARIANT_LABELS[variant],
                   "n_features": n_features}
            for metric in scoring:
                row[f"{metric}_mean"] = float(scores[f"test_{metric}"].mean())
                row[f"{metric}_std"] = float(scores[f"test_{metric}"].std())
            rows.append(row)
    return pd.DataFrame(rows)


def choose_variant(results: pd.DataFrame) -> dict:
    """Κανόνας επιλογής (ορίστηκε πριν από το πείραμα): μεγαλύτερο μέσο PR-AUC της Logistic Regression· αν άλλες παραλλαγές απέχουν
    λιγότερο από μία τυπική απόκλιση, κρατάμε αυτή με τα λιγότερα χαρακτηριστικά. Το XGBoost είναι έλεγχος."""
    lr = results[results["model"] == "logreg"].set_index("variant")
    best = lr["pr_auc_mean"].idxmax()
    tolerance = lr.loc[best, "pr_auc_std"]
    close = [v for v in lr.index if lr.loc[v, "pr_auc_mean"] >= lr.loc[best, "pr_auc_mean"] - tolerance]
    chosen = next(v for v in SIMPLICITY_ORDER if v in close)

    xgb = results[results["model"] == "xgb"].set_index("variant")
    xgb_gap = xgb["pr_auc_mean"].max() - xgb.loc[chosen, "pr_auc_mean"]
    xgb_agrees = bool(xgb_gap <= xgb.loc[chosen, "pr_auc_std"])

    if chosen == best:
        reason = f"Μεγαλύτερο μέσο PR-AUC στη Logistic Regression ({lr.loc[best, 'pr_auc_mean']:.4f})."
    else:
        reason = (f"Το μεγαλύτερο μέσο PR-AUC το έδωσε η «{best}» ({lr.loc[best, 'pr_auc_mean']:.4f}), αλλά οι "
                  f"παραλλαγές {close} απέχουν λιγότερο από μία τυπική απόκλιση ({tolerance:.4f}). Επιλέχθηκε αυτή "
                  f"με τα λιγότερα χαρακτηριστικά.")
    reason += (" Το XGBoost συμφωνεί (διαφορά μέσα σε μία τυπική απόκλιση)." if xgb_agrees
               else f" Προσοχή: στο XGBoost η επιλογή υστερεί κατά {xgb_gap:.4f} PR-AUC, περισσότερο από μία τυπική απόκλιση.")
    return {"variant": chosen, "n_features": int(lr.loc[chosen, "n_features"]), "best_by_pr_auc": best,
            "within_one_std": close, "xgb_agrees": xgb_agrees, "reason": reason}


def plot_comparison(results: pd.DataFrame):
    """Δύο πάνελ (Logistic Regression, XGBoost): ROC-AUC και PR-AUC ανά παραλλαγή, με ±1 τυπική απόκλιση."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    metrics = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]
    width = 0.25
    for ax, (model, title) in zip(axes, [("logreg", "Logistic Regression"), ("xgb", "XGBoost")]):
        sub = results[results["model"] == model].set_index("variant")
        x = np.arange(len(metrics))
        for i, variant in enumerate(ENCODING_VARIANTS):
            ax.bar(x + (i - 1) * width, [sub.loc[variant, f"{m}_mean"] for m, _ in metrics], width,
                   yerr=[sub.loc[variant, f"{m}_std"] for m, _ in metrics], capsize=3,
                   label=f"{VARIANT_LABELS[variant]} ({sub.loc[variant, 'n_features']} στήλες)")
        ax.set_xticks(x, [lbl for _, lbl in metrics])
        ax.set_ylim(0.5, 0.9)
        ax.set_title(title)
    axes[0].set_ylabel("Τιμή μετρικής (μέσος όρος 5 πτυχών)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.1), ncol=3, fontsize=8)
    fig.suptitle("Σύγκριση παραλλαγών κωδικοποίησης (διασταυρωμένη επικύρωση στο σύνολο εκπαίδευσης)", y=1.02)
    from src.plotting import save_figure

    return save_figure(fig, "encoding_comparison")


def run() -> None:
    from src.data import load_splits

    X_train, y_train, *_ = load_splits()
    results = compare_variants(X_train, y_train)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(TABLES_DIR / "encoding_comparison.csv", index=False)
    plot_comparison(results)
    choice = choose_variant(results)
    CHOICE_PATH.write_text(json.dumps(choice, ensure_ascii=False, indent=2))
    print(results[["model", "variant", "n_features", "roc_auc_mean", "pr_auc_mean", "pr_auc_std"]].round(4)
          .to_string(index=False))
    print(f"Επιλογή: {choice['variant']} ({choice['n_features']} χαρακτηριστικά) — {choice['reason']}")
