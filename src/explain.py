"""Επεξήγηση των μοντέλων με SHAP και LIME, και σύγκριση των δύο (ερωτήματα 3 και 4).

- **SHAP** (Lundberg & Lee, 2017): μοιράζει κάθε πρόβλεψη στα χαρακτηριστικά, με βάση τις τιμές Shapley της
  θεωρίας παιγνίων.
- **LIME** (Ribeiro et al., 2016): για κάθε πελάτη, φτιάχνει τυχαίες παραλλαγές του και ταιριάζει πάνω τους ένα
  απλό γραμμικό μοντέλο, που εξηγεί τοπικά την πρόβλεψη.

**Γιατί στα αρχικά χαρακτηριστικά.** Τα μοντέλα βλέπουν στήλες one-hot. Αν οι μέθοδοι άλλαζαν κάθε στήλη
one-hot χωριστά, θα έφτιαχναν μη ρεαλιστικούς πελάτες, π.χ. με δύο συμβόλαια ταυτόχρονα ή με κανένα. Εδώ οι μέθοδοι
δουλεύουν πάνω στα αρχικά χαρακτηριστικά: κάθε κατηγορική μεταβλητή είναι ένας ακέραιος κωδικός και αλλάζει
**ως μία μονάδα**, πάντα σε έγκυρη κατηγορία. Μόνο μέσα στη συνάρτηση πρόβλεψης ο πελάτης «μεταφράζεται» στην
κωδικοποίηση του μοντέλου.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import itertools

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.preprocessing import OrdinalEncoder

from src import labels
from src.config import BINARY_PASSTHROUGH, CATEGORICAL, FEATURES, MODELS_DIR, NUMERIC, SEED, TABLES_DIR
from src.data import load_splits
from src.encoding import apply_encoding_variant
from src.encoding import load_choice as load_encoding_choice
from src.plotting import COLOR_CHURN, COLOR_STAY, save_figure
from src.preprocessing import original_feature

N_BACKGROUND = 100   # πελάτες του train που ορίζουν την «αναμενόμενη» πρόβλεψη του SHAP
N_EXPLAIN = 500      # πελάτες του test που εξηγούνται
N_COMPARE = 20       # πελάτες για τη σύγκριση SHAP–LIME και τη σταθερότητα του LIME
LIME_SEEDS = (42, 43, 44, 45, 46)
TOP_K = 5
N_DIRECTION = 6      # πόσα χαρακτηριστικά δείχνει το σχήμα «κατεύθυνση ανά κατηγορία»


# --- Αναπαράσταση στα αρχικά χαρακτηριστικά -----------------------------------------------------------


class OriginalSpace:
    """Μετατροπή ανάμεσα στους πελάτες (DataFrame) και σε πίνακα αριθμών με ένα χαρακτηριστικό ανά στήλη.

    Οι κατηγορικές γίνονται ακέραιοι κωδικοί (OrdinalEncoder, εκπαιδευμένος στο train) και οι αριθμητικές
    μένουν ως έχουν. Χαρακτηριστικά που η επιλεγμένη κωδικοποίηση δεν χρησιμοποιεί (π.χ. το `TotalCharges`
    στο `merged_no_total`) δεν εξηγούνται.
    """

    def __init__(self, X_train: pd.DataFrame, variant: str):
        self.variant = variant
        self.dropped = ["TotalCharges"] if variant == "merged_no_total" else []
        self.features = [f for f in FEATURES if f not in self.dropped]
        self.categorical = [f for f in self.features if f in CATEGORICAL + BINARY_PASSTHROUGH]
        self.cat_idx = [self.features.index(f) for f in self.categorical]
        # Οι κατηγορίες είναι αυτές που βλέπει το μοντέλο, μετά την παραλλαγή κωδικοποίησης. Π.χ. με
        # `merged`, το «No internet service» έχει ήδη γίνει «No» και δεν εμφανίζεται ως ξεχωριστή κατηγορία.
        self.encoder = OrdinalEncoder().fit(self.model_view(X_train)[self.categorical].astype(str))

    def model_view(self, X: pd.DataFrame) -> pd.DataFrame:
        """Οι πελάτες όπως τους «βλέπει» το μοντέλο: με την παραλλαγή κωδικοποίησης εφαρμοσμένη."""
        # Δέχεται και πελάτες στους οποίους η παραλλαγή έχει ήδη εφαρμοστεί (π.χ. χωρίς το `TotalCharges`).
        return apply_encoding_variant(X[[f for f in FEATURES if f in X.columns]], self.variant)

    def to_codes(self, X: pd.DataFrame) -> np.ndarray:
        X = self.model_view(X)
        Z = X[self.features].copy()
        Z[self.categorical] = self.encoder.transform(X[self.categorical].astype(str))
        return Z.to_numpy(dtype=float)

    def to_frame(self, Z: np.ndarray) -> pd.DataFrame:
        """Αντίστροφα: από κωδικούς σε πελάτες με τις αρχικές τιμές (ό,τι περιμένουν τα pipelines)."""
        df = pd.DataFrame(np.asarray(Z, dtype=float), columns=self.features)
        codes = np.clip(np.rint(df[self.categorical].to_numpy()), 0, None)
        for j, col in enumerate(self.categorical):
            cats = self.encoder.categories_[j]
            values = cats[np.minimum(codes[:, j].astype(int), len(cats) - 1)]
            df[col] = values.astype(int) if col in BINARY_PASSTHROUGH else values
        for col in self.dropped:
            df[col] = 0.0  # δεν το βλέπει το μοντέλο (το αφαιρεί η παραλλαγή κωδικοποίησης)
        return df[FEATURES]

    def category_names(self) -> dict:
        return {i: list(map(str, self.encoder.categories_[j])) for j, i in enumerate(self.cat_idx)}


def predict_from_original(Z: np.ndarray, space: OriginalSpace, predict_df) -> np.ndarray:
    """Πιθανότητα αποχώρησης για πελάτες σε κωδικούς: κωδικοί → πελάτες → πλήρες pipeline του μοντέλου."""
    return np.asarray(predict_df(space.to_frame(Z))).ravel()


def model_predictors() -> dict:
    """Οι συναρτήσεις πρόβλεψης (πελάτες → πιθανότητα) του τελικού MLP (ensemble) και του XGBoost."""
    from src.nn_train import SeedEnsemble

    prep = joblib.load(MODELS_DIR / "nn_preprocessor.joblib")
    ensemble = SeedEnsemble.load("mlp_tuned")
    xgb = joblib.load(MODELS_DIR / "xgb.joblib")
    return {
        "nn": lambda df: ensemble.predict(prep.transform(df).astype("float32"), verbose=0, batch_size=4096),
        "xgb": lambda df: xgb.predict_proba(df)[:, 1],
    }


# --- Βοηθητικά --------------------------------------------------------------------------------------------


def _top_k(row: pd.Series, k: int = TOP_K) -> set:
    return set(row.abs().sort_values(ascending=False).head(k).index)


def _fmt(value) -> str:
    """Αριθμοί με ελληνική υποδιαστολή (95,1)· οι κατηγορίες μένουν όπως είναι στο dataset."""
    if isinstance(value, (float, np.floating)):
        return f"{value:g}".replace(".", ",")
    return str(value)


def _bar(importance: pd.Series, name: str, title: str, xlabel: str):
    data = importance.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(data) + 1.2))
    ax.barh([labels.feature_name(f) for f in data.index], data.values, color=COLOR_STAY)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return save_figure(fig, name)


def _local_plot(contrib: pd.Series, raw_row: pd.Series, name: str, title: str, xlabel: str):
    """Εξήγηση ενός πελάτη: οι 10 μεγαλύτερες συνεισφορές (πορτοκαλί = αυξάνει το ρίσκο, μπλε = το μειώνει)."""
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh([f"{labels.feature_name(f)} = {labels.value_name(f, _fmt(raw_row[f]))}" for f in top.index], top.values,
            color=[COLOR_CHURN if v > 0 else COLOR_STAY for v in top.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return save_figure(fig, name)


def direction_by_category(shap_df: pd.DataFrame, raw: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Μέση τιμή SHAP ανά κατηγορία (ή ανά τεταρτημόριο για τις αριθμητικές): προς τα πού «σπρώχνει» κάθε τιμή.

    Αντικαθιστά το beeswarm, γιατί τα χρώματά του δεν έχουν νόημα για κατηγορικούς κωδικούς.
    """
    rows = []
    for f in features:
        if f in NUMERIC:
            # Τεταρτημόρια με στρογγυλεμένα όρια, π.χ. «18,7–41,3», αντί για τα ακατέργαστα διαστήματα του pandas.
            _, edges = pd.qcut(raw[f], 4, retbins=True, duplicates="drop")
            names = [f"{edges[i]:.4g}–{edges[i + 1]:.4g}".replace(".", ",") for i in range(len(edges) - 1)]
            labels_ = pd.cut(raw[f], edges, labels=names, include_lowest=True).astype(str)
        else:
            labels_ = raw[f].astype(str)
        for value, idx in shap_df.groupby(labels_.to_numpy()).groups.items():
            rows.append({"feature": f, "category": value, "mean_shap": float(shap_df.loc[idx, f].mean()),
                         "n": int(len(idx))})
    return pd.DataFrame(rows)


def plot_direction(table: pd.DataFrame, features: list[str]):
    cols = 3
    rows = int(np.ceil(len(features) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.6 * rows))
    for ax, f in zip(axes.flat, features):
        sub = table[table["feature"] == f].sort_values("mean_shap")
        cats = sub["category"] if f in NUMERIC else [labels.capitalize(labels.value_name(f, c)) for c in sub["category"]]
        ax.barh(cats, sub["mean_shap"],
                color=[COLOR_CHURN if v > 0 else COLOR_STAY for v in sub["mean_shap"]])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(labels.feature_name(f))
        ax.set_xlabel("Μέση τιμή SHAP")
    for ax in list(axes.flat)[len(features):]:
        ax.axis("off")
    fig.suptitle("Κατεύθυνση της επίδρασης ανά κατηγορία (SHAP, τελικό MLP)\n"
                 "πορτοκαλί = αυξάνει τον κίνδυνο αποχώρησης, μπλε = τον μειώνει", y=1.02)
    fig.tight_layout()
    return save_figure(fig, "shap_nn_by_category")


# --- Κύρια συνάρτηση ------------------------------------------------------------------------------------


def run() -> None:
    X_train, _, _, _, X_test, y_test = load_splits()
    space = OriginalSpace(X_train, load_encoding_choice())
    predictors = model_predictors()

    rng = np.random.default_rng(SEED)
    Z_train = space.to_codes(X_train)
    background = Z_train[rng.choice(len(Z_train), N_BACKGROUND, replace=False)]
    explain_idx = np.sort(rng.choice(len(X_test), N_EXPLAIN, replace=False))
    X_exp_raw = space.model_view(X_test.iloc[explain_idx])
    Z_exp = space.to_codes(X_exp_raw)

    def shap_values(model: str) -> pd.DataFrame:
        f = lambda Z: predict_from_original(Z, space, predictors[model])  # noqa: E731
        explainer = shap.Explainer(f, shap.maskers.Independent(background, max_samples=N_BACKGROUND), seed=SEED)
        return pd.DataFrame(explainer(Z_exp).values, columns=space.features)

    # --- SHAP: νευρωνικό και XGBoost, με την ίδια μέθοδο, ώστε να συγκρίνονται άμεσα ---------------------
    shap_nn = shap_values("nn")
    shap_xgb = shap_values("xgb")
    shap_imp = shap_nn.abs().mean().sort_values(ascending=False)
    xgb_imp = shap_xgb.abs().mean().sort_values(ascending=False)
    shap_imp.rename("mean_abs_shap").to_frame().to_csv(TABLES_DIR / "shap_nn_importance.csv", index_label="feature")
    xgb_imp.rename("mean_abs_shap").to_frame().to_csv(TABLES_DIR / "shap_xgb_importance.csv", index_label="feature")
    _bar(shap_imp, "shap_nn_bar", "SHAP (τελικό MLP): σημασία χαρακτηριστικών",
         "Μέση |τιμή SHAP| (επίδραση στην πιθανότητα αποχώρησης)")
    _bar(xgb_imp, "shap_xgb_bar", "SHAP (XGBoost): σημασία χαρακτηριστικών",
         "Μέση |τιμή SHAP| (επίδραση στην πιθανότητα αποχώρησης)")

    top_features = list(shap_imp.index[:N_DIRECTION])
    direction = direction_by_category(shap_nn, X_exp_raw.reset_index(drop=True), top_features)
    direction.round(4).to_csv(TABLES_DIR / "shap_nn_by_category.csv", index=False)
    plot_direction(direction, top_features)

    # Οι δύο πελάτες για τις τοπικές εξηγήσεις: ο churner με το μεγαλύτερο ρίσκο και αυτός που έμεινε με το μικρότερο.
    prob_exp = predict_from_original(Z_exp, space, predictors["nn"])
    y_exp = np.asarray(y_test)[explain_idx]
    churner = int(np.argmax(np.where(y_exp == 1, prob_exp, -np.inf)))
    stayer = int(np.argmin(np.where(y_exp == 0, prob_exp, np.inf)))
    base = float(np.mean(predict_from_original(background, space, predictors["nn"])))
    customers = ((churner, "churner", "πελάτης που αποχώρησε"), (stayer, "stayer", "πελάτης που παρέμεινε"))
    for pos, tag, who in customers:
        _local_plot(shap_nn.iloc[pos], X_exp_raw.iloc[pos], f"shap_nn_local_{tag}",
                    f"SHAP: {who} (πρόβλεψη {prob_exp[pos]:.2f}, βάση {base:.2f})".replace(".", ","),
                    "Συνεισφορά SHAP στην πιθανότητα αποχώρησης")

    # --- LIME: οι παραλλαγές κάθε πελάτη παίρνουν πάντα έγκυρες κατηγορίες ---------------------------------
    def lime_predict(Z):
        p = predict_from_original(Z, space, predictors["nn"])
        return np.column_stack([1 - p, p])

    def lime_explain(rows: np.ndarray, seed: int) -> pd.DataFrame:
        explainer = LimeTabularExplainer(
            Z_train, feature_names=space.features, class_names=["0", "1"], mode="classification",
            categorical_features=space.cat_idx, categorical_names=space.category_names(),
            discretize_continuous=True, random_state=seed)
        weights = np.zeros((len(rows), len(space.features)))
        for i, row in enumerate(rows):
            exp = explainer.explain_instance(row, lime_predict, num_features=10, labels=(1,))
            for j, w in exp.as_map()[1]:
                weights[i, j] = w
        return pd.DataFrame(weights, columns=space.features)

    lime_nn = lime_explain(Z_exp, SEED)
    lime_imp = lime_nn.abs().mean().sort_values(ascending=False)
    lime_imp.rename("mean_abs_lime_weight").to_frame().to_csv(TABLES_DIR / "lime_nn_importance.csv", index_label="feature")
    _bar(lime_imp, "lime_nn_bar", "LIME (τελικό MLP): σημασία χαρακτηριστικών", "Μέσο |βάρος LIME|")
    for pos, tag, who in customers:
        _local_plot(lime_nn.iloc[pos], X_exp_raw.iloc[pos], f"lime_nn_local_{tag}",
                    f"LIME: {who} (πρόβλεψη {prob_exp[pos]:.2f})".replace(".", ","), "Βάρος LIME")

    # --- Σύγκριση SHAP και LIME ---------------------------------------------------------------------------
    others = [i for i in range(N_EXPLAIN) if i not in (churner, stayer)]
    compare_pos = [churner, stayer, *rng.choice(others, N_COMPARE - 2, replace=False)]
    overlaps = [len(_top_k(shap_nn.iloc[p]) & _top_k(lime_nn.iloc[p])) for p in compare_pos]
    feats = space.features
    rho, pval = spearmanr(shap_imp.reindex(feats), lime_imp.reindex(feats))
    pd.DataFrame([
        {"Μέτρο": "Spearman ρ των συνολικών κατατάξεων (SHAP έναντι LIME)", "Τιμή": rho},
        {"Μέτρο": "p-value του Spearman", "Τιμή": pval},
        {"Μέτρο": f"Μέση επικάλυψη των {TOP_K} κορυφαίων χαρακτηριστικών ανά πελάτη (από {TOP_K}, {N_COMPARE} πελάτες)", "Τιμή": np.mean(overlaps)},
        {"Μέτρο": f"Ελάχιστη επικάλυψη των {TOP_K} κορυφαίων χαρακτηριστικών", "Τιμή": np.min(overlaps)},
        {"Μέτρο": f"Μέγιστη επικάλυψη των {TOP_K} κορυφαίων χαρακτηριστικών", "Τιμή": np.max(overlaps)},
    ]).round(4).to_csv(TABLES_DIR / "shap_lime_agreement.csv", index=False)

    runs = [lime_explain(Z_exp[compare_pos], s) for s in LIME_SEEDS]
    pair_overlaps = [np.mean([len(_top_k(a.iloc[i]) & _top_k(b.iloc[i])) for i in range(len(compare_pos))])
                     for a, b in itertools.combinations(runs, 2)]
    pd.DataFrame([
        {"Μέτρο": f"Μέση επικάλυψη των {TOP_K} κορυφαίων χαρακτηριστικών ανάμεσα σε ζεύγη εκτελέσεων (από {TOP_K})", "Τιμή": np.mean(pair_overlaps)},
        {"Μέτρο": "Ελάχιστη μέση επικάλυψη ζεύγους", "Τιμή": np.min(pair_overlaps)},
        {"Μέτρο": "Πλήθος εκτελέσεων (seeds)", "Τιμή": len(LIME_SEEDS)},
        {"Μέτρο": "Πλήθος πελατών", "Τιμή": len(compare_pos)},
    ]).round(4).to_csv(TABLES_DIR / "lime_stability.csv", index=False)

    # --- Σύγκριση σημαντικότητας ανάμεσα σε μεθόδους και μοντέλα ------------------------------------------
    logreg = pd.read_csv(TABLES_DIR / "logreg_coefficients.csv")
    logreg_imp = logreg.assign(f=logreg["Χαρακτηριστικό"].map(original_feature)).groupby("f")["Συντελεστής"].apply(
        lambda s: s.abs().sum())
    comparison = pd.DataFrame({
        "NN SHAP": shap_imp.reindex(feats).rank(ascending=False),
        "NN LIME": lime_imp.reindex(feats).rank(ascending=False),
        "XGB SHAP": xgb_imp.reindex(feats).rank(ascending=False),
        "Logistic Regression (απόλυτοι συντελεστές)": logreg_imp.reindex(feats).rank(ascending=False),
    }).sort_values("NN SHAP")
    comparison.index.name = "feature"
    comparison.to_csv(TABLES_DIR / "importance_comparison.csv")
    rho_nn_xgb = spearmanr(comparison["NN SHAP"], comparison["XGB SHAP"])[0]

    print("Top 5 NN SHAP:", list(shap_imp.index[:5]))
    print("Top 5 NN LIME:", list(lime_imp.index[:5]))
    print("Top 5 XGB SHAP:", list(xgb_imp.index[:5]))
    print(f"Spearman SHAP–LIME: {rho:.3f}, μέση επικάλυψη top-5: {np.mean(overlaps):.2f}, "
          f"σταθερότητα LIME: {np.mean(pair_overlaps):.2f}, Spearman NN–XGB SHAP: {rho_nn_xgb:.3f}")
