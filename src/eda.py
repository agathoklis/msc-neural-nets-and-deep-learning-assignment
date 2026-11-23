"""Περιγραφική ανάλυση των δεδομένων: το υλικό για το «Describe the dataset» του ερωτήματος 1.

Η ανάλυση γίνεται σε όλο το καθαρισμένο dataset. Είναι μόνο περιγραφική και δεν παίρνει καμία απόφαση
μοντελοποίησης, οπότε δεν υπάρχει διαρροή από το test set.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import labels
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression

from src.config import BINARY_PASSTHROUGH, CATEGORICAL, ID_COLUMN, NUMERIC, TABLES_DIR, TARGET
from src.data import clean, load_raw
from src.plotting import CLASS_COLORS, COLOR_CHURN, save_figure

# Οι 16 κατηγορικές/δυαδικές μεταβλητές: όλες οι στήλες κειμένου και το SeniorCitizen.
CATEGORY_FEATURES = BINARY_PASSTHROUGH + CATEGORICAL

# Οι 6 μεταβλητές που δείχνει το σχήμα του κυρίως κειμένου (οι υπόλοιπες πάνε στο παράρτημα).
KEY_CATEGORY_FEATURES = ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity", "PaperlessBilling"]


def _feature_type(raw: pd.DataFrame, col: str) -> str:
    """Κατατάσσει μια στήλη σε αριθμητική, δυαδική ή κατηγορική (ή αναγνωριστικό)."""
    if col == ID_COLUMN:
        return "id"
    if col in NUMERIC:
        return "numeric"
    return "binary" if raw[col].nunique() == 2 else "categorical"


def dataset_facts(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Τα βασικά νούμερα του dataset, που παραθέτει η ενότητα 2.1 της εργασίας."""
    features = [c for c in raw.columns if c not in (ID_COLUMN, TARGET)]
    types = [_feature_type(raw, c) for c in features]
    # Τα «κενά» στο CSV δεν είναι NaN αλλά κενά strings, οπότε μετράμε και τα δύο.
    missing = {
        c: int(raw[c].isna().sum() + (raw[c].astype(str).str.strip() == "").sum()) for c in raw.columns
    }
    product = df["tenure"] * df["MonthlyCharges"]
    return {
        "n_rows": int(len(raw)),
        "n_columns": int(raw.shape[1]),
        "n_features": len(features),
        "n_numeric": types.count("numeric"),
        "n_binary": types.count("binary"),
        "n_categorical": types.count("categorical"),
        "missing_before_cleaning": {c: n for c, n in missing.items() if n > 0},
        "churn_count": int(df[TARGET].sum()),
        "churn_rate": float(df[TARGET].mean()),
        "corr_tenure_totalcharges": float(df["tenure"].corr(df["TotalCharges"])),
        "corr_totalcharges_tenure_x_monthly": float(df["TotalCharges"].corr(product)),
        "skew": {c: float(df[c].skew()) for c in NUMERIC},
        "source": "IBM Telco Customer Churn sample data (μέσω Kaggle: blastchar/telco-customer-churn)",
    }


def dataset_description(raw: pd.DataFrame) -> pd.DataFrame:
    """Ο πίνακας περιγραφής: μία γραμμή ανά αρχική στήλη, με την ελληνική αντιστοιχία."""
    rows = []
    for col in raw.columns:
        ftype = _feature_type(raw, col)
        values = ""
        if ftype in ("binary", "categorical") or col == TARGET:
            values = ", ".join(map(str, sorted(raw[col].unique(), key=str)))
        note = {"customerID": "αφαιρείται", TARGET: "μεταβλητή-στόχος"}.get(col, "")
        rows.append(
            {
                "Μεταβλητή": col,
                "Περιγραφή": labels.FEATURE_DESCRIPTIONS[col],
                "Ομάδα": labels.FEATURE_GROUPS[col],
                "Τύπος": "Δυαδική" if col == TARGET else labels.FEATURE_SUBTYPES.get(
                    col, "Κατηγορική ονομαστική" if ftype == "categorical" else labels.FEATURE_TYPES[ftype]),
                "Πλήθος τιμών": int(raw[col].nunique()),
                "Τιμές": values,
                "Σημείωση": note,
            }
        )
    return pd.DataFrame(rows)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Περιγραφικά στατιστικά των αριθμητικών μεταβλητών, χωριστά για όσους έμειναν και όσους έφυγαν."""
    rows = []
    for col in NUMERIC:
        for cls, group in df.groupby(TARGET)[col]:
            d = group.describe()
            rows.append(
                {
                    "Μεταβλητή": col,
                    "Ομάδα": labels.CLASS_LABELS[cls],
                    "Πλήθος": int(d["count"]),
                    "Μέσος όρος": d["mean"],
                    "Τυπική απόκλιση": d["std"],
                    "Ελάχιστο": d["min"],
                    "25%": d["25%"],
                    "Διάμεσος": d["50%"],
                    "75%": d["75%"],
                    "Μέγιστο": d["max"],
                    "Ασυμμετρία": group.skew(),
                }
            )
    return pd.DataFrame(rows).round(3)


def churn_rate_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Πλήθος, μερίδιο και ποσοστό αποχώρησης για κάθε κατηγορία κάθε κατηγορικής μεταβλητής."""
    rows = []
    for col in CATEGORY_FEATURES:
        for value, group in df.groupby(col)[TARGET]:
            rows.append(
                {
                    "Μεταβλητή": col,
                    "Κατηγορία": value,
                    "Πλήθος": int(len(group)),
                    "Μερίδιο": len(group) / len(df),
                    "Ποσοστό αποχώρησης": group.mean(),
                }
            )
    return pd.DataFrame(rows).round(4)


# --- Σχέσεις ανάμεσα στα χαρακτηριστικά ------------------------------------------------------------

SERVICES = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                   "StreamingMovies"]


def cramers_v(a, b) -> float:
    """Cramér's V: η ένταση της σχέσης δύο κατηγορικών μεταβλητών, από 0 (καμία) έως 1 (πλήρης).

    Βασίζεται στο χ² του πίνακα συνάφειας, κανονικοποιημένο ώστε να μην εξαρτάται από το μέγεθος του
    δείγματος ή το πλήθος των κατηγοριών (Cramér, 1946).
    """
    table = pd.crosstab(a, b)
    if min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1))))


def correlation_ratio(categories, values) -> float:
    """Correlation ratio η: πόσο από τη διακύμανση μιας αριθμητικής μεταβλητής εξηγούν οι κατηγορίες μιας άλλης."""
    df = pd.DataFrame({"c": categories, "v": values})
    mean = df["v"].mean()
    groups = df.groupby("c")["v"]
    between = (groups.count() * (groups.mean() - mean) ** 2).sum()
    total = ((df["v"] - mean) ** 2).sum()
    return float(np.sqrt(between / total))


def categorical_associations(df: pd.DataFrame) -> pd.DataFrame:
    """Cramér's V για κάθε ζεύγος των 16 κατηγορικών μεταβλητών, ταξινομημένο από την ισχυρότερη σχέση."""
    rows = [
        {"feature_a": a, "feature_b": b, "cramers_v": cramers_v(df[a], df[b])}
        for i, a in enumerate(CATEGORY_FEATURES) for b in CATEGORY_FEATURES[i + 1:]
    ]
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).round(4)


def categorical_numeric_associations(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation ratio η για κάθε ζεύγος κατηγορικής × αριθμητικής μεταβλητής."""
    rows = [{"categorical": c, "numeric": n, "eta": correlation_ratio(df[c], df[n])}
            for c in CATEGORY_FEATURES for n in NUMERIC]
    return pd.DataFrame(rows).sort_values("eta", ascending=False).round(4)


def r2_monthly_from_services(df: pd.DataFrame) -> float:
    """Πόσο από τη μηνιαία χρέωση εξηγούν μόνο οι 9 υπηρεσίες (R² γραμμικής παλινδρόμησης)."""
    X = pd.get_dummies(df[SERVICES], drop_first=True).astype(float)
    return float(LinearRegression().fit(X, df["MonthlyCharges"]).score(X, df["MonthlyCharges"]))


def redundancy_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Οι ακριβείς επαναλήψεις πληροφορίας στο dataset. Κάθε γραμμή είναι ένας έλεγχος που πρέπει να βγει True."""
    no_internet = df["InternetService"] == "No"
    rows = []
    for col in INTERNET_ADDONS:
        same = ((df[col] == "No internet service") == no_internet)
        rows.append({"Έλεγχος": f"{col} = 'No internet service' ⇔ InternetService = 'No'",
                     "Αποτέλεσμα": bool(same.all()), "Γραμμές": int(no_internet.sum()), "Τιμή": ""})
    no_phone = df["PhoneService"] == "No"
    same = (df["MultipleLines"] == "No phone service") == no_phone
    rows.append({"Έλεγχος": "MultipleLines = 'No phone service' ⇔ PhoneService = 'No'",
                 "Αποτέλεσμα": bool(same.all()), "Γραμμές": int(no_phone.sum()), "Τιμή": ""})
    r2 = r2_monthly_from_services(df)
    rows.append({"Έλεγχος": "R² του MonthlyCharges από τις 9 υπηρεσίες ≥ 0,99",
                 "Αποτέλεσμα": r2 >= 0.99, "Γραμμές": len(df), "Τιμή": round(r2, 4)})
    corr = df["TotalCharges"].corr(df["tenure"] * df["MonthlyCharges"])
    rows.append({"Έλεγχος": "Συσχέτιση TotalCharges με tenure × MonthlyCharges ≥ 0,99",
                 "Αποτέλεσμα": corr >= 0.99, "Γραμμές": len(df), "Τιμή": round(corr, 4)})
    return pd.DataFrame(rows)


def plot_cramers_v(df: pd.DataFrame):
    """Χάρτης θερμότητας του Cramér's V ανάμεσα στις 16 κατηγορικές μεταβλητές."""
    n = len(CATEGORY_FEATURES)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = matrix[j, i] = cramers_v(df[CATEGORY_FEATURES[i]], df[CATEGORY_FEATURES[j]])
    fig, ax = plt.subplots(figsize=(10, 8.5))
    names = [labels.feature_name(f) for f in CATEGORY_FEATURES]
    sns.heatmap(pd.DataFrame(matrix, index=names, columns=names), annot=True, fmt=".2f",
                cmap="Reds", vmin=0, vmax=1, square=True, ax=ax, annot_kws={"size": 7},
                cbar_kws={"label": "Cramér's V (0 = καμία σχέση, 1 = πλήρης)"})
    ax.set_title("Σχέσεις ανάμεσα στις κατηγορικές μεταβλητές (Cramér's V)")
    return save_figure(fig, "eda_cramers_v")


# --- Σχήματα ----------------------------------------------------------------------------------


def plot_target(df: pd.DataFrame):
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([labels.CLASS_LABELS[c] for c in counts.index], counts.values,
                  color=[CLASS_COLORS[c] for c in counts.index])
    for bar, n in zip(bars, counts.values):
        ax.annotate(f"{n:,}".replace(",", ".") + f"\n({n / len(df):.1%})".replace(".", ","),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom")
    ax.set_title("Κατανομή της μεταβλητής-στόχου (αποχώρηση)")
    ax.set_ylabel("Πλήθος πελατών")
    ax.set_ylim(0, counts.max() * 1.2)
    return save_figure(fig, "eda_target")


def plot_numeric(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, NUMERIC):
        for cls in (0, 1):
            ax.hist(df.loc[df[TARGET] == cls, col], bins=30, alpha=0.6, density=True,
                    color=CLASS_COLORS[cls], label=labels.CLASS_LABELS[cls])
        ax.set_title(labels.feature_name(col))
        ax.set_xlabel(labels.FEATURE_DESCRIPTIONS[col])
        ax.set_ylabel("Πυκνότητα")
    axes[0].legend(title="Πελάτης")
    fig.suptitle("Κατανομές των αριθμητικών μεταβλητών ανά ομάδα πελατών", y=1.03)
    return save_figure(fig, "eda_numeric")


def plot_categorical_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    for ax, col in zip(axes.flat, CATEGORY_FEATURES):
        counts = df[col].value_counts()
        ax.barh([labels.capitalize(labels.value_name(col, v)) for v in counts.index], counts.values, color="#8C8C8C")
        ax.set_title(labels.feature_name(col))
        ax.set_xlabel("Πλήθος πελατών")
        ax.invert_yaxis()
    fig.suptitle("Κατανομές των κατηγορικών μεταβλητών", y=1.0)
    fig.tight_layout()
    return save_figure(fig, "eda_categorical_distributions")


def plot_churn_by_category(df: pd.DataFrame):
    overall = df[TARGET].mean()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, KEY_CATEGORY_FEATURES):
        rates = df.groupby(col)[TARGET].mean().sort_values()
        ax.barh([labels.capitalize(labels.value_name(col, v)) for v in rates.index], rates.values, color=COLOR_CHURN)
        ax.axvline(overall, color="black", linestyle="--", linewidth=1, label="Συνολικό ποσοστό")
        for i, r in enumerate(rates.values):
            ax.text(r + 0.01, i, f"{r:.1%}".replace(".", ","), va="center", fontsize=9)
        ax.set_title(labels.feature_name(col))
        ax.set_xlim(0, max(0.6, rates.max() * 1.2))
        ax.set_xlabel("Ποσοστό αποχώρησης")
    axes.flat[0].legend(loc="lower right")
    fig.suptitle("Ποσοστό αποχώρησης ανά κατηγορία", y=1.0)
    fig.tight_layout()
    return save_figure(fig, "eda_churn_by_category")


def plot_correlation(df: pd.DataFrame):
    corr = df[NUMERIC + [TARGET]].corr().rename(index=labels.feature_name, columns=labels.feature_name)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"label": "Συντελεστής Pearson"})
    ax.set_title("Συσχετίσεις αριθμητικών μεταβλητών και στόχου")
    return save_figure(fig, "eda_correlation")


def run() -> None:
    """Παράγει όλους τους πίνακες και τα σχήματα της περιγραφικής ανάλυσης."""
    raw = load_raw()
    df = clean(raw)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    facts = dataset_facts(raw, df)
    facts["r2_monthlycharges_from_services"] = r2_monthly_from_services(df)
    (TABLES_DIR / "dataset_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2))
    dataset_description(raw).to_csv(TABLES_DIR / "dataset_description.csv", index=False)
    numeric_summary(df).to_csv(TABLES_DIR / "numeric_summary.csv", index=False)
    churn_rate_by_category(df).to_csv(TABLES_DIR / "churn_rate_by_category.csv", index=False)

    plot_target(df)
    plot_numeric(df)
    plot_categorical_distributions(df)
    plot_churn_by_category(df)
    plot_correlation(df)

    # Σχέσεις ανάμεσα στα χαρακτηριστικά
    assoc = categorical_associations(df)
    assoc.to_csv(TABLES_DIR / "feature_associations_categorical.csv", index=False)
    categorical_numeric_associations(df).to_csv(TABLES_DIR / "feature_associations_cat_num.csv", index=False)
    redundancy = redundancy_checks(df)
    redundancy.to_csv(TABLES_DIR / "feature_redundancy.csv", index=False)
    plot_cramers_v(df)
    print(f"Ζεύγη με Cramér's V > 0,5: {(assoc['cramers_v'] > 0.5).sum()} από {len(assoc)}; "
          f"R² MonthlyCharges ~ υπηρεσίες: {facts['r2_monthlycharges_from_services']:.4f}; "
          f"όλοι οι έλεγχοι επανάληψης True: {bool(redundancy['Αποτέλεσμα'].all())}")

    print(f"Γραμμές: {facts['n_rows']}, ποσοστό αποχώρησης: {facts['churn_rate']:.3f}")
    print(f"Συσχέτιση tenure–TotalCharges: {facts['corr_tenure_totalcharges']:.3f}")
    print("Ασυμμετρία:", {k: round(v, 3) for k, v in facts["skew"].items()})
