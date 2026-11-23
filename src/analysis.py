"""Τελική αξιολόγηση στο test set και ανάλυση για την επιχείρηση.

**Μόνο αυτό το module αγγίζει το test set**, και μόνο αφού έχουν παγώσει όλα τα μοντέλα και τα
thresholds τους. Έτσι η αξιολόγηση είναι μια έντιμη εκτίμηση του πώς θα τα πήγαιναν τα μοντέλα σε
πελάτες που δεν έχουν δει ποτέ.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import labels
from src.config import MODEL_NAMES, MODELS_DIR, TABLES_DIR
from src.data import load_splits
from src.evaluation import (
    bootstrap_ci,
    compute_metrics,
    load_model_metrics,
    load_predictions,
    plot_confusion_matrix,
    plot_pr,
    plot_roc,
    save_predictions,
)
from src.calibration import PlattCalibrator, calibration_summary, paired_family
from src.plotting import COLOR_CHURN, COLOR_STAY, save_figure

NN_MODELS = {"mlp_simple", "mlp_tuned"}
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


# --- Τελική αξιολόγηση -------------------------------------------------------------------------------


def predict_proba(model_name: str, X) -> np.ndarray:
    """Πιθανότητα αποχώρησης από ένα αποθηκευμένο μοντέλο, σε αρχικά (μη κωδικοποιημένα) δεδομένα.

    Τα μοντέλα του scikit-learn είναι αποθηκευμένα ως ολόκληρα pipelines. Τα νευρωνικά χρειάζονται τον
    δικό τους αποθηκευμένο preprocessor, που εκπαιδεύτηκε στο ίδιο train set.
    """
    if model_name in NN_MODELS:
        import keras

        from src.nn_train import SeedEnsemble

        prep = joblib.load(MODELS_DIR / "nn_preprocessor.joblib")
        # Το τελικό MLP είναι ensemble 5 seeds· το απλό MLP είναι ένα μοντέλο (seed 42).
        model = (SeedEnsemble.load(model_name) if model_name == "mlp_tuned"
                 else keras.models.load_model(MODELS_DIR / f"{model_name}.keras"))
        return np.asarray(model.predict(prep.transform(X).astype("float32"), verbose=0)).ravel()
    return joblib.load(MODELS_DIR / f"{model_name}.joblib").predict_proba(X)[:, 1]


def run_final() -> None:
    """Προβλέψεις όλων των μοντέλων στο test, μετρικές σε δύο thresholds, διαστήματα εμπιστοσύνης, σχήματα."""
    *_, X_test, y_test = load_splits()
    rows, curves = [], {}
    for name in MODEL_NAMES:
        prob = predict_proba(name, X_test)
        save_predictions(name, "test", X_test.index, y_test, prob)
        curves[name] = (np.asarray(y_test), prob)
        tuned = load_model_metrics(name)["threshold"]
        for kind, t in (("0.5", 0.5), ("tuned", tuned)):
            m = compute_metrics(y_test, prob, t)
            auc_lo, auc_hi = bootstrap_ci(y_test, prob, t, "roc_auc")
            f1_lo, f1_hi = bootstrap_ci(y_test, prob, t, "f1")
            rows.append({"model": name, "Μοντέλο": labels.MODEL_DISPLAY_NAMES[name], "threshold_type": kind,
                         **{k: m[k] for k in ["threshold", *METRIC_COLUMNS, "tn", "fp", "fn", "tp"]},
                         "roc_auc_ci_low": auc_lo, "roc_auc_ci_high": auc_hi,
                         "f1_ci_low": f1_lo, "f1_ci_high": f1_hi})
        plot_confusion_matrix(y_test, prob, tuned, name, f"cm_{name}")

    table = pd.DataFrame(rows)
    table.round(4).to_csv(TABLES_DIR / "final_comparison.csv", index=False)

    ranking = table[table["threshold_type"] == "tuned"].sort_values("roc_auc", ascending=False)
    # Με 10 καμπύλες το σχήμα γίνεται δυσανάγνωστο. Τονίζονται το τελικό νευρωνικό και τα 3 καλύτερα.
    highlight = list(dict.fromkeys(["mlp_tuned", *ranking["model"].head(3)]))
    plot_roc(curves, "roc_all_models", highlight=highlight)
    plot_pr(curves, "pr_all_models", highlight=highlight)

    # Σύγκριση ανά ζεύγος στις ίδιες γραμμές του test
    y = np.asarray(y_test)
    probs = {name: curves[name][1] for name in MODEL_NAMES}
    best = ranking["model"].iloc[0]
    families = [paired_family(y, probs, "mlp_tuned", "vs_mlp_tuned")]
    if best != "mlp_tuned":
        families.append(paired_family(y, probs, best, f"vs_best_{best}"))
    paired = pd.concat(families, ignore_index=True)
    paired.round(4).to_csv(TABLES_DIR / "paired_comparison.csv", index=False)
    plot_paired(paired[paired["family"] == "vs_mlp_tuned"])

    # Βαθμονόμηση στο validation, εφαρμογή στο test
    cal_rows, calibrated = [], {}
    for name in MODEL_NAMES:
        val = load_predictions(name, "val")
        calibrator = PlattCalibrator().fit(val["y_prob"], val["y_true"])
        calibrated[name] = calibrator.transform(probs[name])
        save_predictions(name, "test_calibrated", X_test.index, y_test, calibrated[name])
        cal_rows.append({"model": name, "Μοντέλο": labels.MODEL_DISPLAY_NAMES[name],
                         **calibration_summary(y, probs[name], calibrated[name])})
    pd.DataFrame(cal_rows).round(4).to_csv(TABLES_DIR / "calibration.csv", index=False)
    plot_calibration(y, probs, calibrated, list(dict.fromkeys([best, "mlp_tuned"])))
    print(paired[["family", "model_a", "model_b", "delta_auc", "ci_low", "ci_high", "p_holm"]].round(4).to_string(index=False))
    print(ranking[["model", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "f1", "threshold"]]
          .round(4).to_string(index=False))


def plot_paired(table: pd.DataFrame):
    """Forest plot: ΔROC-AUC κάθε μοντέλου έναντι του τελικού MLP, με διάστημα 95% (paired bootstrap)."""
    data = table.sort_values("delta_auc")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(data) + 1.5))
    y = np.arange(len(data))
    colors = [COLOR_CHURN if sig else "#888888" for sig in data["significant_at_0.05"]]
    for yi, (_, row), color in zip(y, data.iterrows(), colors):
        ax.errorbar(row["delta_auc"], yi, xerr=[[row["delta_auc"] - row["ci_low"]], [row["ci_high"] - row["delta_auc"]]],
                    fmt="o", color=color, ecolor=color, capsize=3)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y, [labels.MODEL_DISPLAY_NAMES[m] for m in data["model_a"]])
    ax.set_xlabel("ΔROC-AUC έναντι του τελικού MLP (θετικό = καλύτερο από το τελικό MLP)")
    ax.set_title("Σύγκριση ανά ζεύγος στο σύνολο ελέγχου (ζευγαρωτό bootstrap, διάστημα εμπιστοσύνης 95%)\nπορτοκαλί = στατιστικά σημαντικό μετά τη διόρθωση Holm")
    return save_figure(fig, "paired_delta_auc")


def plot_calibration(y, raw: dict, calibrated: dict, models: list[str], n_bins: int = 10):
    """Διάγραμμα αξιοπιστίας: προβλεπόμενη πιθανότητα vs πραγματική συχνότητα, πριν και μετά τη βαθμονόμηση."""
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.8), squeeze=False)
    for ax, name in zip(axes[0], models):
        for probs, label, color in ((raw[name], "Πριν (χωρίς βαθμονόμηση)", COLOR_STAY),
                                    (calibrated[name], "Μετά (κλιμάκωση Platt)", COLOR_CHURN)):
            bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
            pts = [(probs[bins == b].mean(), y[bins == b].mean()) for b in range(n_bins) if (bins == b).sum() >= 5]
            ax.plot(*zip(*pts), marker="o", color=color, label=label)
        ax.plot([0, 1], [0, 1], linestyle=":", color="black", label="Τέλεια βαθμονόμηση")
        ax.set_xlabel("Προβλεπόμενη πιθανότητα αποχώρησης")
        ax.set_ylabel("Πραγματικό ποσοστό αποχώρησης")
        ax.set_title(labels.MODEL_DISPLAY_NAMES[name])
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Βαθμονόμηση πιθανοτήτων (σύνολο ελέγχου)", y=1.02)
    return save_figure(fig, "calibration")


# --- Ανάλυση για την επιχείρηση ---------------------------------------------------------------------


def business_models() -> list[str]:
    """Τα μοντέλα της ανάλυσης: το καλύτερο στο test και, αν τα διαστήματά τους επικαλύπτονται, το τελικό νευρωνικό.

    Αν τα διαστήματα εμπιστοσύνης του ROC-AUC επικαλύπτονται, τα δύο μοντέλα δεν διαφέρουν αξιόπιστα. Τότε
    δείχνουμε και το νευρωνικό, που είναι το κύριο μοντέλο της εργασίας.
    """
    table = pd.read_csv(TABLES_DIR / "final_comparison.csv")
    tuned = table[table["threshold_type"] == "tuned"].set_index("model")
    best = tuned["roc_auc"].idxmax()
    if best == "mlp_tuned":
        return [best]
    overlap = tuned.loc[best, "roc_auc_ci_low"] <= tuned.loc["mlp_tuned", "roc_auc_ci_high"]
    return [best, "mlp_tuned"] if overlap else [best]


def error_profile(model: str, X_test, y_test, prob, threshold) -> pd.DataFrame:
    """Ποιοι είναι οι churners που το μοντέλο δεν εντοπίζει (FN), σε σύγκριση με αυτούς που εντοπίζει (TP)."""
    pred = (prob >= threshold).astype(int)
    y = np.asarray(y_test)
    groups = {
        "TP (σωστά εντοπισμένες αποχωρήσεις)": (y == 1) & (pred == 1),
        "FN (αποχωρήσεις που δεν εντοπίστηκαν)": (y == 1) & (pred == 0),
        "FP (λανθασμένες προβλέψεις αποχώρησης)": (y == 0) & (pred == 1),
        "TN (σωστά «παραμένει»)": (y == 0) & (pred == 0),
    }
    rows = []
    for group, mask in groups.items():
        sub = X_test[mask]
        rows.append({
            "model": model, "Ομάδα": group, "Πλήθος": int(mask.sum()),
            "Μέση διάρκεια συνδρομής (μήνες)": sub["tenure"].mean(),
            "Μέση μηνιαία χρέωση": sub["MonthlyCharges"].mean(),
            "Συχνότερο συμβόλαιο": labels.value_name("Contract", sub["Contract"].mode().iat[0]) if len(sub) else "",
            "Μερίδιο μηνιαίων συμβολαίων": (sub["Contract"] == "Month-to-month").mean(),
            "Συχνότερος τρόπος πληρωμής": labels.value_name("PaymentMethod", sub["PaymentMethod"].mode().iat[0]) if len(sub) else "",
            "Μερίδιο οπτικής ίνας": (sub["InternetService"] == "Fiber optic").mean(),
        })
    return pd.DataFrame(rows)


def lift_table(model: str, y_test, prob) -> pd.DataFrame:
    """Lift και gains: αν η εταιρεία στοχεύσει το k% των πελατών με τον μεγαλύτερο κίνδυνο, τι ποσοστό των αποχωρήσεων εντοπίζει;

    - recall@k: τι ποσοστό όλων των churners βρίσκεται μέσα στο k%·
    - precision@k: τι ποσοστό του k% είναι πράγματι churners·
    - lift: πόσες φορές καλύτερα από το να διάλεγε πελάτες στην τύχη.

    Από την precision@k προκύπτει και η συνθήκη ισορροπίας: μια καμπάνια με κόστος C ανά πελάτη, που
    κρατά έναν churner με πιθανότητα s και αξία V, είναι κερδοφόρα όταν s·V·precision@k > C, δηλαδή
    όταν s·V/C > 1/precision@k. Δεν υποθέτουμε καμία τιμή για C, V, s· η εταιρεία βάζει τις δικές της.
    """
    y = np.asarray(y_test)
    order = np.argsort(-prob, kind="stable")
    base_rate = y.mean()
    rows = []
    for k in range(5, 55, 5):
        n = int(round(len(y) * k / 100))
        caught = int(y[order[:n]].sum())
        precision = caught / n
        rows.append({"model": model, "k_percent": k, "n_targeted": n, "churners_caught": caught,
                     "recall_at_k": caught / y.sum(), "precision_at_k": precision,
                     "lift": precision / base_rate, "required_sV_over_C": 1 / precision})
    return pd.DataFrame(rows)


def plot_gains(lifts: pd.DataFrame, y_test, probs: dict):
    """Καμπύλη cumulative gains: ποσοστό churners που εντοπίζονται ανάλογα με το ποσοστό πελατών στην καμπάνια."""
    y = np.asarray(y_test)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    x = np.linspace(0, 1, 201)
    for name, prob in probs.items():
        order = np.argsort(-prob, kind="stable")
        captured = np.concatenate([[0], np.cumsum(y[order]) / y.sum()])
        share = np.arange(len(y) + 1) / len(y)
        ax.plot(share * 100, captured * 100, linewidth=2, label=labels.MODEL_DISPLAY_NAMES[name])
    ax.plot(x * 100, x * 100, color="black", linestyle=":", label="Τυχαία επιλογή")
    perfect = np.minimum(x / y.mean(), 1)
    ax.plot(x * 100, perfect * 100, color="grey", linestyle="--", linewidth=1, label="Τέλειο μοντέλο")
    first = lifts[lifts["model"] == list(probs)[0]].set_index("k_percent")
    if 20 in first.index:
        r20 = first.loc[20, "recall_at_k"] * 100
        ax.annotate(f"20% → {r20:.1f}% των αποχωρήσεων".replace(".", ","), (20, r20), xytext=(32, r20 - 18),
                    arrowprops={"arrowstyle": "->"})
    ax.set_xlabel("Ποσοστό πελατών στην καμπάνια (κατά φθίνοντα κίνδυνο), %")
    ax.set_ylabel("Ποσοστό αποχωρήσεων που εντοπίζονται, %")
    ax.set_title("Καμπύλη σωρευτικών κερδών (σύνολο ελέγχου)")
    ax.legend(loc="lower right")
    return save_figure(fig, "cumulative_gains")


def plot_breakeven(lifts: pd.DataFrame):
    """Η ελάχιστη τιμή του s·V/C για να είναι κερδοφόρα η καμπάνια, ανάλογα με το k%."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, sub in lifts.groupby("model", sort=False):
        ax.plot(sub["k_percent"], sub["required_sV_over_C"], marker="o", linewidth=2,
                label=labels.MODEL_DISPLAY_NAMES[name])
    ax.set_xlabel("Ποσοστό πελατών στην καμπάνια (k%)")
    ax.set_ylabel("Ελάχιστο s·V / C για κέρδος")
    ax.set_title("Συνθήκη ισορροπίας της καμπάνιας διατήρησης")
    ax.text(0.02, 0.97, "Κερδοφόρο όταν s·V/C > 1 / precision@k\nη εταιρεία συμπληρώνει τα δικά της C, V, s",
            transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "edgecolor": "#CCCCCC"})
    ax.legend(loc="lower right")
    return save_figure(fig, "breakeven_curve")


SEGMENTS = {
    "a": ("Μηνιαίο συμβόλαιο και διάρκεια συνδρομής ≤ 12 μήνες", lambda X: (X["Contract"] == "Month-to-month") & (X["tenure"] <= 12)),
    "b": ("Οπτική ίνα χωρίς τεχνική υποστήριξη", lambda X: (X["InternetService"] == "Fiber optic") & (X["TechSupport"] == "No")),
    "c": ("Πληρωμή με ηλεκτρονική επιταγή", lambda X: X["PaymentMethod"] == "Electronic check"),
    "d": ("Ηλικιωμένοι πελάτες", lambda X: X["SeniorCitizen"] == 1),
    "e": ("Διετές συμβόλαιο (αναφορά χαμηλού κινδύνου)", lambda X: X["Contract"] == "Two year"),
}


def segment_table(model: str, X_test, y_test, prob_calibrated) -> pd.DataFrame:
    """Μέγεθος, πραγματικό ποσοστό αποχώρησης και μέσο **βαθμονομημένο** ρίσκο για κάθε τμήμα πελατών.

    Χρησιμοποιούνται οι βαθμονομημένες πιθανότητες, ώστε το «ρίσκο» να συγκρίνεται άμεσα με το πραγματικό
    ποσοστό αποχώρησης.
    """
    y = np.asarray(y_test)
    rows = []
    for key, (label, rule) in SEGMENTS.items():
        mask = rule(X_test).to_numpy()
        rows.append({"model": model, "segment": key, "Τμήμα": label, "Πλήθος": int(mask.sum()),
                     "Μερίδιο πελατών": mask.mean(), "Πραγματικό ποσοστό αποχώρησης": y[mask].mean(),
                     "Μέσο βαθμονομημένο ρίσκο": prob_calibrated[mask].mean()})
    rows.append({"model": model, "segment": "all", "Τμήμα": "Όλοι οι πελάτες", "Πλήθος": len(y),
                 "Μερίδιο πελατών": 1.0, "Πραγματικό ποσοστό αποχώρησης": y.mean(),
                 "Μέσο βαθμονομημένο ρίσκο": prob_calibrated.mean()})
    return pd.DataFrame(rows)


def run_business() -> None:
    """Λάθη, lift, συνθήκη ισορροπίας και τμήματα πελατών, χωρίς καμία υπόθεση για χρηματικά ποσά."""
    *_, X_test, y_test = load_splits()
    models = business_models()
    probs = {m: load_predictions(m, "test").set_index("row_id").loc[X_test.index, "y_prob"].to_numpy()
             for m in models}

    errors, lifts, segments = [], [], []
    for m in models:
        t = load_model_metrics(m)["threshold"]
        errors.append(error_profile(m, X_test, y_test, probs[m], t))
        lifts.append(lift_table(m, y_test, probs[m]))
        cal = load_predictions(m, "test_calibrated").set_index("row_id").loc[X_test.index, "y_prob"].to_numpy()
        segments.append(segment_table(m, X_test, y_test, cal))
    errors, lifts, segments = (pd.concat(x, ignore_index=True) for x in (errors, lifts, segments))

    errors.round(4).to_csv(TABLES_DIR / "error_profile.csv", index=False)
    lifts.round(4).to_csv(TABLES_DIR / "lift_at_k.csv", index=False)
    lifts[["model", "k_percent", "precision_at_k", "required_sV_over_C"]].round(4).to_csv(
        TABLES_DIR / "breakeven_at_k.csv", index=False)
    segments.round(4).to_csv(TABLES_DIR / "segments.csv", index=False)
    plot_gains(lifts, y_test, probs)
    plot_breakeven(lifts)

    print("Μοντέλα ανάλυσης:", models)
    print(lifts[lifts["k_percent"].isin([10, 20])].round(3).to_string(index=False))
    print(segments.round(3).to_string(index=False))
