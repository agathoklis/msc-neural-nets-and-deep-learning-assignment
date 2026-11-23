"""Κοινή αξιολόγηση για όλα τα μοντέλα.

Κάθε μοντέλο αξιολογείται με τις ίδιες συναρτήσεις, ώστε οι μετρικές να είναι συγκρίσιμες. Οι
προβλέψεις αποθηκεύονται ως πιθανότητες (όχι ως 0/1), για να μπορεί αργότερα να αλλάξει το threshold
χωρίς να ξανατρέξει το μοντέλο.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import labels
from src.config import METRICS_DIR, PREDICTIONS_DIR, SEED, THRESHOLD_GRID
from src.plotting import save_figure


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Υπολογίζει όλες τις μετρικές της εκφώνησης (και το PR-AUC) για ένα threshold.

    Το ROC-AUC και το PR-AUC δεν εξαρτώνται από το threshold. Οι υπόλοιπες μετρικές εξαρτώνται, γι' αυτό
    το threshold επιστρέφεται μαζί τους.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # zero_division=0: αν ένα μοντέλο δεν προβλέψει κανέναν churner, το precision είναι 0 και όχι σφάλμα.
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def tune_threshold(y_true, y_prob) -> float:
    """Βρίσκει το threshold που μεγιστοποιεί το F1, δοκιμάζοντας τιμές από 0,01 έως 0,99.

    Καλείται **μόνο** με δεδομένα validation. Αν διαλέγαμε το threshold στο test, η αξιολόγηση θα ήταν
    υπεραισιόδοξη.
    """
    start, stop, step = THRESHOLD_GRID
    grid = np.round(np.arange(start, stop + step / 2, step), 2)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def bootstrap_ci(y_true, y_prob, threshold: float, metric: str, n: int = 1000, seed: int = SEED,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Διάστημα εμπιστοσύνης 95% για μια μετρική, με επαναδειγματοληψία (bootstrap).

    Με ~1.000 πελάτες στο test, μικρές διαφορές ανάμεσα στα μοντέλα μπορεί να είναι τυχαίες. Το διάστημα
    δείχνει πόσο θα «κουνιόταν» η μετρική σε ένα άλλο, εξίσου πιθανό, δείγμα πελατών.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue  # δείγμα με μία μόνο κλάση: το AUC δεν ορίζεται
        values.append(compute_metrics(y_true[idx], y_prob[idx], threshold)[metric])
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


# --- Έλεγχος άκρων του χώρου αναζήτησης ------------------------------------------------------------

def grid_edge_rows(model_name: str, space: dict, best_params: dict, threshold: float | None = None) -> list[dict]:
    """Βρίσκει τις υπερπαραμέτρους που η καλύτερη τιμή τους είναι η πρώτη ή η τελευταία του πλέγματος.

    Αν το καλύτερο προκύπτει στην άκρη, το πραγματικό καλύτερο μπορεί να βρίσκεται έξω από το πλέγμα. Ελέγχονται
    μόνο πλέγματα με αριθμητικές, διατεταγμένες τιμές (π.χ. όχι «sqrt» ή None).
    """
    rows = []
    for name, values in space.items():
        key = name.replace("model__", "")
        numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(numeric) < 3 or key not in best_params:
            continue
        best = best_params[key]
        if best in (min(numeric), max(numeric)):
            rows.append({"model": model_name, "parameter": key, "value": best,
                         "edge": "min" if best == min(numeric) else "max", "grid": str(sorted(numeric))})
    if threshold is not None:
        start, stop, _ = THRESHOLD_GRID
        if threshold <= start or threshold >= stop:
            rows.append({"model": model_name, "parameter": "threshold", "value": threshold,
                         "edge": "min" if threshold <= start else "max", "grid": f"{start}–{stop}"})
    return rows


def update_grid_edge_report(model_name: str, rows: list[dict]) -> None:
    """Γράφει (ή αντικαθιστά) τις γραμμές ενός μοντέλου στο `grid_edge_report.csv`."""
    from src.config import TABLES_DIR

    path = TABLES_DIR / "grid_edge_report.csv"
    columns = ["model", "parameter", "value", "edge", "grid"]
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)
    new = pd.concat([old[old["model"] != model_name], pd.DataFrame(rows, columns=columns)], ignore_index=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    for r in rows:
        print(f"ΠΡΟΣΟΧΗ: {model_name}.{r['parameter']} = {r['value']} στην άκρη ({r['edge']}) του πλέγματος {r['grid']}")


def search_flatness_row(model_name: str, search) -> dict:
    """Πόσο «επίπεδος» είναι ο χώρος αναζήτησης: πόσες ρυθμίσεις απέχουν από την καλύτερη λιγότερο από μία
    τυπική απόκλιση των folds. Αν είναι οι περισσότερες, η επιλογή της «καλύτερης» (και το αν πέφτει στην άκρη
    του πλέγματος) είναι θέμα θορύβου και όχι πραγματικής διαφοράς."""
    r = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    scores = r["mean_test_score"].to_numpy()
    fold_std = float(r["std_test_score"].iloc[0])
    return {"model": model_name, "n_configs": len(scores), "best_cv_auc": float(scores[0]), "fold_std": fold_std,
            "cv_auc_5th": float(scores[min(4, len(scores) - 1)]), "cv_auc_10th": float(scores[min(9, len(scores) - 1)]),
            "within_1_std": int((scores >= scores[0] - fold_std).sum()),
            "share_within_1_std": float((scores >= scores[0] - fold_std).mean())}


def update_search_flatness(row: dict) -> None:
    """Γράφει (ή αντικαθιστά) τη γραμμή ενός μοντέλου στο `search_flatness.csv`."""
    from src.config import TABLES_DIR

    path = TABLES_DIR / "search_flatness.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=list(row))
    new = pd.concat([old[old["model"] != row["model"]], pd.DataFrame([row])], ignore_index=True)
    new.round(4).to_csv(path, index=False)


# --- Αποθήκευση -------------------------------------------------------------------------------


def save_predictions(model_name: str, split: str, row_ids, y_true, y_prob) -> None:
    """Αποθηκεύει τις πιθανότητες ενός μοντέλου για ένα σύνολο (train/val/test)."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": np.asarray(row_ids), "y_true": np.asarray(y_true), "y_prob": np.asarray(y_prob)}).to_csv(
        PREDICTIONS_DIR / f"{model_name}_{split}.csv", index=False
    )


def load_predictions(model_name: str, split: str) -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_DIR / f"{model_name}_{split}.csv")


def save_model_metrics(model_name: str, threshold: float, val_metrics: dict, params: dict, extra: dict | None = None) -> None:
    """Αποθηκεύει το threshold, τις μετρικές validation και τις υπερπαραμέτρους ενός μοντέλου."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"threshold": threshold, "val": val_metrics, "params": params, **(extra or {})}
    (METRICS_DIR / f"{model_name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def load_model_metrics(model_name: str) -> dict:
    return json.loads((METRICS_DIR / f"{model_name}.json").read_text())


# --- Σχήματα ----------------------------------------------------------------------------------


def plot_confusion_matrix(y_true, y_prob, threshold: float, model_name: str, name: str):
    """Πίνακας σύγχυσης στο δοσμένο threshold, με ελληνικές ετικέτες."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred, labels=[0, 1]),
        display_labels=[labels.CLASS_LABELS[0], labels.CLASS_LABELS[1]],
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_xlabel("Προβλεπόμενη κλάση")
    ax.set_ylabel("Πραγματική κλάση")
    title = labels.MODEL_DISPLAY_NAMES.get(model_name, model_name)
    ax.set_title(f"{title} (κατώφλι {threshold:.2f})".replace(".", ","))
    return save_figure(fig, name)


def _styled_curves(curves: dict, highlight: list[str] | None):
    """Κοινή λογική χρωμάτων: τα μοντέλα του highlight χρωματιστά, τα υπόλοιπα γκρι."""
    palette = plt.get_cmap("tab10")
    for i, (model, data) in enumerate(curves.items()):
        emphasized = highlight is None or model in highlight
        style = {"color": palette(i % 10) if emphasized else "#BBBBBB", "linewidth": 2 if emphasized else 1,
                 "zorder": 3 if emphasized else 1}
        yield model, data, style


def plot_roc(curves: dict, name: str, highlight: list[str] | None = None):
    """Καμπύλες ROC. `curves`: {μοντέλο: (y_true, y_prob)}."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for model, (y_true, y_prob), style in _styled_curves(curves, highlight):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        label = f"{labels.MODEL_DISPLAY_NAMES.get(model, model)} (AUC {auc:.3f})".replace(".", ",")
        ax.plot(fpr, tpr, label=label, **style)
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1, label="Τυχαία πρόβλεψη")
    ax.set_xlabel("Ποσοστό ψευδώς θετικών (FPR)")
    ax.set_ylabel("Ποσοστό αληθώς θετικών (TPR)")
    ax.set_title("Καμπύλες ROC")
    ax.legend(loc="lower right", fontsize=8)
    return save_figure(fig, name)


def plot_pr(curves: dict, name: str, highlight: list[str] | None = None):
    """Καμπύλες precision–recall. `curves`: {μοντέλο: (y_true, y_prob)}."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    base_rate = None
    for model, (y_true, y_prob), style in _styled_curves(curves, highlight):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        label = f"{labels.MODEL_DISPLAY_NAMES.get(model, model)} (AP {ap:.3f})".replace(".", ",")
        ax.plot(recall, precision, label=label, **style)
        base_rate = float(np.mean(y_true))
    if base_rate is not None:
        ax.axhline(base_rate, color="black", linestyle=":", linewidth=1, label="Τυχαία πρόβλεψη")
    ax.set_xlabel("Ανάκληση")
    ax.set_ylabel("Ακρίβεια")
    ax.set_title("Καμπύλες ακρίβειας–ανάκλησης")
    ax.legend(loc="upper right", fontsize=8)
    return save_figure(fig, name)
