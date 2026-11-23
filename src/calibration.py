"""Βαθμονόμηση των πιθανοτήτων και στατιστική σύγκριση μοντέλων ανά ζεύγος.

**Βαθμονόμηση.** Τα class weights «φουσκώνουν» τις πιθανότητες: το μοντέλο λέει κατά μέσο όρο ~41% ρίσκο,
ενώ αποχωρεί το 26,5%. Η κατάταξη των πελατών δεν επηρεάζεται, αλλά οι πιθανότητες δεν διαβάζονται ως
πραγματικές. Το Platt scaling (Platt, 1999) ταιριάζει μια λογιστική καμπύλη από τις «ωμές» πιθανότητες στις
πραγματικές συχνότητες. Εκπαιδεύεται **μόνο στο validation**. Επειδή είναι μονότονη συνάρτηση, δεν αλλάζει
την κατάταξη, άρα ούτε το ROC-AUC.

**Σύγκριση ανά ζεύγος.** Όλα τα μοντέλα αξιολογούνται στους ίδιους πελάτες του test. Με paired bootstrap
(Efron & Tibshirani, 1993) επαναδειγματοληπτούμε τους πελάτες και υπολογίζουμε κάθε φορά τη *διαφορά* AUC
των δύο μοντέλων. Αυτό είναι πολύ πιο ισχυρό τεστ από το να κοιτάμε αν επικαλύπτονται τα διαστήματα.
Επειδή κάνουμε πολλές συγκρίσεις, οι p-values διορθώνονται με τη μέθοδο Holm (1979).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.config import SEED

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Platt scaling: λογιστική παλινδρόμηση πάνω στο logit της «ωμής» πιθανότητας."""

    def fit(self, prob_val, y_val):
        self.lr = LogisticRegression(C=1e6).fit(_logit(prob_val).reshape(-1, 1), np.asarray(y_val))
        return self

    def transform(self, prob):
        return self.lr.predict_proba(_logit(prob).reshape(-1, 1))[:, 1]


def expected_calibration_error(y, prob, n_bins: int = 10) -> float:
    """ECE: μέση απόσταση ανάμεσα σε προβλεπόμενη πιθανότητα και πραγματική συχνότητα, σε 10 ίσα διαστήματα."""
    y, prob = np.asarray(y), np.asarray(prob)
    bins = np.minimum((prob * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(prob[mask].mean() - y[mask].mean())
    return float(ece)


def calibration_summary(y, prob_raw, prob_cal) -> dict:
    return {
        "brier_raw": float(brier_score_loss(y, prob_raw)), "brier_calibrated": float(brier_score_loss(y, prob_cal)),
        "ece_raw": expected_calibration_error(y, prob_raw), "ece_calibrated": expected_calibration_error(y, prob_cal),
        "mean_prob_raw": float(np.mean(prob_raw)), "mean_prob_calibrated": float(np.mean(prob_cal)),
        "actual_rate": float(np.mean(y)),
    }


# --- Σύγκριση ανά ζεύγος ------------------------------------------------------------------------------


def paired_bootstrap_auc(y, prob_a, prob_b, n: int = 2000, seed: int = SEED) -> dict:
    """ΔAUC = AUC(a) − AUC(b) με paired bootstrap στις ίδιες γραμμές.

    Η p-value (δίπλευρη) είναι 2 × min(μερίδιο Δ ≤ 0, μερίδιο Δ ≥ 0): πόσο συχνά, σε ένα άλλο εξίσου
    πιθανό δείγμα πελατών, η διαφορά θα άλλαζε πρόσημο.
    """
    y, a, b = np.asarray(y), np.asarray(prob_a), np.asarray(prob_b)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        deltas.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    deltas = np.asarray(deltas)
    point = roc_auc_score(y, a) - roc_auc_score(y, b)
    if np.allclose(deltas, 0):
        p = 1.0
    else:
        p = min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {"delta_auc": float(point), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(p)}


def holm(p_values) -> np.ndarray:
    """Διόρθωση Holm για πολλαπλές συγκρίσεις: ελέγχει την πιθανότητα έστω και ενός ψευδούς «σημαντικού»."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def paired_family(y, probs: dict, reference: str, family: str) -> pd.DataFrame:
    """Όλα τα μοντέλα σε σύγκριση με ένα μοντέλο αναφοράς, με διόρθωση Holm μέσα στην οικογένεια."""
    rows = [{"family": family, "model_a": m, "model_b": reference, **paired_bootstrap_auc(y, probs[m], probs[reference])}
            for m in probs if m != reference]
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p_value"])
    table["significant_at_0.05"] = table["p_holm"] < 0.05
    return table
