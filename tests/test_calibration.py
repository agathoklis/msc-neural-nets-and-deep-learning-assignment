"""Έλεγχοι της βαθμονόμησης και της στατιστικής σύγκρισης: σύγκριση ανά ζεύγος, διόρθωση Holm, βαθμονόμηση."""

import numpy as np
from sklearn.metrics import roc_auc_score

from src.calibration import PlattCalibrator, expected_calibration_error, holm, paired_bootstrap_auc


def test_identical_predictions_give_zero_delta():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = rng.random(400)
    res = paired_bootstrap_auc(y, p, p, n=200)
    assert res["delta_auc"] == 0 and res["p_value"] == 1.0


def test_clearly_better_model_is_significant():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    good = np.clip(y * 0.6 + rng.random(1000) * 0.4, 0, 1)
    bad = rng.random(1000)
    res = paired_bootstrap_auc(y, good, bad, n=300)
    assert res["delta_auc"] > 0.2 and res["p_value"] < 0.01 and res["ci_low"] > 0


def test_holm_known_values():
    # p = [0.01, 0.04, 0.03] → ταξινομημένα 0.01·3=0.03, 0.03·2=0.06, 0.04·1=0.04 → max-σωρευτικά 0.03, 0.06, 0.06
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_platt_keeps_ranking_and_fixes_inflated_probabilities():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    inflated = np.clip(0.3 + 0.5 * y + rng.normal(0, 0.2, 2000), 0.01, 0.99)
    cal = PlattCalibrator().fit(inflated[:1000], y[:1000]).transform(inflated[1000:])
    assert abs(roc_auc_score(y[1000:], cal) - roc_auc_score(y[1000:], inflated[1000:])) < 1e-9
    assert expected_calibration_error(y[1000:], cal) < expected_calibration_error(y[1000:], inflated[1000:])
