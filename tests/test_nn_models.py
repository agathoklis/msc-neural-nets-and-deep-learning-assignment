"""Έλεγχοι των αρχιτεκτονικών: τα μοντέλα χτίζονται, έχουν τη σωστή δομή και εκπαιδεύονται."""

import numpy as np
import pytest

from src.config import N_ENCODED_FEATURES, NN_DEFAULT_PARAMS

keras = pytest.importorskip("keras")
from src.nn_models import build_advanced_mlp, build_simple_mlp  # noqa: E402


def _layer_types(model):
    return {type(layer).__name__ for layer in model.layers}


def test_simple_mlp_output_shape():
    model = build_simple_mlp(N_ENCODED_FEATURES)
    assert model.output_shape == (None, 1)


def test_advanced_mlp_has_all_techniques():
    model = build_advanced_mlp(NN_DEFAULT_PARAMS, N_ENCODED_FEATURES)
    assert model.output_shape == (None, 1)
    types = _layer_types(model)
    assert {"Add", "BatchNormalization", "Dropout"} <= types


def test_switches_remove_techniques():
    """Με use_residual=False και use_bn=False, δεν πρέπει να υπάρχουν Add και BatchNormalization."""
    params = {**NN_DEFAULT_PARAMS, "use_residual": False, "use_bn": False}
    types = _layer_types(build_advanced_mlp(params, N_ENCODED_FEATURES))
    assert "Add" not in types and "BatchNormalization" not in types


def test_one_training_step_runs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, N_ENCODED_FEATURES)).astype("float32")
    y = rng.integers(0, 2, 64).astype("float32")
    for model in (build_simple_mlp(N_ENCODED_FEATURES), build_advanced_mlp(NN_DEFAULT_PARAMS, N_ENCODED_FEATURES)):
        history = model.fit(X, y, epochs=1, batch_size=32, verbose=0)
        assert np.isfinite(history.history["loss"][0])
