"""Έλεγχοι του tuning: οι δύο προσαρμογείς δίνουν τον ίδιο χώρο αναζήτησης."""

import pytest

from src.config import NN_SEARCH_SPACE

kt = pytest.importorskip("keras_tuner")
optuna = pytest.importorskip("optuna")
from src.nn_tuning import kt_params, optuna_params  # noqa: E402


def _check(params: dict):
    assert set(params) == set(NN_SEARCH_SPACE)
    for name, value in params.items():
        spec = NN_SEARCH_SPACE[name]
        if spec["type"] == "choice":
            assert value in spec["values"]
        else:
            assert spec["min"] <= value <= spec["max"]


def test_kerastuner_adapter_matches_space():
    hp = kt.HyperParameters()
    _check(kt_params(hp))
    # Όλες οι υπερπαράμετροι καταχωρίστηκαν στον χώρο του KerasTuner.
    assert set(hp.values) == set(NN_SEARCH_SPACE)


def test_optuna_adapter_matches_space():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(20):
        trial = study.ask()
        _check(optuna_params(trial))
        study.tell(trial, 0.5)
