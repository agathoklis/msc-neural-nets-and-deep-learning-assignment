"""Σύγκριση τεσσάρων μεθόδων hyperparameter tuning (ερώτημα 3).

Μέθοδοι: Random Search, Bayesian Optimization και Hyperband του KerasTuner, και TPE του Optuna.

Για να είναι δίκαιη η σύγκριση:
1. **Ίδιος χώρος αναζήτησης.** Όλες διαβάζουν το `NN_SEARCH_SPACE` του config μέσα από τους δύο
   «προσαρμογείς» (`kt_params`, `optuna_params`).
2. **Ίδιος προϋπολογισμός σε εποχές, όχι σε δοκιμές.** Το Hyperband κάνει πολλές σύντομες δοκιμές,
   οπότε η σύγκριση με ίσο αριθμό δοκιμών θα ήταν άδικη. Τρέχει πρώτο, οι εποχές που κατανάλωσε
   ορίζουν τον προϋπολογισμό B, και οι άλλες τρεις σταματούν μόλις τον φτάσουν.
3. **Ίδια δεδομένα, ίδια callbacks, ίδιο κριτήριο** (το καλύτερο val AUC).
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import shutil
import time
import warnings

import numpy as np
import keras_tuner as kt
import matplotlib.pyplot as plt
import optuna
import pandas as pd

from src.config import MODELS_DIR, NN_MAX_EPOCHS, NN_SEARCH_SPACE, SEED, TABLES_DIR, set_seeds
from src.nn_models import build_advanced_mlp
from src.nn_train import BEST_PARAMS_PATH, EpochLog, NNData, fit_model, prepare_nn_data
from src.plotting import save_figure

TUNING_DIR = MODELS_DIR / "tuning"
HYPERBAND_MAX_EPOCHS = 100
HYPERBAND_FACTOR = 3

# Τυχαίες δοκιμές πριν αρχίσουν να «μαθαίνουν» η Bayesian Optimization και το TPE. Η προεπιλογή του
# KerasTuner είναι 3 × (πλήθος υπερπαραμέτρων) = 27 δοκιμές. Με το ίδιο seed, αυτές είναι ίδιες με του
# Random Search, οπότε μέσα στον προϋπολογισμό η Bayesian δεν θα διέφερε ουσιαστικά από τη Random. Ίδια
# τιμή και για τις δύο «έξυπνες» μεθόδους, για δίκαιη σύγκριση.
N_INITIAL_RANDOM_TRIALS = 5

METHOD_LABELS = {
    "hyperband": "Hyperband (KerasTuner)",
    "random": "Random Search (KerasTuner)",
    "bayesian": "Bayesian Optimization (KerasTuner)",
    "optuna_tpe": "TPE (Optuna)",
}


# --- Προσαρμογείς του χώρου αναζήτησης ----------------------------------------------------------


def kt_params(hp) -> dict:
    """Μετατρέπει τον κοινό χώρο αναζήτησης σε υπερπαραμέτρους του KerasTuner."""
    params = {}
    for name, spec in NN_SEARCH_SPACE.items():
        if spec["type"] == "choice":
            params[name] = hp.Choice(name, spec["values"])
        else:
            params[name] = hp.Float(name, spec["min"], spec["max"], sampling="log")
    return params


def optuna_params(trial) -> dict:
    """Μετατρέπει τον κοινό χώρο αναζήτησης σε προτάσεις του Optuna."""
    params = {}
    for name, spec in NN_SEARCH_SPACE.items():
        if spec["type"] == "choice":
            params[name] = trial.suggest_categorical(name, spec["values"])
        else:
            params[name] = trial.suggest_float(name, spec["min"], spec["max"], log=True)
    return params


class ChurnHyperModel(kt.HyperModel):
    """Το προχωρημένο MLP ως HyperModel. Το `fit` χρησιμοποιεί το batch size που διάλεξε ο tuner."""

    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features

    def build(self, hp):
        return build_advanced_mlp(kt_params(hp), self.n_features)

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args, batch_size=hp.get("batch_size"), **kwargs)


# --- Οι τέσσερις αναζητήσεις ----------------------------------------------------------------------


def _search_kwargs(data: NNData, log: EpochLog) -> dict:
    """Ό,τι περνά στο `tuner.search`: ίδια δεδομένα, βάρη και callbacks με κάθε άλλη εκπαίδευση."""
    from src.nn_train import make_callbacks

    return {
        "x": data.X_train, "y": data.y_train,
        "validation_data": (data.X_val, data.y_val),
        "class_weight": data.class_weight,
        "callbacks": make_callbacks([log]),
        "verbose": 0,
    }


def _clean_params(values: dict) -> dict:
    """Κρατά μόνο τις υπερπαραμέτρους του χώρου αναζήτησης, με τον σωστό τύπο.

    Το KerasTuner αποθηκεύει τις λογικές τιμές ως 0/1 και το Hyperband προσθέτει εσωτερικά πεδία
    (π.χ. «tuner/epochs»). Εδώ τα πεδία αφαιρούνται και οι τιμές γίνονται ξανά True/False.
    """
    params = {}
    for name, spec in NN_SEARCH_SPACE.items():
        value = values[name]
        if spec["type"] == "choice" and isinstance(spec["values"][0], bool):
            value = bool(value)
        params[name] = value
    return params


def _kt_best(tuner) -> tuple[float, dict]:
    trial = tuner.oracle.get_best_trials(1)[0]
    return float(trial.score), _clean_params(trial.hyperparameters.values)


def _kt_top10(tuner) -> pd.DataFrame:
    rows = [{"val_auc": t.score, **_clean_params(t.hyperparameters.values)} for t in tuner.oracle.get_best_trials(10)]
    return pd.DataFrame(rows)


TUNING_SEEDS = (42, 43, 44)


def run_hyperband(data: NNData, seed: int = SEED) -> dict:
    """Hyperband: ξεκινά πολλές ρυθμίσεις με λίγες εποχές και συνεχίζει μόνο τις καλύτερες (Li et al., 2018)."""
    set_seeds(seed)
    log = EpochLog()
    tuner = kt.Hyperband(
        ChurnHyperModel(data.n_features), objective=kt.Objective("val_auc", "max"),
        max_epochs=HYPERBAND_MAX_EPOCHS, factor=HYPERBAND_FACTOR, hyperband_iterations=1,
        seed=seed, directory=TUNING_DIR, project_name=f"hyperband_s{seed}", overwrite=True,
    )
    start = time.time()
    tuner.search(**_search_kwargs(data, log))
    score, params = _kt_best(tuner)
    return {"log": log, "score": score, "params": params, "top10": _kt_top10(tuner), "seconds": time.time() - start}


def run_kt_budgeted(name: str, data: NNData, budget: int, seed: int = SEED) -> dict:
    """Random Search ή Bayesian Optimization, μέχρι να εξαντληθεί ο προϋπολογισμός εποχών.

    Ο tuner τρέχει μία δοκιμή τη φορά (αυξάνοντας το `max_trials` κατά 1), και μετά από κάθε δοκιμή
    ελέγχεται αν οι εποχές έφτασαν το B. Έτσι η υπέρβαση είναι το πολύ μία δοκιμή.
    """
    set_seeds(seed)
    log = EpochLog()
    common = dict(objective=kt.Objective("val_auc", "max"), max_trials=1, seed=seed,
                  directory=TUNING_DIR, project_name=f"{name}_s{seed}", overwrite=True)
    if name == "bayesian":
        tuner = kt.BayesianOptimization(ChurnHyperModel(data.n_features),
                                        num_initial_points=N_INITIAL_RANDOM_TRIALS, **common)
    else:
        tuner = kt.RandomSearch(ChurnHyperModel(data.n_features), **common)
    start = time.time()
    # Χωρίς ρητό `epochs`, το KerasTuner εκπαιδεύει για 1 μόνο εποχή. Δίνουμε το ίδιο μέγιστο με κάθε
    # άλλη εκπαίδευση· το early stopping σταματά νωρίτερα.
    kwargs = {**_search_kwargs(data, log), "epochs": NN_MAX_EPOCHS}
    while log.total_epochs < budget:
        tuner.search(**kwargs)
        tuner.oracle.max_trials += 1
    score, params = _kt_best(tuner)
    return {"log": log, "score": score, "params": params, "top10": _kt_top10(tuner), "seconds": time.time() - start}


def run_optuna(data: NNData, budget: int, seed: int = SEED) -> dict:
    """Optuna με τον sampler TPE (Akiba et al., 2019), χωρίς pruning, μέχρι τον ίδιο προϋπολογισμό εποχών."""
    set_seeds(seed)
    log = EpochLog()

    def objective(trial):
        params = optuna_params(trial)
        model = build_advanced_mlp(params, data.n_features)
        history = fit_model(model, data, batch_size=params["batch_size"], extra_callbacks=[log])
        return max(history.history["val_auc"])

    def stop_at_budget(study, trial):
        if log.total_epochs >= budget:
            study.stop()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=N_INITIAL_RANDOM_TRIALS),
                                study_name=f"optuna_tpe_s{seed}")
    start = time.time()
    study.optimize(objective, callbacks=[stop_at_budget])
    top = study.trials_dataframe().sort_values("value", ascending=False).head(10)
    top10 = top.rename(columns={"value": "val_auc"})[["val_auc"] + [f"params_{k}" for k in NN_SEARCH_SPACE]]
    top10.columns = [c.replace("params_", "") for c in top10.columns]
    return {"log": log, "score": float(study.best_value), "params": dict(study.best_params),
            "top10": top10.reset_index(drop=True), "seconds": time.time() - start}


# --- Αποτελέσματα -----------------------------------------------------------------------------------


def _best_so_far_curve(fits: list[dict], grid: np.ndarray) -> np.ndarray:
    """Το καλύτερο val AUC μέχρι κάθε σημείο του κοινού πλέγματος εποχών (NaN πριν την πρώτη δοκιμή)."""
    f = pd.DataFrame(fits)
    epochs, best = f["cumulative_epochs"].to_numpy(), f["best_val_auc"].cummax().to_numpy()
    idx = np.searchsorted(epochs, grid, side="right") - 1
    return np.where(idx >= 0, best[np.clip(idx, 0, None)], np.nan)


def curves_table(runs: list[dict]) -> pd.DataFrame:
    """Όλες οι εκπαιδεύσεις κάθε αναζήτησης (μέθοδος, seed, εποχές, καλύτερο val AUC).

    Αποθηκεύεται, ώστε το σχήμα σύγκλισης να μπορεί να ξανασχεδιαστεί (π.χ. με άλλες ετικέτες) χωρίς να
    ξανατρέξει η αναζήτηση, που είναι χρονοβόρα.
    """
    rows = [{"method": r["method"], "seed": r["seed"], "fit": i, **fit}
            for r in runs for i, fit in enumerate(r["log"].fits)]
    return pd.DataFrame(rows)


def plot_convergence(curves: pd.DataFrame):
    """Μέση καμπύλη σύγκλισης ανά μέθοδο (3 seeds), με ζώνη ±1 τυπικής απόκλισης."""
    max_epochs = int(curves.groupby(["method", "seed"])["cumulative_epochs"].max().max())
    grid = np.linspace(1, max_epochs, 400)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name in METHOD_LABELS:
        sub = curves[curves["method"] == name]
        curves_ = np.array([_best_so_far_curve(g.sort_values("fit").to_dict("records"), grid)
                            for _, g in sub.groupby("seed")])
        with warnings.catch_warnings():
            # Στην αρχή του πλέγματος κάποιες μέθοδοι δεν έχουν ακόμα ολοκληρώσει δοκιμή (NaN): αναμενόμενο.
            warnings.simplefilter("ignore", RuntimeWarning)
            mean, std = np.nanmean(curves_, axis=0), np.nanstd(curves_, axis=0)
        line, = ax.plot(grid, mean, linewidth=2, label=METHOD_LABELS[name])
        ax.fill_between(grid, mean - std, mean + std, color=line.get_color(), alpha=0.15)
    ax.set_xlabel("Συνολικές εποχές εκπαίδευσης")
    ax.set_ylabel("Καλύτερο ROC-AUC επικύρωσης μέχρι εκείνη τη στιγμή")
    ax.set_title(f"Σύγκριση μεθόδων αναζήτησης υπερπαραμέτρων (μέσος όρος {len(TUNING_SEEDS)} seeds, ±1 τυπική απόκλιση)")
    ax.legend(loc="lower right")
    return save_figure(fig, "tuning_convergence")


def plot_method_comparison(run_table: pd.DataFrame):
    """Σύγκριση των μεθόδων από τα αποτελέσματα κάθε εκτέλεσης (μέθοδος × seed): ποιότητα και κόστος.

    Αριστερά το καλύτερο ROC-AUC επικύρωσης (κάθε seed ως σημείο, μέσος όρος ±1 τ.α.). Δεξιά ο χρόνος σε
    λεπτά (λογαριθμική κλίμακα). Φτιάχνεται μόνο από τον πίνακα `tuning_runs.csv`, χωρίς νέα αναζήτηση.
    """
    order = list(METHOD_LABELS)
    names = [METHOD_LABELS[m] for m in order]
    y = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, col, xlabel in ((axes[0], "best_val_auc", "Καλύτερο ROC-AUC επικύρωσης"),
                            (axes[1], "minutes", "Χρόνος αναζήτησης (λεπτά, λογαριθμική κλίμακα)")):
        for yi, m in zip(y, order):
            vals = run_table.loc[run_table["method"] == m, col].to_numpy()
            ax.scatter(vals, np.full(len(vals), yi), color="#AAAAAA", zorder=2, s=25)
            ax.errorbar(vals.mean(), yi, xerr=vals.std(ddof=1), fmt="o", color="#DD8452", capsize=4, zorder=3)
        ax.set_xlabel(xlabel)
    axes[1].set_xscale("log")
    # Απλοί αριθμοί λεπτών αντί για δυνάμεις του 10 στον λογαριθμικό άξονα.
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200) if run_table["minutes"].min() / 1.5 <= t <= run_table["minutes"].max() * 1.5]
    axes[1].set_xticks(ticks, [str(t) for t in ticks])
    axes[1].minorticks_off()
    axes[0].set_yticks(y, names)
    fig.suptitle(f"Σύγκριση μεθόδων αναζήτησης υπερπαραμέτρων ({run_table['seed'].nunique()} seeds ανά μέθοδο)\n"
                 "γκρι = κάθε seed, πορτοκαλί = μέσος όρος ±1 τυπική απόκλιση", y=1.04)
    return save_figure(fig, "tuning_methods")


def run() -> None:
    """Τρέχει τις 4 αναζητήσεις με 3 seeds η καθεμία και αποθηκεύει τη σύγκριση και την καλύτερη ρύθμιση."""
    data = prepare_nn_data()
    if TUNING_DIR.exists():
        shutil.rmtree(TUNING_DIR)

    runs = []
    for seed in TUNING_SEEDS:
        hb = {"method": "hyperband", "seed": seed, **run_hyperband(data, seed)}
        budget = hb["log"].total_epochs
        hb["budget"] = budget
        runs.append(hb)
        print(f"[seed {seed}] hyperband: {hb['score']:.4f}, προϋπολογισμός {budget} εποχές", flush=True)
        for name in ("random", "bayesian"):
            r = {"method": name, "seed": seed, "budget": budget, **run_kt_budgeted(name, data, budget, seed)}
            runs.append(r)
            print(f"[seed {seed}] {name}: {r['score']:.4f}", flush=True)
        r = {"method": "optuna_tpe", "seed": seed, "budget": budget, **run_optuna(data, budget, seed)}
        runs.append(r)
        print(f"[seed {seed}] optuna_tpe: {r['score']:.4f}", flush=True)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    run_rows = [{"method": r["method"], "seed": r["seed"], "budget": r["budget"], "best_val_auc": r["score"],
                 "n_trials": len(r["log"].fits), "epochs_used": r["log"].total_epochs, "minutes": r["seconds"] / 60}
                for r in runs]
    run_table = pd.DataFrame(run_rows)
    run_table.round(4).to_csv(TABLES_DIR / "tuning_runs.csv", index=False)
    plot_method_comparison(run_table)

    summary = run_table.groupby("method", sort=False).agg(
        best_val_auc_mean=("best_val_auc", "mean"), best_val_auc_std=("best_val_auc", "std"),
        n_trials_mean=("n_trials", "mean"), epochs_mean=("epochs_used", "mean"),
        minutes_mean=("minutes", "mean"), minutes_std=("minutes", "std")).reset_index()
    summary.insert(1, "Μέθοδος", summary["method"].map(METHOD_LABELS))
    summary = summary.sort_values("best_val_auc_mean", ascending=False)
    summary.round(4).to_csv(TABLES_DIR / "tuning_comparison.csv", index=False)

    for name in METHOD_LABELS:
        top = pd.concat([r["top10"].assign(seed=r["seed"]) for r in runs if r["method"] == name])
        top.sort_values("val_auc", ascending=False).head(10).round(6).to_csv(
            TABLES_DIR / f"tuning_top10_{name}.csv", index=False)
    curves = curves_table(runs)
    curves.to_csv(TABLES_DIR / "tuning_curves.csv", index=False)
    plot_convergence(curves)

    best = max(runs, key=lambda r: r["score"])
    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(json.dumps(
        {"method": best["method"], "seed": best["seed"], "val_auc": best["score"],
         "budget_epochs": best["budget"], "params": best["params"]},
        ensure_ascii=False, indent=2,
    ))
    print(summary.round(4).to_string(index=False))
    print(f"Καλύτερη ρύθμιση: {best['method']} (seed {best['seed']}), val AUC {best['score']:.4f}: {best['params']}")
