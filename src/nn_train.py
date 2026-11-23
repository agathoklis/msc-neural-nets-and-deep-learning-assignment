"""Εκπαίδευση του νευρωνικού: απλό MLP, τελικό μοντέλο και ablation.

Όλες οι εκπαιδεύσεις, και αυτές μέσα στο tuning, περνούν από τα ίδια `prepare_nn_data`,
`make_callbacks` και `fit_model`. Έτσι κάθε δοκιμή έχει τα ίδια δεδομένα, τα ίδια βάρη κλάσεων και τα
ίδια κριτήρια διακοπής.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # λιγότερα μηνύματα του TensorFlow στην κονσόλα

import json
from dataclasses import dataclass

import joblib
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    METRICS_DIR,
    MODELS_DIR,
    NN_EARLY_STOPPING_PATIENCE,
    NN_MAX_EPOCHS,
    NN_REDUCE_LR_FACTOR,
    NN_REDUCE_LR_PATIENCE,
    SEED,
    TABLES_DIR,
    set_seeds,
)
from src.data import load_splits
from src.evaluation import compute_metrics, save_model_metrics, save_predictions, tune_threshold
from src.imbalance import keras_class_weight, load_choice, resample
from src.nn_models import build_advanced_mlp, build_simple_mlp
from src.plotting import COLOR_CHURN, COLOR_STAY, save_figure
from src.preprocessing import build_preprocessor

PREPROCESSOR_PATH = MODELS_DIR / "nn_preprocessor.joblib"
BEST_PARAMS_PATH = METRICS_DIR / "nn_tuning_best.json"


# --- Δεδομένα ---------------------------------------------------------------------------------


@dataclass
class NNData:
    """Τα δεδομένα του νευρωνικού σε μορφή πινάκων float32, μαζί με ό,τι χρειάζεται για την αξιολόγηση."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    val_index: pd.Index
    train_index: pd.Index
    class_weight: dict | None
    n_features: int


def prepare_nn_data(save_preprocessor: bool = False) -> NNData:
    """Προετοιμάζει train και validation για το Keras.

    Ο preprocessor μαθαίνει μόνο από το train, όπως και στα άλλα μοντέλα. Η μέθοδος ανισορροπίας
    έρχεται από την ανάλυση ανισορροπίας: με class weights, τα δεδομένα μένουν ίδια και τα βάρη περνούν στο `fit`·
    με resampling, αλλάζει μόνο το train.
    """
    X_train, y_train, X_val, y_val, *_ = load_splits()
    method = load_choice()
    train_index = X_train.index
    X_fit, y_fit = resample(X_train, y_train, method)

    prep = build_preprocessor().fit(X_fit)
    if save_preprocessor:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(prep, PREPROCESSOR_PATH)

    return NNData(
        X_train=prep.transform(X_fit).astype("float32"),
        y_train=np.asarray(y_fit, dtype="float32"),
        X_val=prep.transform(X_val).astype("float32"),
        y_val=np.asarray(y_val, dtype="float32"),
        val_index=X_val.index,
        train_index=train_index,
        class_weight=keras_class_weight(np.asarray(y_fit), method),
        n_features=prep.transform(X_fit.head(1)).shape[1],
    )


# --- Callbacks --------------------------------------------------------------------------------


class EpochLog(keras.callbacks.Callback):
    """Καταγράφει κάθε εκπαίδευση: πόσες εποχές έτρεξε και ποιο ήταν το καλύτερο val AUC της.

    Στη σύγκριση των μεθόδων tuning, ένα κοινό αντικείμενο μετρά τις εποχές όλων των δοκιμών μιας
    μεθόδου. Έτσι κάθε μέθοδος σταματά στον ίδιο συνολικό «προϋπολογισμό» εποχών.
    """

    def __init__(self):
        super().__init__()
        self.total_epochs = 0
        self.fits: list[dict] = []
        self._current: list[float] = []

    def __deepcopy__(self, memo):
        # Το KerasTuner αντιγράφει (deepcopy) τα callbacks σε κάθε δοκιμή. Αν αντιγραφόταν και ο
        # μετρητής, κάθε δοκιμή θα μετρούσε σε δικό της αντίγραφο και το σύνολο θα έμενε 0. Επιστρέφοντας
        # τον εαυτό του, ο μετρητής μένει ένας και κοινός για όλες τις δοκιμές μιας μεθόδου.
        return self

    def on_train_begin(self, logs=None):
        self._current = []

    def on_epoch_end(self, epoch, logs=None):
        self.total_epochs += 1
        self._current.append(float((logs or {}).get("val_auc", np.nan)))

    def on_train_end(self, logs=None):
        if self._current:
            self.fits.append(
                {"epochs": len(self._current), "best_val_auc": float(np.nanmax(self._current)),
                 "cumulative_epochs": self.total_epochs}
            )


def make_callbacks(extra: list | None = None) -> list:
    """Τα callbacks κάθε εκπαίδευσης του νευρωνικού.

    - Early stopping: σταματά όταν το val AUC δεν βελτιωθεί για 15 εποχές και επαναφέρει τα βάρη της
      καλύτερης εποχής. Είναι ο βασικός μηχανισμός ελέγχου του overfitting (Prechelt, 1998).
    - ReduceLROnPlateau: αν το ROC-AUC επικύρωσης δεν βελτιωθεί για 5 εποχές, μειώνει το learning rate στο μισό, ώστε
      η βελτιστοποίηση να κάνει μικρότερα, πιο προσεκτικά βήματα.
    """
    return [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=NN_EARLY_STOPPING_PATIENCE,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=NN_REDUCE_LR_FACTOR,
                                          patience=NN_REDUCE_LR_PATIENCE),
        *(extra or []),
    ]


def fit_model(model, data: NNData, batch_size: int, epochs: int = NN_MAX_EPOCHS, extra_callbacks=None):
    """Εκπαιδεύει ένα μοντέλο με τα κοινά δεδομένα και callbacks και επιστρέφει το history."""
    return model.fit(
        data.X_train, data.y_train,
        validation_data=(data.X_val, data.y_val),
        epochs=epochs, batch_size=batch_size,
        class_weight=data.class_weight,
        callbacks=make_callbacks(extra_callbacks),
        verbose=0,
    )


# --- Αξιολόγηση και αποθήκευση ------------------------------------------------------------------


def plot_curves(history, name: str, title: str):
    """Καμπύλες εκπαίδευσης: loss και AUC, σε train και validation.

    Αν το train συνεχίζει να βελτιώνεται ενώ το validation χειροτερεύει, υπάρχει overfitting. Η
    κάθετη γραμμή δείχνει την εποχή που κράτησε το early stopping.
    """
    h = history.history
    best_epoch = int(np.argmax(h["val_auc"])) + 1
    epochs = np.arange(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in ((axes[0], "loss", "Απώλεια (binary cross-entropy)"), (axes[1], "auc", "ROC-AUC")):
        ax.plot(epochs, h[key], color=COLOR_STAY, label="Εκπαίδευση")
        ax.plot(epochs, h[f"val_{key}"], color=COLOR_CHURN, label="Επικύρωση")
        ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1, label=f"Καλύτερη εποχή ({best_epoch})")
        ax.set_xlabel("Εποχή")
        ax.set_ylabel(label)
        ax.legend()
    fig.suptitle(title, y=1.02)
    return save_figure(fig, name)


def evaluate_and_save(model, data: NNData, model_name: str, params: dict, extra: dict | None = None) -> dict:
    """Προβλέψεις σε train και validation, επιλογή threshold στο validation και αποθήκευση."""
    prob_train = model.predict(data.X_train, verbose=0).ravel()
    prob_val = model.predict(data.X_val, verbose=0).ravel()
    threshold = tune_threshold(data.y_val, prob_val)
    # Με resampling, οι γραμμές του train δεν αντιστοιχούν πια σε πραγματικούς πελάτες· τότε δεν
    # αποθηκεύουμε προβλέψεις train (δεν χρησιμοποιούνται πουθενά στην αξιολόγηση).
    if len(data.train_index) == len(prob_train):
        save_predictions(model_name, "train", data.train_index, data.y_train.astype(int), prob_train)
    save_predictions(model_name, "val", data.val_index, data.y_val.astype(int), prob_val)
    val_metrics = {
        "at_0.5": compute_metrics(data.y_val, prob_val, 0.5),
        "at_tuned": compute_metrics(data.y_val, prob_val, threshold),
    }
    save_model_metrics(model_name, threshold, val_metrics, params, extra=extra)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / f"{model_name}.keras")
    return val_metrics["at_tuned"]


# --- Στάδια του pipeline ------------------------------------------------------------------------


def run_simple() -> None:
    """Το απλό MLP (σημείο αναφοράς): batch 64, learning rate 1e-3, ίδια callbacks με όλα."""
    set_seeds(SEED)
    data = prepare_nn_data(save_preprocessor=True)
    model = build_simple_mlp(data.n_features, lr=1e-3)
    history = fit_model(model, data, batch_size=64)
    plot_curves(history, "nn_simple_curves", "Απλό MLP: καμπύλες εκπαίδευσης")
    stop_epoch = len(history.history["loss"])
    m = evaluate_and_save(model, data, "mlp_simple", {"batch_size": 64, "lr": 1e-3},
                          extra={"epochs_run": stop_epoch,
                                 "best_epoch": int(np.argmax(history.history["val_auc"])) + 1})
    print(f"Απλό MLP: val ROC-AUC {m['roc_auc']:.4f}, F1 {m['f1']:.4f}, εποχές {stop_epoch}")


def load_best_params() -> dict:
    """Η καλύτερη ρύθμιση από τη σύγκριση των μεθόδων tuning."""
    return json.loads(BEST_PARAMS_PATH.read_text())["params"]


def train_advanced(params: dict, data: NNData, seed: int = SEED):
    """Εκπαιδεύει από την αρχή ένα προχωρημένο MLP με τις δοσμένες παραμέτρους."""
    set_seeds(seed)
    model = build_advanced_mlp(params, data.n_features)
    history = fit_model(model, data, batch_size=params["batch_size"])
    return model, history


ENSEMBLE_SEEDS = (42, 43, 44, 45, 46)


class SeedEnsemble:
    """Μέσος όρος των πιθανοτήτων πολλών νευρωνικών με την ίδια ρύθμιση και διαφορετικό seed.

    Το αποτέλεσμα ενός νευρωνικού «κουνιέται» από seed σε seed (±0,005 AUC), όσο και οι διαφορές που μας
    ενδιαφέρουν. Ο μέσος όρος 5 εκδοχών μειώνει αυτή τη διακύμανση και δίνει πιο αξιόπιστο μοντέλο.
    """

    def __init__(self, models: list):
        self.models = models

    def predict(self, X, verbose=0, batch_size=None):
        kwargs = {"verbose": verbose} | ({"batch_size": batch_size} if batch_size else {})
        return np.mean([m.predict(X, **kwargs).ravel() for m in self.models], axis=0)

    def save(self, path):
        """Αποθηκεύει κάθε μοντέλο χωριστά: `mlp_tuned_seed<k>.keras`."""
        for seed, m in zip(ENSEMBLE_SEEDS, self.models):
            m.save(path.with_name(f"{path.stem}_seed{seed}.keras"))

    @classmethod
    def load(cls, name: str = "mlp_tuned"):
        return cls([keras.models.load_model(MODELS_DIR / f"{name}_seed{s}.keras") for s in ENSEMBLE_SEEDS])


def _seed_row(model_name: str, seed: int, model, data: NNData, history) -> dict:
    prob = model.predict(data.X_val, verbose=0).ravel()
    t = tune_threshold(data.y_val, prob)
    m = compute_metrics(data.y_val, prob, t)
    return {"model": model_name, "seed": seed, "val_auc": m["roc_auc"], "val_f1": m["f1"],
            "best_epoch": int(np.argmax(history.history["val_auc"])) + 1, "epochs_run": len(history.history["loss"])}


def run_final() -> None:
    """Το τελικό νευρωνικό: η καλύτερη ρύθμιση, εκπαιδευμένη με 5 seeds, ως ensemble.

    Καταγράφεται και η διακύμανση ανάμεσα στα seeds, για το τελικό MLP και (για δίκαιη σύγκριση) για το απλό.
    """
    params = load_best_params()
    data = prepare_nn_data(save_preprocessor=True)
    rows, models = [], []
    for seed in ENSEMBLE_SEEDS:
        model, history = train_advanced(params, data, seed=seed)
        models.append(model)
        rows.append(_seed_row("mlp_tuned", seed, model, data, history))
        if seed == SEED:
            plot_curves(history, "nn_tuned_curves", f"Τελικό MLP: καμπύλες εκπαίδευσης (seed {seed})")
    for seed in ENSEMBLE_SEEDS:
        set_seeds(seed)
        simple = build_simple_mlp(data.n_features, lr=1e-3)
        history = fit_model(simple, data, batch_size=64)
        rows.append(_seed_row("mlp_simple", seed, simple, data, history))

    ensemble = SeedEnsemble(models)
    m = evaluate_and_save(ensemble, data, "mlp_tuned", params,
                          extra={"ensemble_seeds": list(ENSEMBLE_SEEDS)})

    table = pd.DataFrame(rows)
    summary = table.groupby("model").agg(val_auc=("val_auc", "mean"), val_f1=("val_f1", "mean"),
                                         best_epoch=("best_epoch", "mean"), epochs_run=("epochs_run", "mean"))
    summary_std = table.groupby("model")[["val_auc", "val_f1"]].std()
    extra_rows = [{"model": f"{name}_mean", "seed": "mean", **summary.loc[name].to_dict()} for name in summary.index]
    extra_rows += [{"model": f"{name}_std", "seed": "std", **summary_std.loc[name].to_dict()} for name in summary_std.index]
    extra_rows.append({"model": "mlp_tuned_ensemble", "seed": "ensemble", "val_auc": m["roc_auc"], "val_f1": m["f1"]})
    out = pd.concat([table, pd.DataFrame(extra_rows)], ignore_index=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(TABLES_DIR / "nn_seed_variability.csv", index=False)
    print(f"Τελικό MLP: μεμονωμένα seeds {summary.loc['mlp_tuned', 'val_auc']:.4f} ± "
          f"{summary_std.loc['mlp_tuned', 'val_auc']:.4f}, ensemble val ROC-AUC {m['roc_auc']:.4f}, F1 {m['f1']:.4f}")


ABLATION_SEEDS = (42, 43, 44, 45, 46)

# Οι τεχνικές και η «ενεργή» τιμή τους όταν προστίθενται σε απλό δίκτυο (κατεύθυνση β).
TECHNIQUES = {
    "bn": ({"use_bn": False}, {"use_bn": True}, "Batch Normalization"),
    "dropout": ({"dropout": 0.0}, {"dropout": 0.2}, "dropout"),
    "l1l2": ({"l1": 0.0, "l2": 0.0}, {"l2": 1e-4}, "ποινές L1/L2"),
    "residual": ({"use_residual": False}, {"use_residual": True}, "υπολειμματικές συνδέσεις"),
}

# Ετικέτες της κατεύθυνσης «προσθήκη» (με την τιμή που προστίθεται), όπως στον Πίνακα 6 της εργασίας.
ADD_LABELS = {
    "bn": "Batch Normalization",
    "dropout": "dropout 0,2",
    "l1l2": "ποινή L2 10⁻⁴",
    "residual": "υπολειμματικές συνδέσεις",
}


def _evaluate_variant(params: dict, data: NNData) -> dict:
    """Εκπαιδεύει μια παραλλαγή με όλα τα seeds του ablation και επιστρέφει μέσους όρους στο validation."""
    aucs, f1s, stops = [], [], []
    for seed in ABLATION_SEEDS:
        model, history = train_advanced(params, data, seed=seed)
        prob = model.predict(data.X_val, verbose=0).ravel()
        m = compute_metrics(data.y_val, prob, tune_threshold(data.y_val, prob))
        aucs.append(m["roc_auc"])
        f1s.append(m["f1"])
        stops.append(len(history.history["loss"]))
    return {"val_auc_mean": np.mean(aucs), "val_auc_std": np.std(aucs), "val_f1_mean": np.mean(f1s),
            "stop_epoch_mean": np.mean(stops)}


def _is_noop(params: dict, change: dict) -> bool:
    """True αν η αλλαγή δεν αλλάζει τίποτα (π.χ. «χωρίς dropout» όταν το dropout είναι ήδη 0)."""
    return all(params.get(k) == v for k, v in change.items())


def plot_ablation(table: pd.DataFrame):
    """Οριζόντιες μπάρες val AUC ±1 τ.α., ομαδοποιημένες ανά κατεύθυνση του ablation."""
    data = table.dropna(subset=["val_auc_mean"]).iloc[::-1]
    colors = ["#4C72B0" if d == "remove" else "#DD8452" for d in data["direction"]]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(data) + 1.5))
    ax.barh(data["Παραλλαγή"], data["val_auc_mean"], xerr=data["val_auc_std"], color=colors, capsize=3)
    lo = (data["val_auc_mean"] - data["val_auc_std"]).min()
    ax.set_xlim(lo - 0.005, (data["val_auc_mean"] + data["val_auc_std"]).max() + 0.003)
    ax.set_xlabel(f"ROC-AUC επικύρωσης (μέσος όρος {len(ABLATION_SEEDS)} seeds, ±1 τυπική απόκλιση)")
    ax.set_title("Μελέτη αφαίρεσης: αφαίρεση από την τελική ρύθμιση (μπλε) και προσθήκη σε δίκτυο χωρίς τεχνικές (πορτοκαλί)")
    return save_figure(fig, "nn_ablation")


def run_ablation() -> None:
    """Ablation και προς τις δύο κατευθύνσεις, μόνο στο validation, με 5 seeds.

    (α) Αφαιρείται μία τεχνική τη φορά από την τελική ρύθμιση. Αν η ρύθμιση δεν χρησιμοποιεί ήδη μια
        τεχνική, η γραμμή σημειώνεται ως «δεν εφαρμόζεται» αντί να εκπαιδευτεί ένα πανομοιότυπο μοντέλο.
    (β) Προστίθεται μία τεχνική τη φορά σε ένα απλό δίκτυο ίδιου πλάτους, βάθους, lr και batch, και στο
        τέλος όλες μαζί. Έτσι μετράται τι προσφέρει κάθε τεχνική ακόμα κι αν το tuning δεν τη διάλεξε.
    """
    best = load_best_params()
    data = prepare_nn_data()
    rows = []

    full = _evaluate_variant(best, data)
    rows.append({"direction": "remove", "variant": "full", "Παραλλαγή": "Τελική ρύθμιση", **full})
    for key, (off, _, label) in TECHNIQUES.items():
        row = {"direction": "remove", "variant": f"no_{key}", "Παραλλαγή": f"Τελική χωρίς {label}"}
        if _is_noop(best, off):
            row["note"] = "δεν εφαρμόζεται: η τελική ρύθμιση δεν τη χρησιμοποιεί"
        else:
            row.update(_evaluate_variant({**best, **off}, data))
        rows.append(row)
        print(f"remove {key}: {row.get('val_auc_mean', 'n/a')}", flush=True)

    plain = {**best, **{k: v for off, _, _ in TECHNIQUES.values() for k, v in off.items()}}
    base = _evaluate_variant(plain, data)
    rows.append({"direction": "add", "variant": "plain", "Παραλλαγή": "Δίκτυο χωρίς τεχνικές", **base})
    for key, (_, on, label) in TECHNIQUES.items():
        rows.append({"direction": "add", "variant": f"plus_{key}", "Παραλλαγή": f"Χωρίς τεχνικές + {ADD_LABELS[key]}",
                     **_evaluate_variant({**plain, **on}, data)})
        print(f"add {key}: {rows[-1]['val_auc_mean']:.4f}", flush=True)
    all_on = {**plain, **{k: v for _, on, _ in TECHNIQUES.values() for k, v in on.items()}}
    rows.append({"direction": "add", "variant": "plus_all", "Παραλλαγή": "Χωρίς τεχνικές + όλες οι τεχνικές",
                 **_evaluate_variant(all_on, data)})

    table = pd.DataFrame(rows)
    ref = {"remove": full["val_auc_mean"], "add": base["val_auc_mean"]}
    table["delta_auc_vs_reference"] = table["val_auc_mean"] - table["direction"].map(ref)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(TABLES_DIR / "nn_ablation.csv", index=False)
    plot_ablation(table)
    print(table[["direction", "variant", "val_auc_mean", "val_auc_std", "delta_auc_vs_reference"]].round(4).to_string(index=False))
