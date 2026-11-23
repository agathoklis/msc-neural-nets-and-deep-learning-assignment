"""Τα 7 μοντέλα σύγκρισης (ερώτημα 4).

Κάθε μοντέλο εκπαιδεύεται με τον ίδιο τρόπο:
- ίδια μέθοδος ανισορροπίας (από την ανάλυση ανισορροπίας)·
- αναζήτηση υπερπαραμέτρων με 5-fold CV στο train και κριτήριο ROC-AUC·
- threshold στο validation·
- αποθήκευση προβλέψεων, μετρικών και μοντέλου με τα ίδια ονόματα αρχείων.

Όλα, εκτός από το CatBoost, περνούν από τον κοινό preprocessor. Το CatBoost παίρνει τις κατηγορικές
στήλες ως έχουν, γιατί είναι φτιαγμένο να τις χειρίζεται μόνο του (Prokhorenkova et al., 2018).
"""

import joblib
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import CATEGORICAL, MODELS_DIR, SEED
from src.data import load_splits
from src.evaluation import (
    compute_metrics,
    grid_edge_rows,
    save_model_metrics,
    save_predictions,
    tune_threshold,
    update_grid_edge_report,
    search_flatness_row,
    update_search_flatness,
)
from src.imbalance import class_weight_kwargs, load_choice, make_pipeline


def model_specs(y_train, method: str) -> dict:
    """Για κάθε μοντέλο: ο estimator, το είδος αναζήτησης, ο χώρος αναζήτησης και το πλήθος δοκιμών.

    Τα ίδια τα μοντέλα τρέχουν σε ένα νήμα (n_jobs=1), ενώ η αναζήτηση τρέχει παράλληλα. Αν
    παραλληλίζονταν και τα δύο, οι πυρήνες θα «μάλωναν» και όλα θα γίνονταν πιο αργά.
    """
    cw = lambda name: class_weight_kwargs(name, method, y_train)  # noqa: E731
    return {
        "rf": dict(
            estimator=RandomForestClassifier(random_state=SEED, n_jobs=1, **cw("rf")),
            search="random", n_iter=30,
            # Σε πρώτη δοκιμή το min_samples_leaf βγήκε στη μέγιστη τιμή (10)· επεκτάθηκε.
            space={"n_estimators": [200, 400, 800], "max_depth": [None, 5, 10, 20],
                   "min_samples_leaf": [1, 2, 5, 10, 20, 50], "max_features": ["sqrt", 0.3, 0.5]},
        ),
        "xgb": dict(
            estimator=XGBClassifier(eval_metric="auc", random_state=SEED, n_jobs=1, **cw("xgb")),
            search="random", n_iter=40,
            # Σε πρώτη δοκιμή n_estimators (200, ελάχιστο) και min_child_weight (10, μέγιστο) στην άκρη.
            space={"n_estimators": [50, 100, 200, 400, 800], "max_depth": [2, 3, 4, 6], "learning_rate": [0.01, 0.03, 0.1],
                   "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
                   "min_child_weight": [1, 5, 10, 20, 50]},
        ),
        "lgbm": dict(
            estimator=LGBMClassifier(random_state=SEED, subsample_freq=1, verbose=-1, n_jobs=1, **cw("lgbm")),
            search="random", n_iter=40,
            # Σε πρώτη δοκιμή 6 υπερπαράμετροι στην «συντηρητική» άκρη· όλες επεκτάθηκαν προς τα εκεί.
            space={"n_estimators": [50, 100, 200, 400, 800], "num_leaves": [4, 7, 15, 31, 63],
                   "learning_rate": [0.003, 0.005, 0.01, 0.03, 0.1], "min_child_samples": [10, 20, 50, 100, 200],
                   "subsample": [0.5, 0.6, 0.7, 0.85, 1.0], "colsample_bytree": [0.3, 0.45, 0.6, 0.8, 1.0]},
        ),
        "catboost": dict(
            # Οι κατηγορικές στήλες δίνονται στο fit και όχι στον constructor: το CatBoost αλλάζει
            # εσωτερικά την παράμετρο cat_features, και τότε το scikit-learn δεν μπορεί να αντιγράψει
            # (clone) το μοντέλο στο cross-validation.
            estimator=CatBoostClassifier(random_seed=SEED, verbose=0, thread_count=1, **cw("catboost")),
            search="random", n_iter=20, preprocess=False, fit_params={"cat_features": CATEGORICAL},
            # Σε πρώτη δοκιμή iterations (300) και depth (4) στο ελάχιστο, l2_leaf_reg (10) στο μέγιστο.
            space={"iterations": [100, 200, 300, 600, 1000], "depth": [2, 3, 4, 6, 8], "learning_rate": [0.03, 0.1],
                   "l2_leaf_reg": [1, 3, 10, 30, 100]},
        ),
        "svm": dict(
            # probability=True: το SVM δίνει αποστάσεις και όχι πιθανότητες. Έτσι τις μετατρέπει σε
            # πιθανότητες (Platt scaling), ώστε να συγκρίνεται με τα άλλα μοντέλα στα ίδια thresholds.
            estimator=SVC(kernel="rbf", probability=True, random_state=SEED, **cw("svm")),
            search="grid", space={"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.1]},
        ),
        "knn": dict(
            estimator=KNeighborsClassifier(),
            # Σε πρώτη δοκιμή το καλύτερο k ήταν 51, στην άκρη του πλέγματος· γι' αυτό επεκτάθηκε.
            search="grid", space={"n_neighbors": [5, 15, 31, 51, 75, 101, 151, 201], "weights": ["uniform", "distance"]},
        ),
        "nb": dict(
            estimator=GaussianNB(),
            # Σε πρώτη δοκιμή το var_smoothing βγήκε στη μέγιστη τιμή (1e-5)· επεκτάθηκε.
            search="grid", space={"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1]},
        ),
    }


def fit_benchmark(name: str, spec: dict, method: str, X_train, y_train):
    """Αναζήτηση υπερπαραμέτρων για ένα μοντέλο, με το ίδιο CV και κριτήριο για όλα."""
    pipeline = make_pipeline(spec["estimator"], method, preprocess=spec.get("preprocess", True))
    space = {f"model__{k}": v for k, v in spec["space"].items()}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    if spec["search"] == "random":
        search = RandomizedSearchCV(pipeline, space, n_iter=spec["n_iter"], scoring="roc_auc", cv=cv,
                                    random_state=SEED, n_jobs=-1)
    else:
        search = GridSearchCV(pipeline, space, scoring="roc_auc", cv=cv, n_jobs=-1)
    fit_params = {f"model__{k}": v for k, v in spec.get("fit_params", {}).items()}
    return search.fit(X_train, y_train, **fit_params)


def run(models: list[str] | None = None) -> None:
    """Εκπαιδεύει (όλα ή κάποια από) τα μοντέλα σύγκρισης και αποθηκεύει τα αποτελέσματά τους."""
    X_train, y_train, X_val, y_val, *_ = load_splits()
    method = load_choice()
    specs = model_specs(y_train, method)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name in models or list(specs):
        search = fit_benchmark(name, specs[name], method, X_train, y_train)
        pipeline = search.best_estimator_
        prob_train = pipeline.predict_proba(X_train)[:, 1]
        prob_val = pipeline.predict_proba(X_val)[:, 1]
        threshold = tune_threshold(y_val, prob_val)

        save_predictions(name, "train", X_train.index, y_train, prob_train)
        save_predictions(name, "val", X_val.index, y_val, prob_val)
        val_metrics = {"at_0.5": compute_metrics(y_val, prob_val, 0.5),
                       "at_tuned": compute_metrics(y_val, prob_val, threshold)}
        params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        update_grid_edge_report(name, grid_edge_rows(name, specs[name]["space"], params, threshold))
        update_search_flatness(search_flatness_row(name, search))
        save_model_metrics(name, threshold, val_metrics, params,
                           extra={"cv_roc_auc": float(search.best_score_), "imbalance_method": method})
        joblib.dump(pipeline, MODELS_DIR / f"{name}.joblib")

        m = val_metrics["at_tuned"]
        print(f"{name}: CV AUC {search.best_score_:.4f} | val AUC {m['roc_auc']:.4f}, F1 {m['f1']:.4f} "
              f"(threshold {threshold:.2f}) | {params}")
