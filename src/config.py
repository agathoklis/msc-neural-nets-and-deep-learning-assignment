"""Κεντρικές ρυθμίσεις του project: διαδρομές, seed, λίστες στηλών και χώρος αναζήτησης του νευρωνικού.

Όλα τα υπόλοιπα modules διαβάζουν από εδώ, ώστε μια απόφαση (π.χ. ποιες στήλες είναι αριθμητικές)
να γράφεται μία φορά και να ισχύει παντού.
"""

from pathlib import Path

# --- Διαδρομές -------------------------------------------------------------------------------
# Η ρίζα υπολογίζεται από τη θέση του αρχείου, ώστε ο κώδικας να τρέχει από οποιονδήποτε φάκελο
# (π.χ. και από το notebook, που τρέχει μέσα στο notebooks/).
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUTS = ROOT / "outputs"
SPLITS_PATH = OUTPUTS / "splits.json"
PREDICTIONS_DIR = OUTPUTS / "predictions"
METRICS_DIR = OUTPUTS / "metrics"
TABLES_DIR = OUTPUTS / "tables"
FIGURES_DIR = OUTPUTS / "figures"
MODELS_DIR = OUTPUTS / "models"

# --- Αναπαραγωγιμότητα ------------------------------------------------------------------------
SEED = 42

# --- Στήλες -----------------------------------------------------------------------------------
ID_COLUMN = "customerID"  # αναγνωριστικό, δεν έχει προγνωστική αξία, αφαιρείται
TARGET = "Churn"

# Συνεχείς μεταβλητές: κανονικοποιούνται με StandardScaler.
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]

# Ήδη κωδικοποιημένη ως 0/1 στο dataset, οπότε περνά αυτούσια.
BINARY_PASSTHROUGH = ["SeniorCitizen"]

# Όλες οι στήλες κειμένου (δυαδικές και πολυκατηγορικές). Κωδικοποιούνται με one-hot,
# εκτός από το CatBoost, που τις παίρνει ως έχουν.
CATEGORICAL = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURES = NUMERIC + BINARY_PASSTHROUGH + CATEGORICAL  # 19 αρχικά χαρακτηριστικά
N_ENCODED_FEATURES = 40  # μετά την προεπεξεργασία· ελέγχεται στα tests

# --- Διαχωρισμός ------------------------------------------------------------------------------
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # ως ποσοστό του συνόλου· μετατρέπεται σε 15/85 του υπολοίπου στο data.py

# --- Threshold --------------------------------------------------------------------------------
THRESHOLD_GRID = (0.01, 0.99, 0.01)  # (από, έως, βήμα) για την αναζήτηση του καλύτερου F1 (ευρύ, ώστε το βέλτιστο να μην πέφτει στην άκρη)

# --- Ονόματα μοντέλων (χρησιμοποιούνται και στα ονόματα αρχείων) ------------------------------
MODEL_NAMES = [
    "logreg", "mlp_simple", "mlp_tuned", "rf", "xgb", "lgbm", "catboost", "svm", "knn", "nb",
]

# --- Χώρος αναζήτησης του νευρωνικού ----------------------------------------------------------
# Μοναδική πηγή για τις 4 μεθόδους tuning (KerasTuner Random/Bayesian/Hyperband και Optuna),
# ώστε η σύγκριση να είναι δίκαιη: όλες ψάχνουν ακριβώς στις ίδιες τιμές.
NN_SEARCH_SPACE = {
    "units": {"type": "choice", "values": [32, 64, 128]},
    "n_blocks": {"type": "choice", "values": [1, 2, 3]},
    "dropout": {"type": "choice", "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
    "l1": {"type": "choice", "values": [0.0, 1e-5, 1e-4]},
    "l2": {"type": "choice", "values": [0.0, 1e-5, 1e-4, 1e-3]},
    "use_residual": {"type": "choice", "values": [True, False]},
    "use_bn": {"type": "choice", "values": [True, False]},
    "lr": {"type": "float_log", "min": 1e-4, "max": 1e-2},
    "batch_size": {"type": "choice", "values": [32, 64, 128]},
}

# Προεπιλεγμένες τιμές: χρησιμοποιούνται στα tests και ως «λογική» ρύθμιση πριν από το tuning.
NN_DEFAULT_PARAMS = {
    "units": 64,
    "n_blocks": 2,
    "dropout": 0.2,
    "l1": 0.0,
    "l2": 1e-4,
    "use_residual": True,
    "use_bn": True,
    "lr": 1e-3,
    "batch_size": 64,
}

# Ρυθμίσεις εκπαίδευσης του νευρωνικού, κοινές σε κάθε fit.
NN_MAX_EPOCHS = 200
NN_EARLY_STOPPING_PATIENCE = 15
NN_REDUCE_LR_PATIENCE = 5
NN_REDUCE_LR_FACTOR = 0.5


def set_seeds(seed: int = SEED) -> None:
    """Ορίζει το seed σε Python, NumPy και (αν είναι εγκατεστημένο) TensorFlow/Keras.

    Το TensorFlow εισάγεται μέσα στη συνάρτηση, ώστε τα modules που δεν το χρειάζονται
    (π.χ. το EDA) να μην πληρώνουν τον χρόνο φόρτωσής του.
    """
    import os
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras

        keras.utils.set_random_seed(seed)
    except ImportError:
        pass
