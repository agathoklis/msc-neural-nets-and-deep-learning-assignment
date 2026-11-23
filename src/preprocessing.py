"""Προεπεξεργασία των χαρακτηριστικών.

Ένας κοινός preprocessor για όλα τα μοντέλα (εκτός από το CatBoost, που κρατά τις κατηγορίες ως έχουν).
Αποτελείται από δύο βήματα:
1. `variant`: η παραλλαγή κωδικοποίησης από το σχετικό πείραμα (π.χ. συγχώνευση των περιττών «No internet service»).
2. `encode`: κανονικοποίηση των αριθμητικών και one-hot των κατηγορικών.

Μπαίνει πάντα μέσα σε Pipeline, ώστε να «μαθαίνει» (fit) μόνο από τα δεδομένα εκπαίδευσης. Π.χ. ο μέσος
όρος και η τυπική απόκλιση της κανονικοποίησης υπολογίζονται μόνο στο train, για να μη διαρρέει
πληροφορία από το test.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import BINARY_PASSTHROUGH, CATEGORICAL, FEATURES, NUMERIC
from src.encoding import apply_encoding_variant, load_choice

# Πλήθος στηλών μετά την προεπεξεργασία ανά παραλλαγή (ελέγχεται στα tests).
ENCODED_FEATURE_COUNTS = {"full": 40, "merged": 26, "merged_no_total": 25}


def variant_step(variant: str) -> FunctionTransformer:
    """Το βήμα της παραλλαγής κωδικοποίησης, ως μετασχηματιστής που μπαίνει σε Pipeline."""
    return FunctionTransformer(apply_encoding_variant, kw_args={"variant": variant})


def build_preprocessor(variant: str | None = None) -> Pipeline:
    """Επιστρέφει έναν (μη εκπαιδευμένο) preprocessor για την παραλλαγή `variant`.

    Χωρίς `variant` χρησιμοποιείται αυτή που επέλεξε το πείραμα κωδικοποίησης (`encoding_choice.json`).
    - Αριθμητικές στήλες → StandardScaler: έτσι είναι στην ίδια κλίμακα. Το χρειάζονται τα μοντέλα
      που βασίζονται σε αποστάσεις ή κλίσεις (νευρωνικό, Logistic Regression, SVM, k-NN).
    - `SeniorCitizen` → αυτούσια, αφού είναι ήδη 0/1.
    - Στήλες κειμένου → one-hot. Με `drop="if_binary"` οι δυαδικές γίνονται μία στήλη 0/1 αντί για δύο.
      Το `handle_unknown="ignore"` προστατεύει από κατηγορίες που ίσως δεν εμφανίζονται στο train.
    """
    variant = variant or load_choice()
    numeric = [c for c in NUMERIC if not (variant == "merged_no_total" and c == "TotalCharges")]
    encode = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("bin", "passthrough", BINARY_PASSTHROUGH),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        verbose_feature_names_out=False,  # ονόματα όπως «Contract_Two year» αντί για «cat__Contract_Two year»
    )
    return Pipeline([("variant", variant_step(variant)), ("encode", encode)])


def feature_names(fitted) -> list[str]:
    """Τα ονόματα των στηλών εξόδου ενός εκπαιδευμένου preprocessor."""
    encoder = fitted.named_steps["encode"] if isinstance(fitted, Pipeline) else fitted
    return list(encoder.get_feature_names_out())


def original_feature(name: str) -> str:
    """Επιστρέφει την αρχική στήλη από την οποία προέρχεται μια στήλη εξόδου.

    Π.χ. «Contract_Two year» → «Contract». Κρατάμε το μακρύτερο ταίριασμα, ώστε ένα όνομα να μην
    αντιστοιχιστεί κατά λάθος σε στήλη με κοινό πρόθεμα.
    """
    if name in FEATURES:
        return name
    matches = [f for f in FEATURES if name.startswith(f + "_")]
    if not matches:
        raise KeyError(f"Άγνωστη στήλη: {name}")
    return max(matches, key=len)
