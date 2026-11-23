"""Όλα τα ελληνικά κείμενα των σχημάτων και των πινάκων, σε ένα σημείο.

Κανόνας: τίτλοι, άξονες, υπομνήματα και επικεφαλίδες πινάκων στα ελληνικά. Τα ονόματα των μεταβλητών
και οι τιμές τους μένουν στα αγγλικά, όπως στο dataset. Η αντιστοιχία δίνεται στο
`FEATURE_DESCRIPTIONS`, που τροφοδοτεί τον πίνακα περιγραφής του dataset.
"""

# Ετικέτες των δύο κλάσεων του στόχου.
CLASS_LABELS = {0: "Παρέμεινε", 1: "Αποχώρησε"}

# Περιγραφή κάθε στήλης του dataset στα ελληνικά: η αντιστοιχία αγγλικό όνομα → ελληνική περιγραφή
# που εμφανίζεται στον Πίνακα 1 της εργασίας.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "customerID": "Αναγνωριστικό πελάτη",
    "gender": "Φύλο πελάτη",
    "SeniorCitizen": "Ηλικιωμένος πελάτης (1 = ναι, 0 = όχι)",
    "Partner": "Έχει σύντροφο",
    "Dependents": "Έχει εξαρτώμενα μέλη",
    "tenure": "Διάρκεια συνδρομής (μήνες)",
    "PhoneService": "Τηλεφωνική υπηρεσία",
    "MultipleLines": "Πολλαπλές τηλεφωνικές γραμμές",
    "InternetService": "Τύπος σύνδεσης στο διαδίκτυο",
    "OnlineSecurity": "Υπηρεσία διαδικτυακής ασφάλειας",
    "OnlineBackup": "Υπηρεσία διαδικτυακών αντιγράφων ασφαλείας",
    "DeviceProtection": "Προστασία συσκευών",
    "TechSupport": "Τεχνική υποστήριξη",
    "StreamingTV": "Τηλεόραση μέσω διαδικτύου",
    "StreamingMovies": "Ταινίες μέσω διαδικτύου",
    "Contract": "Τύπος συμβολαίου",
    "PaperlessBilling": "Ηλεκτρονική τιμολόγηση (χωρίς χαρτί)",
    "PaymentMethod": "Τρόπος πληρωμής",
    "MonthlyCharges": "Μηνιαία χρέωση",
    "TotalCharges": "Συνολικές χρεώσεις",
    "Churn": "Αποχώρησε τον τελευταίο μήνα (στόχος)",
}

# Ομάδα κάθε στήλης, για την οργάνωση του Πίνακα 1.
FEATURE_GROUPS: dict[str, str] = {
    "customerID": "Αναγνωριστικό",
    "gender": "Δημογραφικά",
    "SeniorCitizen": "Δημογραφικά",
    "Partner": "Δημογραφικά",
    "Dependents": "Δημογραφικά",
    "tenure": "Λογαριασμός",
    "Contract": "Λογαριασμός",
    "PaperlessBilling": "Λογαριασμός",
    "PaymentMethod": "Λογαριασμός",
    "PhoneService": "Υπηρεσίες",
    "MultipleLines": "Υπηρεσίες",
    "InternetService": "Υπηρεσίες",
    "OnlineSecurity": "Υπηρεσίες",
    "OnlineBackup": "Υπηρεσίες",
    "DeviceProtection": "Υπηρεσίες",
    "TechSupport": "Υπηρεσίες",
    "StreamingTV": "Υπηρεσίες",
    "StreamingMovies": "Υπηρεσίες",
    "MonthlyCharges": "Χρέωση",
    "TotalCharges": "Χρέωση",
    "Churn": "Στόχος",
}

# Σύντομα ελληνικά ονόματα των χαρακτηριστικών, για τα σχήματα και τους πίνακες της εργασίας. Ίδια με τον
# κατάλογο όρων της εργασίας (report/glossary.md). Τα αγγλικά ονόματα μένουν μόνο ως αναγνωριστικά στον κώδικα.
FEATURE_NAMES: dict[str, str] = {
    "customerID": "Αναγνωριστικό πελάτη",
    "gender": "Φύλο",
    "SeniorCitizen": "Ηλικιωμένος πελάτης",
    "Partner": "Σύντροφος",
    "Dependents": "Εξαρτώμενα μέλη",
    "tenure": "Διάρκεια συνδρομής",
    "PhoneService": "Τηλεφωνία",
    "MultipleLines": "Πολλαπλές γραμμές",
    "InternetService": "Σύνδεση στο διαδίκτυο",
    "OnlineSecurity": "Διαδικτυακή ασφάλεια",
    "OnlineBackup": "Διαδικτυακά αντίγραφα ασφαλείας",
    "DeviceProtection": "Προστασία συσκευών",
    "TechSupport": "Τεχνική υποστήριξη",
    "StreamingTV": "Τηλεόραση μέσω διαδικτύου",
    "StreamingMovies": "Ταινίες μέσω διαδικτύου",
    "Contract": "Συμβόλαιο",
    "PaperlessBilling": "Ηλεκτρονική τιμολόγηση",
    "PaymentMethod": "Τρόπος πληρωμής",
    "MonthlyCharges": "Μηνιαία χρέωση",
    "TotalCharges": "Συνολικές χρεώσεις",
    "Churn": "Αποχώρηση",
}

# Ελληνικές τιμές των κατηγορικών χαρακτηριστικών (με μικρά· κεφαλαίο μπαίνει όπου χρειάζεται).
VALUE_NAMES: dict[str, str] = {
    "Yes": "ναι",
    "No": "όχι",
    "Male": "άνδρας",
    "Female": "γυναίκα",
    "Month-to-month": "μηνιαίο",
    "One year": "ετήσιο",
    "Two year": "διετές",
    "DSL": "DSL",
    "Fiber optic": "οπτική ίνα",
    "No internet service": "χωρίς σύνδεση στο διαδίκτυο",
    "No phone service": "χωρίς τηλεφωνία",
    "Electronic check": "ηλεκτρονική επιταγή",
    "Mailed check": "ταχυδρομική επιταγή",
    "Bank transfer (automatic)": "τραπεζική μεταφορά (αυτόματη)",
    "Credit card (automatic)": "πιστωτική κάρτα (αυτόματη)",
}


def feature_name(feature: str) -> str:
    """Το ελληνικό όνομα ενός χαρακτηριστικού (ή το ίδιο το όνομα, αν δεν υπάρχει στον κατάλογο)."""
    return FEATURE_NAMES.get(feature, feature)


def value_name(feature: str, value) -> str:
    """Η ελληνική μορφή μιας τιμής. Για το `SeniorCitizen` (0/1) δίνει όχι/ναι· οι αριθμοί μένουν ως έχουν."""
    if feature == "SeniorCitizen" and str(value) in ("0", "1"):
        return "ναι" if str(value) == "1" else "όχι"
    return VALUE_NAMES.get(str(value), str(value))


def capitalize(text: str) -> str:
    """Κεφαλαίο μόνο το πρώτο γράμμα (για ετικέτες σχημάτων)."""
    return text[:1].upper() + text[1:] if text else text


def encoded_name(column: str) -> str:
    """Ελληνικό όνομα μιας στήλης μετά την προεπεξεργασία, π.χ. «Contract_Month-to-month» → «Συμβόλαιο: μηνιαίο»."""
    if column in FEATURE_NAMES:
        return FEATURE_NAMES[column]
    matches = [f for f in FEATURE_NAMES if column.startswith(f + "_")]
    if not matches:
        return column
    feature = max(matches, key=len)
    return f"{FEATURE_NAMES[feature]}: {value_name(feature, column[len(feature) + 1:])}"


# Ελληνικά ονόματα για τους τύπους μεταβλητών.
FEATURE_TYPES = {"numeric": "Αριθμητική", "binary": "Δυαδική", "categorical": "Κατηγορική", "id": "Αναγνωριστικό"}

# Ακριβέστερος τύπος, όπως στον Πίνακα 1 της εργασίας (διακριτή/συνεχής, ονομαστική/διατακτική).
FEATURE_SUBTYPES = {"tenure": "Αριθμητική διακριτή", "MonthlyCharges": "Αριθμητική συνεχής",
                    "TotalCharges": "Αριθμητική συνεχής", "Contract": "Κατηγορική διατακτική"}

# Κείμενα σχημάτων ανά σχήμα: {"όνομα_σχήματος": {"title": ..., "xlabel": ..., ...}}.
FIGURES: dict[str, dict[str, str]] = {}

# Επικεφαλίδες στηλών των πινάκων: {"όνομα_στήλης_στον_κώδικα": "Ελληνική επικεφαλίδα"}.
TABLE_HEADERS: dict[str, str] = {}

# Ονόματα μοντέλων όπως εμφανίζονται σε σχήματα και πίνακες.
MODEL_DISPLAY_NAMES = {
    "logreg": "Logistic Regression",
    "mlp_simple": "Απλό MLP",
    "mlp_tuned": "Τελικό MLP",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "catboost": "CatBoost",
    "svm": "SVM",
    "knn": "k-NN",
    "nb": "Naive Bayes",
}
