# Πρόβλεψη αποχώρησης πελατών (Telco Customer Churn)

Κώδικας της εργασίας για το μάθημα **AG704 Neural Nets & Deep Learning** (MSc Artificial Intelligence).

Ένα νευρωνικό δίκτυο (TensorFlow/Keras) συγκρίνεται με Logistic Regression και 7 ακόμα μοντέλα στο
Telco Customer Churn dataset (IBM, μέσω Kaggle), με επεξήγηση (SHAP, LIME) και ανάλυση για την επιχείρηση.

## Δομή

| Φάκελος / αρχείο | Περιεχόμενο |
|---|---|
| `notebooks/churn_analysis.ipynb` | Όλη η ανάλυση με τη σειρά της εκφώνησης, εκτελεσμένη (το καλύτερο σημείο εκκίνησης) |
| `src/` | Ο κώδικας: δεδομένα, προεπεξεργασία, μοντέλα, tuning, αξιολόγηση, επεξήγηση |
| `tests/` | Έλεγχοι (`pytest`) |
| `main.py` | Εκτέλεση του pipeline ανά στάδιο |
| `outputs/` | Τα αποτελέσματα που παρουσιάζει η εργασία (πίνακες, σχήματα, μετρικές, προβλέψεις) |
| `data/` | Το αρχικό dataset |

## Εγκατάσταση

Python 3.12 (το TensorFlow δεν υποστηρίζει ακόμα την 3.14).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Σε macOS, τα XGBoost και LightGBM χρειάζονται το OpenMP: `brew install libomp`.

## Εκτέλεση

```bash
.venv/bin/python -m pytest              # έλεγχοι
.venv/bin/python main.py --help         # τα στάδια του pipeline
.venv/bin/python main.py all            # όλα από την αρχή
```

Σε macOS, για μεγάλες εκτελέσεις: `caffeinate -is .venv/bin/python main.py all`, ώστε ο υπολογιστής να μην
πέσει σε ύπνο. Με σταθερό seed (42), δύο εκτελέσεις από την αρχή δίνουν πανομοιότυπα αποτελέσματα (εκτός από
τους χρόνους εκτέλεσης).

Τα εκπαιδευμένα μοντέλα (`outputs/models/`) δεν περιλαμβάνονται λόγω μεγέθους· ξαναδημιουργούνται με
`main.py all`. Το notebook διαβάζει τα αποθηκευμένα αποτελέσματα του `outputs/`.
