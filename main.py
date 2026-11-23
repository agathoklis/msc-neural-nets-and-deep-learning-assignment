"""Σημείο εκκίνησης του pipeline: `python main.py <stage>`.

Κάθε στάδιο αντιστοιχεί σε ένα βήμα της ανάλυσης και καλεί μία συνάρτηση `run()` στο αντίστοιχο
module του `src/`. Το `all` τρέχει όλα τα στάδια με τη σωστή σειρά, ώστε να αναπαράγονται όλα τα
αποτελέσματα από την αρχή.
"""

import argparse
import importlib
import time

# Στάδιο → (module, συνάρτηση, περιγραφή). Η σειρά εδώ είναι και η σειρά του `all`.
STAGES = {
    "eda": ("src.eda", "run", "Περιγραφική ανάλυση (EDA)"),
    "encoding": ("src.encoding", "run", "Πείραμα κωδικοποίησης"),
    "imbalance": ("src.imbalance", "run", "Ανάλυση ανισορροπίας κλάσεων"),
    "baseline": ("src.baseline", "run", "Logistic Regression"),
    "nn-simple": ("src.nn_train", "run_simple", "Απλό MLP"),
    "nn-tune": ("src.nn_tuning", "run", "Σύγκριση μεθόδων tuning"),
    "nn-final": ("src.nn_train", "run_final", "Τελικό νευρωνικό"),
    "nn-ablation": ("src.nn_train", "run_ablation", "Ablation"),
    "benchmarks": ("src.benchmarks", "run", "Μοντέλα σύγκρισης"),
    "final": ("src.analysis", "run_final", "Τελική αξιολόγηση στο test set"),
    "explain": ("src.explain", "run", "SHAP και LIME"),
    "business": ("src.analysis", "run_business", "Ανάλυση για την επιχείρηση"),
}


def run_stage(name: str) -> None:
    """Φορτώνει το module ενός σταδίου και καλεί τη συνάρτησή του.

    Η φόρτωση γίνεται μόνο όταν χρειάζεται, ώστε π.χ. το `eda` να μη φορτώνει το TensorFlow.
    """
    module_name, func_name, description = STAGES[name]
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
    except (ModuleNotFoundError, AttributeError) as exc:
        raise NotImplementedError(f"Το στάδιο '{name}' δεν έχει υλοποιηθεί ακόμα ({exc}).") from exc
    print(f"==> {name}: {description}")
    start = time.time()
    func()
    print(f"<== {name}: ολοκληρώθηκε σε {time.time() - start:.1f} δευτερόλεπτα")


def main() -> None:
    """Διαβάζει το όνομα του σταδίου από τη γραμμή εντολών και το εκτελεί."""
    stage_help = "\n".join(f"  {name:<12} {desc}" for name, (_, _, desc) in STAGES.items())
    parser = argparse.ArgumentParser(
        description="Pipeline της εργασίας Telco Customer Churn.",
        epilog=f"Στάδια:\n{stage_help}\n  {'all':<12} Όλα τα στάδια με τη σειρά",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("stage", choices=[*STAGES, "all"], help="το στάδιο που θα εκτελεστεί")
    args = parser.parse_args()

    stages = list(STAGES) if args.stage == "all" else [args.stage]
    for name in stages:
        run_stage(name)


if __name__ == "__main__":
    main()
