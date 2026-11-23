"""Κοινό στυλ για όλα τα σχήματα: γραμματοσειρά με ελληνικούς χαρακτήρες, 300 dpi, ίδια χρώματα.

Κάθε σχήμα της εργασίας περνά από το `save_figure`, ώστε όλα να έχουν την ίδια ανάλυση και εμφάνιση.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # χωρίς παράθυρο: τα σχήματα μόνο αποθηκεύονται (τρέχει και χωρίς οθόνη)
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR

DPI = 300

# Δύο χρώματα για τις δύο κλάσεις, ίδια σε όλα τα σχήματα: ο αναγνώστης μαθαίνει μία φορά
# ότι το πορτοκαλί σημαίνει «αποχώρησε».
COLOR_STAY = "#4C72B0"
COLOR_CHURN = "#DD8452"
CLASS_COLORS = {0: COLOR_STAY, 1: COLOR_CHURN}


def setup_style() -> None:
    """Ρυθμίζει το matplotlib με γραμματοσειρά που αποδίδει ελληνικά.

    Η DejaVu Sans έρχεται μαζί με το matplotlib και έχει ελληνικούς χαρακτήρες, άρα δεν
    εξαρτόμαστε από γραμματοσειρές του συστήματος.
    """
    # Ελληνική μορφή αριθμών στους άξονες (υποδιαστολή «,»), ίδια με το κείμενο της εργασίας.
    # Αν το σύστημα δεν έχει ελληνικό locale, τα σχήματα βγαίνουν κανονικά με τελεία.
    import locale

    try:
        locale.setlocale(locale.LC_NUMERIC, "el_GR.UTF-8")
        use_locale = True
    except locale.Error:
        use_locale = False
    plt.rcParams.update(
        {
            "axes.formatter.use_locale": use_locale,
            "font.family": "DejaVu Sans",
            "figure.dpi": 100,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def save_figure(fig, name: str, directory: Path = FIGURES_DIR) -> Path:
    """Αποθηκεύει ένα σχήμα ως PNG στον φάκελο των σχημάτων και κλείνει το figure.

    Το κλείσιμο έχει σημασία όταν παράγονται δεκάδες σχήματα στη σειρά, για να μη γεμίζει η μνήμη.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


setup_style()
