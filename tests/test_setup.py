"""Έλεγχοι της βασικής δομής: ο σκελετός του project στέκεται και τα ελληνικά αποδίδονται στα σχήματα."""

import subprocess
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager

from src import config, plotting


def test_main_help_lists_all_stages():
    """Το `python main.py --help` πρέπει να αναφέρει όλα τα στάδια του pipeline."""
    out = subprocess.run(
        [sys.executable, "main.py", "--help"], cwd=config.ROOT, capture_output=True, text=True, check=True
    ).stdout
    from main import STAGES

    for stage in [*STAGES, "all"]:
        assert stage in out


def test_greek_font_has_greek_glyphs(tmp_path):
    """Η γραμματοσειρά των σχημάτων πρέπει να υπάρχει και να έχει ελληνικούς χαρακτήρες."""
    path = font_manager.findfont("DejaVu Sans", fallback_to_default=False)
    assert "DejaVuSans" in path
    fig, ax = plt.subplots()
    ax.set_title("Δοκιμή ελληνικών")
    saved = plotting.save_figure(fig, "greek_test", directory=tmp_path)
    assert saved.exists() and saved.stat().st_size > 0


def test_feature_lists_are_consistent():
    """Οι λίστες στηλών καλύπτουν ακριβώς τα 19 χαρακτηριστικά, χωρίς επικαλύψεις."""
    assert len(config.FEATURES) == 19
    assert len(set(config.FEATURES)) == 19
    assert config.TARGET not in config.FEATURES and config.ID_COLUMN not in config.FEATURES
