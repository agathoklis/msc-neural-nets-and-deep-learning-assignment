"""Αρχιτεκτονικές του νευρωνικού δικτύου (ερώτημα 3).

Δύο μοντέλα:
- ένα απλό MLP χωρίς καμία τεχνική κανονικοποίησης, ως σημείο αναφοράς·
- ένα «προχωρημένο» MLP με Batch Normalization, Dropout, ποινές L1/L2 και residual συνδέσεις.
  Κάθε τεχνική ενεργοποιείται ή απενεργοποιείται από τις παραμέτρους, ώστε το tuning να διαλέξει
  και το ablation να μετρήσει τι προσφέρει η καθεμία.

Το `build_advanced_mlp` δέχεται ένα απλό λεξικό παραμέτρων. Έτσι το καλούν με τον ίδιο τρόπο και το
KerasTuner και το Optuna, και η σύγκριση των μεθόδων tuning γίνεται πάνω στο ίδιο ακριβώς μοντέλο.
"""

import keras
from keras import layers, regularizers


def _metrics() -> list:
    """Οι μετρικές που παρακολουθούνται στην εκπαίδευση. Το «auc» είναι το κριτήριο του early stopping."""
    return [
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ]


def build_simple_mlp(n_features: int, lr: float = 1e-3) -> keras.Model:
    """Απλό MLP: δύο κρυφά επίπεδα (32 και 16 νευρώνες) με ReLU και έξοδο sigmoid.

    Δεν έχει dropout, ποινές ή batch normalization. Δείχνει τι πετυχαίνει ένα δίκτυο χωρίς τεχνικές, ώστε να
    φανεί αν οι προχωρημένες τεχνικές προσθέτουν κάτι.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="mlp_simple",
    )
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=_metrics())
    return model


def _dense(units: int, l1: float, l2: float) -> layers.Dense:
    """Πυκνό επίπεδο με ποινή L1/L2 στα βάρη.

    Η ποινή «τιμωρεί» τα μεγάλα βάρη: η L1 σπρώχνει άχρηστα βάρη στο μηδέν, η L2 κρατά όλα τα βάρη
    μικρά. Και οι δύο περιορίζουν την υπερπροσαρμογή (overfitting).
    """
    return layers.Dense(units, kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2))


def residual_block(x, units: int, dropout: float, l1: float, l2: float, use_bn: bool):
    """Residual block: δύο πυκνά επίπεδα και μια «παράκαμψη» (skip connection) που προσθέτει την είσοδο στην έξοδο.

    Η παράκαμψη επιτρέπει στο block να μάθει μόνο μια *διόρθωση* πάνω στην είσοδό του. Αν δεν χρειάζεται
    διόρθωση, αρκεί να μηδενίσει τα βάρη του (He et al., 2016). Αν το πλάτος της εισόδου διαφέρει από
    το `units`, η είσοδος περνά πρώτα από ένα γραμμικό επίπεδο (projection), ώστε να μπορεί να προστεθεί.
    """
    shortcut = x
    if x.shape[-1] != units:
        shortcut = layers.Dense(units, use_bias=False)(x)

    y = _dense(units, l1, l2)(x)
    if use_bn:
        y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Dropout(dropout)(y)
    y = _dense(units, l1, l2)(y)
    if use_bn:
        y = layers.BatchNormalization()(y)

    out = layers.Add()([shortcut, y])
    return layers.Activation("relu")(out)


def _plain_block(x, units: int, dropout: float, l1: float, l2: float, use_bn: bool):
    """Απλό block χωρίς παράκαμψη: Dense → (BN) → ReLU → Dropout. Χρησιμοποιείται όταν `use_residual=False`."""
    x = _dense(units, l1, l2)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return layers.Dropout(dropout)(x)


def build_advanced_mlp(params: dict, n_features: int) -> keras.Model:
    """Το τελικό MLP, που κατασκευάζεται από ένα λεξικό υπερπαραμέτρων.

    Δομή: είσοδος → αρχικό block (Dense → BN → ReLU → Dropout) → `n_blocks` blocks (residual ή απλά)
    → έξοδος sigmoid.
    - Batch Normalization: κανονικοποιεί τις ενεργοποιήσεις κάθε mini-batch, ώστε η εκπαίδευση να είναι
      πιο σταθερή και γρήγορη (Ioffe & Szegedy, 2015).
    - Dropout: σε κάθε βήμα απενεργοποιεί τυχαία ένα ποσοστό νευρώνων, ώστε το δίκτυο να μη στηρίζεται σε
      λίγους συγκεκριμένους (Srivastava et al., 2014).

    Το `batch_size` του λεξικού δεν αφορά την αρχιτεκτονική· χρησιμοποιείται στο `fit`.
    """
    units, dropout = params["units"], params["dropout"]
    l1, l2, use_bn = params["l1"], params["l2"], params["use_bn"]

    inputs = keras.Input(shape=(n_features,))
    x = _plain_block(inputs, units, dropout, l1, l2, use_bn)
    for _ in range(params["n_blocks"]):
        if params["use_residual"]:
            x = residual_block(x, units, dropout, l1, l2, use_bn)
        else:
            x = _plain_block(x, units, dropout, l1, l2, use_bn)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="mlp_advanced")
    model.compile(optimizer=keras.optimizers.Adam(params["lr"]), loss="binary_crossentropy", metrics=_metrics())
    return model
