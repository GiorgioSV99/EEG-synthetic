from imblearn.over_sampling import SMOTE
import numpy as np

def apply_smote_3d(X, y):
    """
    Applica SMOTE a dati 3D (epoche, canali, campioni).
    Restituisce i dati bilanciati in formato 3D.
    """
    n_epochs, n_channels, n_times = X.shape

    # 1. Flattening: trasformiamo in (n_epoche, features)
    X_flat = X.reshape(n_epochs, -1)

    # 2. Applicazione SMOTE
    smote = SMOTE(random_state=42)
    X_res_flat, y_res = smote.fit_resample(X_flat, y)

    # 3. Reshape: riportiamo in 3D (nuove_epoche, canali, campioni)
    X_res_3d = X_res_flat.reshape(-1, n_channels, n_times)

    print(f"Bilanciamento completato: {n_epochs} -> {X_res_3d.shape[0]} epoche")
    return X_res_3d, y_res