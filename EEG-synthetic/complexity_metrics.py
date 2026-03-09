import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from pycol_complexity import complexity
# Assumiamo che la libreria 'complexity' sia già importata nel tuo ambiente
# import complexity 

def calculate_complexity_metrics(X, y, method='pca', channel_idx=None, n_components=50, subset_ratio=0.10, random_state=42, imb=True):
    """
    Calcola le metriche di complessità del dataset usando due strategie diverse.
    
    Parametri:
    - X: array (n_epochs, n_channels, n_times)
    - y: array (n_epochs,)
    - method: str, 'pca' oppure 'spatial'
    - channel_idx: int o None. Se method='spatial':
                   None -> Fa la media di TUTTI i canali.
                   int  -> Seleziona SOLO quel canale specifico (es. indice di Pz).
    - n_components: int, numero di componenti per la PCA.
    - subset_ratio: float, frazione del dataset da usare (es. 0.10 = 10%).
    
    Ritorna:
    - comp_obj: L'oggetto complexity inizializzato e calcolato.
    """
    
    print(f"--- Calcolo Complessità | Metodo: {method.upper()} ---")
    
    # 1. PRE-PROCESSING
    if method == 'spatial':
        if channel_idx is None:
            # Media su tutti i canali (axis 1) -> Shape: (Epochs, Times)
            print("Strategia: Media Spaziale (Tutti i canali)")
            X_processed = X.mean(axis=1)
        else:
            # Selezione canale specifico -> Shape: (Epochs, Times)
            print(f"Strategia: Canale Singolo (Indice: {channel_idx})")
            X_processed = X[:, channel_idx, :]
            
    elif method == 'pca':
        # Flattening: (Epochs, Channels*Times)
        print(f"Strategia: PCA (Comp: {n_components})")
        X_flat = X.reshape(X.shape[0], -1)
        
        pca = PCA(n_components=n_components, random_state=random_state)
        X_processed = pca.fit_transform(X_flat)
        
    else:
        raise ValueError("Metodo non valido. Scegliere 'pca' o 'spatial'.")

    # 2. SUBSAMPLING (STRATIFICATO)
    # Usiamo train_test_split per estrarre una piccola percentuale (subset_ratio) mantenendo le proporzioni delle classi
    # Nota: test_size sarà la parte che teniamo (il subset piccolo)
    _, X_sub, _, y_sub = train_test_split(
        X_processed, 
        y, 
        test_size=subset_ratio, 
        stratify=y, 
        random_state=random_state
    )
    
    print(f"Dataset ridotto per complessità: {X_sub.shape}")

    # 3. CALCOLO METRICHE (Libreria Complexity)
    dic_dataset = {
        'X': X_sub,
        'y': y_sub.astype(int)
    }

    # Inizializzazione
    comp_obj = complexity.Complexity(dataset=dic_dataset, file_type="array")

    # Calcolo Metriche
    print("Calcolo metriche in corso...")
    
    # Instance-Level
    comp_obj.kDN(imb=imb)  # k-Discordant Neighbors
    comp_obj.N3(imb=imb)   # Error Rate of NN classifier
    comp_obj.N4(imb=imb)   # Non-Linearity of NN classifier

    # Structural
    comp_obj.N1(imb=imb)   # Fraction of borderline points
    comp_obj.N2(imb=imb)   # Ratio of intra/extra class NN distance

    # Multi-Resolution
    comp_obj.C1(imb=imb)   # Entropy of class proportions
    comp_obj.C2(imb=imb)   # Imbalance ratio (KL divergence)
    
    print("Calcolo completato.")
    
    return comp_obj

# ==========================================
# ESEMPIO DI UTILIZZO
# ==========================================