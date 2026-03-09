import mne
import numpy as np
import scipy.io
import os

import os
import numpy as np
import scipy.io
import mne
import matplotlib.pyplot as plt
import numpy as np

class BCIAUTLoader:
    def __init__(self, base_path, sfreq_orig=250, new_sfreq=128, tmin = -0.2, tmax=0.8, 
                 ch_names=None, montage_name='standard_1020'):
        """
        Inizializza il loader con i parametri di configurazione globali.
        """
        self.base_path = base_path
        self.sfreq_orig = sfreq_orig
        self.new_sfreq = new_sfreq
        self.tmax = tmax
        self.tmin = tmin
        
        if ch_names is None:
            self.ch_names = ['C3', 'Cz', 'C4', 'CPz', 'P3', 'Pz', 'P4', 'POz']
        else:
            self.ch_names = ch_names
            
        self.montage_name = montage_name

    def _format_ids(self, subject_int, session_int):
        """Helper per formattare gli ID (es. 3 -> 'SBJ03', 6 -> 'S06')"""
        sbj_str = f"SBJ{subject_int:02d}"
        sess_str = f"S{session_int:02d}"
        return sbj_str, sess_str

    def _load_single_epoch(self, subject_str, session_str, mode):
        """
        Carica una singola sessione (Logica originale incapsulata).
        Ritorna un oggetto mne.EpochsArray o None se il file non esiste.
        """
        folder_path = os.path.join(self.base_path, subject_str, session_str, mode)
        data_file = 'trainData.mat' if mode == 'Train' else 'testData.mat'
        target_file = 'trainTargets.txt' if mode == 'Train' else 'testTargets.txt'

        data_path = os.path.join(folder_path, data_file)
        target_path = os.path.join(folder_path, target_file)

        # Controllo esistenza file
        if not os.path.exists(data_path) or not os.path.exists(target_path):
            print(f"Attenzione: File non trovato: {data_path}")
            return None

        # Caricamento MAT
        try:
            mat = scipy.io.loadmat(data_path)
            key = data_file.split('.')[0]
            data_original = mat[key]
        except Exception as e:
            print(f"Errore caricamento .mat {data_path}: {e}")
            return None

        # Transpose (Epochs, Channels, Times)
        data_mne = np.transpose(data_original, (2, 0, 1))

        # Caricamento Targets
        targets = np.loadtxt(target_path).astype(int)

        # Creazione Info e Epochs
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq_orig, ch_types='eeg')
        epochs = mne.EpochsArray(data_mne, info, tmin=self.tmin, verbose=False)
        
        # Montage
        montage = mne.channels.make_standard_montage(self.montage_name)
        epochs.set_montage(montage)

        # Pre-processing
        epochs.filter(l_freq=0.1, h_freq=15.0, fir_design='firwin', verbose=False)
        epochs.resample(self.new_sfreq, verbose=False)
        epochs.crop(tmax=self.tmax, verbose=False)
        epochs.apply_baseline(baseline=(None, 0), verbose=False)

        # Assegnazione Eventi
        events = np.zeros((len(targets), 3), dtype=int)
        events[:, 0] = np.arange(len(targets))
        events[:, 2] = targets
        epochs.events = events
        epochs.event_id = {'Non-Target': 0, 'Target': 1}

        return epochs

    def get_data(self, subjects, sessions, modes=['Train', 'Test'], verbose=True):
            
            # Gestione shortcut 'all'
            if subjects == 'all':
                subjects = list(range(1, 16)) 
            if sessions == 'all':
                sessions = list(range(1, 8))  
                
            if isinstance(subjects, int): subjects = [subjects]
            if isinstance(sessions, int): sessions = [sessions]
            if isinstance(modes, str): modes = [modes]

            epochs_list = []
            
            # --- MODIFICA 1: Liste per salvare i metadati ---
            meta_subjects = []
            meta_sessions = []
            
            if verbose:
                print(f"Caricamento Soggetti: {subjects}, Sessioni: {sessions}, Modalità: {modes}")

            for subj in subjects:
                for sess in sessions:
                    sbj_str, sess_str = self._format_ids(subj, sess)
                    
                    for mode in modes:
                        ep = self._load_single_epoch(sbj_str, sess_str, mode)
                        
                        if ep is not None:
                            epochs_list.append(ep)
                            
                            # --- MODIFICA 2: Crea vettori di ID lunghi quanto il numero di trial caricati ---
                            n_trials_in_epoch = len(ep)
                            
                            # Crea un array pieno del numero del soggetto attuale (es. [3, 3, 3...])
                            meta_subjects.append(np.full(n_trials_in_epoch, subj))
                            
                            # Crea un array pieno del numero della sessione attuale (es. [1, 1, 1...])
                            meta_sessions.append(np.full(n_trials_in_epoch, sess))

            if not epochs_list:
                raise ValueError("Nessun dato trovato con i parametri specificati.")

            # Concatena tutte le epoche trovate
            all_epochs = mne.concatenate_epochs(epochs_list, verbose=False)
            
            # Estrai Numpy Arrays
            X = all_epochs.get_data()[:,:,:-1]
            y = all_epochs.events[:, 2]
            
            # --- MODIFICA 3: Concatena le liste dei metadati in un unico array NumPy ---
            subjects_arr = np.concatenate(meta_subjects)
            sessions_arr = np.concatenate(meta_sessions)
            
            if verbose:
                print(f"-> Totale caricato: {X.shape} - Classi: {np.unique(y, return_counts=True)}")
                
            # --- MODIFICA 4: Ritorna anche i vettori metadati ---
            return X, y, subjects_arr, sessions_arr

    def normalize_data_z_score(self, X_train, X_test):
        """
        Utilità per normalizzare (Z-score) basandosi sul Train set.
        """
        mu = X_train.mean(axis=(0), keepdims=True)
        std = X_train.std(axis=(0), keepdims=True)
        
        X_train_norm = (X_train - mu) / std
        # Normalizziamo il test set usando le statistiche del train (corretto per ML)
        X_test_norm = (X_test - mu) / std
        
        return X_train_norm, X_test_norm
    
    def normalize_data_minmax(self, X_train, X_test):
        # 1. Calcola il Min e il Max globali del training set
        x_min = np.min(X_train)
        x_max = np.max(X_train)

        X_train = (X_train - x_min) / (x_max - x_min)
        X_test = (X_test - x_min) / (x_max - x_min)
        return X_train, X_test



def plot_normalized_arrays(X_train, y_train, X_test, y_test, sfreq=128, tmin=-0.2):
    """
    Plotta la media globale (su tutti i canali) dei dati Normalizzati (Z-score).
    
    Parametri:
    - X_train, X_test: array (n_epochs, n_channels, n_times)
    - y_train, y_test: array (n_epochs,) con valori 0 (Non-Target) e 1 (Target)
    - sfreq: Frequenza di campionamento (default 64Hz come impostato nel loader)
    - tmin: Tempo di inizio epoca (default -0.2s)
    """
    
    # 1. Creiamo l'asse dei tempi
    n_samples = X_train.shape[2]
    # Calcola la durata totale in secondi
    duration = n_samples / sfreq
    # Crea il vettore tempo: da tmin fino a tmin+duration
    times = np.linspace(tmin, tmin + duration, n_samples)

    # 2. Setup Grafico
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Dizionario per iterare tra Train e Test
    datasets = [
        ("TRAIN Set (Normalizzato)", X_train, y_train, axes[0]),
        ("TEST Set (Normalizzato)", X_test, y_test, axes[1])
    ]

    print("--- Generazione Grafici NumPy ---")

    for title, X, y, ax in datasets:
        # Separiamo le classi
        # X[y == 0] prende tutte le epoche Non-Target
        # .mean(axis=0) fa la media su tutte le epoche -> otteniamo (Canali, Tempi)
        # .mean(axis=0) fa la media su tutti i canali -> otteniamo (Tempi,)
        erp_nontarget = X[y == 0].mean(axis=0).mean(axis=0)
        erp_target    = X[y == 1].mean(axis=0).mean(axis=0)
        
        # Plot Non-Target
        ax.plot(times, erp_nontarget, color='blue', linestyle='--', label='Non-Target', alpha=0.7)
        
        # Plot Target (P300)
        ax.plot(times, erp_target, color='red', linewidth=2.5, label='Target (P300)')
        
        # Abbellimenti
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Tempo (s)")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Evidenzia l'area dove ci aspettiamo la P300 (0.3s - 0.5s)
        ax.axvspan(0.3, 0.5, color='gray', alpha=0.1, label='Finestra P300')
        
        # Linee zero
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)

    # Legenda e Label asse Y
    axes[0].set_ylabel("Ampiezza (Z-Score / Std Dev)")
    axes[0].legend(loc='upper right')
    
    plt.suptitle("Confronto ERP su Dati Normalizzati", fontsize=16)
    plt.tight_layout()
    plt.show()



# --- ESEMPI DI UTILIZZO ---