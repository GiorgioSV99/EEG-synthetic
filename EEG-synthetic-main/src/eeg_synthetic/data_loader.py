"""
Data loader module for BCI AUT P300 dataset.

This module provides the BCIAUTLoader class for loading, preprocessing,
and normalizing EEG data from the BCI AUT P300 dataset.
"""

from typing import List, Optional, Tuple
import os

import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from typing import Union

class BCIAUTLoader:
    """
    Loader for the BCI AUT P300 dataset.
    
    This class handles loading EEG data from the BCI AUT dataset, applies
    preprocessing (filtering, resampling, baseline correction), and provides
    normalization utilities.
    
    The BCI AUT dataset contains P300 event-related potentials (ERPs) from
    15 subjects across 7 sessions each, with 8 EEG channels.
    
    Attributes:
        base_path (str): Root directory containing the dataset
        sfreq_orig (int): Original sampling frequency in Hz
        new_sfreq (int): Target sampling frequency after resampling in Hz
        tmin (float): Start time of epoch in seconds (relative to event)
        tmax (float): End time of epoch in seconds (relative to event)
        ch_names (List[str]): List of EEG channel names
        montage_name (str): Name of the electrode montage
        
    Example:
        >>> loader = BCIAUTLoader(base_path='data/', new_sfreq=128)
        >>> X, y, subjects, sessions = loader.get_data(
        ...     subjects=[1, 2, 3],
        ...     sessions='all',
        ...     modes='Train'
        ... )
        >>> print(f"Loaded {X.shape[0]} epochs with shape {X.shape}")
    """
    
    def __init__(
        self,
        base_path: str,
        sfreq_orig: int = 250,
        new_sfreq: int = 128,
        tmin: float = -0.2,
        tmax: float = 0.8,
        ch_names: Optional[List[str]] = None,
        montage_name: str = 'standard_1020'
    ):
        """
        Initialize the BCI AUT data loader.
        
        Args:
            base_path: Root directory containing the dataset structure
            sfreq_orig: Original sampling frequency in Hz (default: 250)
            new_sfreq: Target sampling frequency after resampling in Hz (default: 128)
            tmin: Start time of epoch in seconds, relative to event (default: -0.2)
            tmax: End time of epoch in seconds, relative to event (default: 0.8)
            ch_names: List of EEG channel names. If None, uses default 8 channels
            montage_name: Name of electrode montage for spatial information (default: 'standard_1020')
            
        Note:
            The default channel names are: ['C3', 'Cz', 'C4', 'CPz', 'P3', 'Pz', 'P4', 'POz']
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

    def _format_ids(self, subject_int: int, session_int: int) -> Tuple[str, str]:
        """
        Format subject and session integers into string identifiers.
        
        Args:
            subject_int: Subject number (1-15)
            session_int: Session number (1-7)
            
        Returns:
            Tuple of (subject_string, session_string)
            
        Example:
            >>> loader._format_ids(3, 6)
            ('SBJ03', 'S06')
        """
        sbj_str = f"SBJ{subject_int:02d}"
        sess_str = f"S{session_int:02d}"
        return sbj_str, sess_str

    def _load_single_epoch(
        self,
        subject_str: str,
        session_str: str,
        mode: str
    ) -> Optional[mne.EpochsArray]:
        """
        Load and preprocess a single session's EEG data.
        
        This method loads raw data from .mat files, applies preprocessing
        (filtering, resampling, baseline correction), and creates an MNE
        EpochsArray object.
        
        Preprocessing steps:
        1. Load data from .mat file and targets from .txt file
        2. Create MNE info structure and EpochsArray
        3. Set electrode montage for spatial information
        4. Apply bandpass filter (0.1-15 Hz)
        5. Resample to target frequency
        6. Crop to specified time window
        7. Apply baseline correction
        
        Args:
            subject_str: Subject identifier (e.g., 'SBJ01')
            session_str: Session identifier (e.g., 'S01')
            mode: Data mode, either 'Train' or 'Test'
            
        Returns:
            MNE EpochsArray object with preprocessed data, or None if files not found
            
        Note:
            Expected file structure:
            base_path/SBJ01/S01/Train/trainData.mat
            base_path/SBJ01/S01/Train/trainTargets.txt
        """
        folder_path = os.path.join(self.base_path, subject_str, session_str, mode)
        data_file = 'trainData.mat' if mode == 'Train' else 'testData.mat'
        target_file = 'trainTargets.txt' if mode == 'Train' else 'testTargets.txt'

        data_path = os.path.join(folder_path, data_file)
        target_path = os.path.join(folder_path, target_file)

        # Check file existence
        if not os.path.exists(data_path) or not os.path.exists(target_path):
            print(f"Warning: File not found: {data_path}")
            return None

        # Load MAT file
        try:
            mat = scipy.io.loadmat(data_path)
            key = data_file.split('.')[0]
            data_original = mat[key]
        except Exception as e:
            print(f"Error loading .mat file {data_path}: {e}")
            return None

        # Transpose to MNE format: (Epochs, Channels, Times)
        data_mne = np.transpose(data_original, (2, 0, 1))

        # Load target labels
        targets = np.loadtxt(target_path).astype(int)

        # Create MNE Info and EpochsArray
        info = mne.create_info(
            ch_names=self.ch_names,
            sfreq=self.sfreq_orig,
            ch_types='eeg'
        )
        epochs = mne.EpochsArray(data_mne, info, tmin=self.tmin, verbose=False)
        
        # Set electrode montage for spatial information
        montage = mne.channels.make_standard_montage(self.montage_name)
        epochs.set_montage(montage)

        # Apply preprocessing pipeline
        epochs.filter(l_freq=0.1, h_freq=15.0, fir_design='firwin', verbose=False)
        epochs.resample(self.new_sfreq, verbose=False)
        epochs.crop(tmax=self.tmax, verbose=False)
        epochs.apply_baseline(baseline=(None, 0), verbose=False)

        # Assign event information
        events = np.zeros((len(targets), 3), dtype=int)
        events[:, 0] = np.arange(len(targets))
        events[:, 2] = targets
        epochs.events = events
        epochs.event_id = {'Non-Target': 0, 'Target': 1}

        return epochs

    def get_data(
        self,
        subjects: Union[int, List[int], str],
        sessions: Union[int, List[int], str],
        modes: Union[str, List[str]] = ['Train', 'Test'],
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load EEG data for specified subjects, sessions, and modes.
        
        This is the main method for loading data. It handles multiple subjects
        and sessions, applies preprocessing, and returns numpy arrays ready
        for machine learning.
        
        Args:
            subjects: Subject ID(s) to load. Can be:
                - Single integer (e.g., 1)
                - List of integers (e.g., [1, 2, 3])
                - String 'all' to load all 15 subjects
            sessions: Session ID(s) to load. Can be:
                - Single integer (e.g., 1)
                - List of integers (e.g., [1, 2, 3])
                - String 'all' to load all 7 sessions
            modes: Data mode(s) to load. Can be:
                - Single string: 'Train' or 'Test'
                - List of strings: ['Train', 'Test']
            verbose: Whether to print loading progress
            
        Returns:
            Tuple containing:
                - X: EEG data array of shape (n_epochs, n_channels, n_times)
                - y: Labels array of shape (n_epochs,) with values 0 (Non-Target) or 1 (Target)
                - subjects_arr: Subject IDs for each epoch, shape (n_epochs,)
                - sessions_arr: Session IDs for each epoch, shape (n_epochs,)
                
        Raises:
            ValueError: If no data is found with the specified parameters
            
        Example:
            >>> loader = BCIAUTLoader(base_path='data/')
            >>> # Load all sessions for subjects 1-3, train data only
            >>> X, y, subj, sess = loader.get_data(
            ...     subjects=[1, 2, 3],
            ...     sessions='all',
            ...     modes='Train'
            ... )
            >>> print(f"Shape: {X.shape}, Classes: {np.unique(y)}")
            Shape: (1200, 8, 128), Classes: [0 1]
        """
        # Handle 'all' shortcut
        if subjects == 'all':
            subjects = list(range(1, 16))
        if sessions == 'all':
            sessions = list(range(1, 8))
            
        # Convert single values to lists
        if isinstance(subjects, int):
            subjects = [subjects]
        if isinstance(sessions, int):
            sessions = [sessions]
        if isinstance(modes, str):
            modes = [modes]

        epochs_list = []
        meta_subjects = []
        meta_sessions = []
        
        if verbose:
            print(f"Loading - Subjects: {subjects}, Sessions: {sessions}, Modes: {modes}")

        # Load data for each subject/session/mode combination
        for subj in subjects:
            for sess in sessions:
                sbj_str, sess_str = self._format_ids(subj, sess)
                
                for mode in modes:
                    ep = self._load_single_epoch(sbj_str, sess_str, mode)
                    
                    if ep is not None:
                        epochs_list.append(ep)
                        
                        # Track metadata for each trial
                        n_trials_in_epoch = len(ep)
                        meta_subjects.append(np.full(n_trials_in_epoch, subj))
                        meta_sessions.append(np.full(n_trials_in_epoch, sess))

        if not epochs_list:
            raise ValueError("No data found with the specified parameters.")

        # Concatenate all loaded epochs
        all_epochs = mne.concatenate_epochs(epochs_list, verbose=False)
        
        # Extract numpy arrays
        X = all_epochs.get_data()[:, :, :-1]  # Remove last time point
        y = all_epochs.events[:, 2]
        
        # Concatenate metadata arrays
        subjects_arr = np.concatenate(meta_subjects)
        sessions_arr = np.concatenate(meta_sessions)
        
        if verbose:
            print(f"-> Total loaded: {X.shape} - Classes: {np.unique(y, return_counts=True)}")
            
        return X, y, subjects_arr, sessions_arr

    def normalize_data_z_score(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize data using Z-score standardization.
        
        Applies Z-score normalization (zero mean, unit variance) using
        statistics computed from the training set. The same statistics
        are applied to the test set to prevent data leakage.
        
        Formula: X_norm = (X - mean) / std
        
        Args:
            X_train: Training data of shape (n_epochs, n_channels, n_times)
            X_test: Test data of shape (n_epochs, n_channels, n_times)
            
        Returns:
            Tuple of (X_train_normalized, X_test_normalized)
            
        Note:
            Test set is normalized using training set statistics to prevent
            data leakage, which is the correct approach for machine learning.
            
        Example:
            >>> X_train_norm, X_test_norm = loader.normalize_data_z_score(
            ...     X_train, X_test
            ... )
            >>> print(f"Train mean: {X_train_norm.mean():.4f}")
            Train mean: 0.0000
        """
        mu = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True)
        
        X_train_norm = (X_train - mu) / std
        X_test_norm = (X_test - mu) / std
        
        return X_train_norm, X_test_norm
    
    def normalize_data_minmax(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize data using Min-Max scaling to [0, 1] range.
        
        Scales data to the range [0, 1] using the global minimum and maximum
        values from the training set. The same scaling is applied to the test
        set to prevent data leakage.
        
        Formula: X_norm = (X - min) / (max - min)
        
        Args:
            X_train: Training data of shape (n_epochs, n_channels, n_times)
            X_test: Test data of shape (n_epochs, n_channels, n_times)
            
        Returns:
            Tuple of (X_train_normalized, X_test_normalized)
            
        Note:
            Uses global min/max from training set for both train and test data.
            
        Example:
            >>> X_train_norm, X_test_norm = loader.normalize_data_minmax(
            ...     X_train, X_test
            ... )
            >>> print(f"Train range: [{X_train_norm.min():.2f}, {X_train_norm.max():.2f}]")
            Train range: [0.00, 1.00]
        """
        x_min = np.min(X_train)
        x_max = np.max(X_train)

        X_train_norm = (X_train - x_min) / (x_max - x_min)
        X_test_norm = (X_test - x_min) / (x_max - x_min)
        
        return X_train_norm, X_test_norm



def plot_normalized_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sfreq: int = 128,
    tmin: float = -0.2
) -> None:
    """
    Plot averaged Event-Related Potentials (ERPs) for normalized data.
    
    Creates a visualization comparing Target vs Non-Target ERPs for both
    training and test sets. The plot shows the grand average across all
    channels and epochs for each class.
    
    The P300 component is typically visible as a positive deflection
    around 300-500ms after stimulus onset for Target stimuli.
    
    Args:
        X_train: Training data of shape (n_epochs, n_channels, n_times)
        y_train: Training labels of shape (n_epochs,) with values 0 or 1
        X_test: Test data of shape (n_epochs, n_channels, n_times)
        y_test: Test labels of shape (n_epochs,) with values 0 or 1
        sfreq: Sampling frequency in Hz (default: 128)
        tmin: Start time of epoch in seconds (default: -0.2)
        
    Returns:
        None (displays matplotlib figure)
        
    Note:
        The gray shaded area (0.3-0.5s) highlights the typical P300 time window.
        
    Example:
        >>> X_train_norm, X_test_norm = loader.normalize_data_z_score(X_train, X_test)
        >>> plot_normalized_arrays(X_train_norm, y_train, X_test_norm, y_test)
    """
    # Create time axis
    n_samples = X_train.shape[2]
    duration = n_samples / sfreq
    times = np.linspace(tmin, tmin + duration, n_samples)

    # Setup figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    datasets = [
        ("TRAIN Set (Normalized)", X_train, y_train, axes[0]),
        ("TEST Set (Normalized)", X_test, y_test, axes[1])
    ]

    print("--- Generating ERP Plots ---")

    for title, X, y, ax in datasets:
        # Compute grand average ERPs for each class
        # Average across epochs, then across channels
        erp_nontarget = X[y == 0].mean(axis=0).mean(axis=0)
        erp_target = X[y == 1].mean(axis=0).mean(axis=0)
        
        # Plot Non-Target ERP
        ax.plot(
            times, erp_nontarget,
            color='blue', linestyle='--',
            label='Non-Target', alpha=0.7
        )
        
        # Plot Target ERP (P300)
        ax.plot(
            times, erp_target,
            color='red', linewidth=2.5,
            label='Target (P300)'
        )
        
        # Styling
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Highlight P300 time window (300-500ms)
        ax.axvspan(0.3, 0.5, color='gray', alpha=0.1, label='P300 Window')
        
        # Add reference lines
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)

    # Labels and legend
    axes[0].set_ylabel("Amplitude (Z-Score / Std Dev)")
    axes[0].legend(loc='upper right')
    
    plt.suptitle("Event-Related Potential Comparison (Normalized Data)", fontsize=16)
    plt.tight_layout()
    plt.show()