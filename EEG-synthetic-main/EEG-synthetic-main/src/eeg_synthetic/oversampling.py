"""
Oversampling module for imbalanced EEG data.

This module provides SMOTE (Synthetic Minority Over-sampling Technique)
adapted for 3D EEG data arrays.
"""

from typing import Tuple

import numpy as np
from imblearn.over_sampling import SMOTE


def apply_smote_3d(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    k_neighbors: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to 3D EEG data (epochs, channels, time points).
    
    SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic
    samples for the minority class by interpolating between existing samples.
    This implementation handles 3D EEG data by flattening, applying SMOTE,
    and reshaping back to 3D.
    
    The P300 dataset is typically highly imbalanced (e.g., 1:5 ratio of
    Target to Non-Target), making SMOTE useful for improving classifier
    performance.
    
    Args:
        X: EEG data array of shape (n_epochs, n_channels, n_times)
        y: Labels array of shape (n_epochs,) with binary class labels
        random_state: Random seed for reproducibility (default: 42)
        k_neighbors: Number of nearest neighbors for SMOTE interpolation (default: 5)
        
    Returns:
        Tuple containing:
            - X_resampled: Balanced EEG data of shape (n_epochs_new, n_channels, n_times)
            - y_resampled: Balanced labels of shape (n_epochs_new,)
            
    Raises:
        ValueError: If X is not 3D or if y length doesn't match X
        
    Example:
        >>> X.shape, y.shape
        ((1000, 8, 128), (1000,))
        >>> np.bincount(y)
        array([800, 200])  # Imbalanced: 800 class 0, 200 class 1
        >>> X_balanced, y_balanced = apply_smote_3d(X, y)
        >>> X_balanced.shape, y_balanced.shape
        ((1600, 8, 128), (1600,))
        >>> np.bincount(y_balanced)
        array([800, 800])  # Balanced: 800 class 0, 800 class 1
        
    Note:
        - SMOTE only generates synthetic samples for the minority class
        - The majority class samples are kept unchanged
        - Synthetic samples are created by interpolating between k nearest neighbors
        - For highly imbalanced data (ratio > 1:10), consider using SMOTE with
          sampling_strategy parameter or other techniques like ADASYN
          
    References:
        Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).
        SMOTE: synthetic minority over-sampling technique. Journal of artificial
        intelligence research, 16, 321-357.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array, got {X.ndim}D array")
    
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} != {len(y)}")
    
    n_epochs, n_channels, n_times = X.shape

    # Flatten to 2D for SMOTE: (n_epochs, n_channels * n_times)
    X_flat = X.reshape(n_epochs, -1)

    # Apply SMOTE
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_resampled_flat, y_resampled = smote.fit_resample(X_flat, y)

    # Reshape back to 3D: (n_epochs_new, n_channels, n_times)
    X_resampled_3d = X_resampled_flat.reshape(-1, n_channels, n_times)

    print(f"SMOTE completed: {n_epochs} -> {X_resampled_3d.shape[0]} epochs")
    print(f"Class distribution before: {np.bincount(y)}")
    print(f"Class distribution after: {np.bincount(y_resampled)}")
    
    return X_resampled_3d, y_resampled
