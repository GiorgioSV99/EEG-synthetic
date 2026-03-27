"""
EEG Synthetic Data Generation and Classification Package

This package provides tools for:
- Loading and preprocessing EEG data (BCI AUT dataset)
- Applying oversampling techniques (SMOTE)
- Training classifiers (Riemannian geometry, EEGNet)
- Calculating dataset complexity metrics
- Generating synthetic EEG data using GANs
"""

from .data_loader import BCIAUTLoader
from .oversampling import apply_smote_3d
from .complexity_metrics import calculate_complexity_metrics

__version__ = "0.1.0"
__all__ = [
    "BCIAUTLoader",
    "apply_smote_3d",
    "calculate_complexity_metrics",
]
