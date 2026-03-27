"""
Dataset complexity metrics module.

This module provides functions to calculate complexity metrics for EEG datasets,
which are useful for evaluating the quality of synthetic data and understanding
dataset difficulty.
"""

from typing import Literal, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from pycol_complexity import complexity


def calculate_complexity_metrics(
    X: np.ndarray,
    y: np.ndarray,
    method: Literal['pca', 'spatial'] = 'pca',
    channel_idx: Optional[int] = None,
    n_components: int = 50,
    subset_ratio: float = 0.10,
    random_state: int = 42,
    imb: bool = True
) -> complexity.Complexity:
    """
    Calculate dataset complexity metrics for EEG data.
    
    Complexity metrics quantify how difficult a classification problem is.
    These metrics are useful for:
    - Evaluating synthetic data quality (should match real data complexity)
    - Understanding why certain classifiers perform better
    - Identifying problematic regions in feature space
    - Comparing different preprocessing strategies
    
    Two preprocessing strategies are available:
    1. **PCA**: Reduces dimensionality using Principal Component Analysis
    2. **Spatial**: Uses spatial averaging or single channel selection
    
    Computed Metrics:
    - **Instance-level**:
        - kDN: k-Discordant Neighbors (ratio of neighbors with different labels)
        - N3: Error rate of nearest neighbor classifier
        - N4: Non-linearity of nearest neighbor classifier
    - **Structural**:
        - N1: Fraction of borderline points (near decision boundary)
        - N2: Ratio of intra-class to inter-class nearest neighbor distance
    - **Multi-resolution**:
        - C1: Entropy of class proportions
        - C2: Imbalance ratio (KL divergence)
    
    Args:
        X: EEG data array of shape (n_epochs, n_channels, n_times)
        y: Labels array of shape (n_epochs,) with class labels
        method: Preprocessing method, either 'pca' or 'spatial'
        channel_idx: For 'spatial' method:
            - None: Average across all channels
            - int: Use only the specified channel (e.g., 5 for Pz)
        n_components: Number of PCA components to retain (for 'pca' method)
        subset_ratio: Fraction of data to use for complexity calculation (0.0-1.0)
            Using a subset (e.g., 0.10 = 10%) speeds up computation significantly
        random_state: Random seed for reproducibility
        imb: Whether to use imbalanced-aware versions of metrics
        
    Returns:
        Complexity object with calculated metrics accessible as attributes
        (e.g., comp_obj.N1, comp_obj.N2, etc.)
        
    Raises:
        ValueError: If method is not 'pca' or 'spatial'
        
    Example:
        >>> # Using PCA preprocessing
        >>> comp = calculate_complexity_metrics(
        ...     X, y, method='pca', n_components=50, subset_ratio=0.10
        ... )
        >>> print(f"N1 (Borderline): {comp.N1:.4f}")
        >>> print(f"N3 (NN Error): {comp.N3:.4f}")
        
        >>> # Using spatial averaging
        >>> comp = calculate_complexity_metrics(
        ...     X, y, method='spatial', channel_idx=None
        ... )
        
        >>> # Using single channel (Pz = index 5)
        >>> comp = calculate_complexity_metrics(
        ...     X, y, method='spatial', channel_idx=5
        ... )
        
    Note:
        - Complexity calculation can be slow for large datasets, hence subset_ratio
        - Higher complexity values generally indicate harder classification problems
        - N1 close to 1.0 indicates many borderline points (difficult problem)
        - N3 close to 0.5 indicates random-like classification (very difficult)
        - Use stratified sampling to maintain class proportions in subset
        
    References:
        Lorena, A. C., et al. (2019). How complex is your classification problem?
        A survey on measuring classification complexity. ACM Computing Surveys, 52(5).
    """
    print(f"--- Calculating Complexity | Method: {method.upper()} ---")
    
    # Preprocessing
    if method == 'spatial':
        if channel_idx is None:
            # Average across all channels: (n_epochs, n_times)
            print("Strategy: Spatial Averaging (All channels)")
            X_processed = X.mean(axis=1)
        else:
            # Select single channel: (n_epochs, n_times)
            print(f"Strategy: Single Channel (Index: {channel_idx})")
            X_processed = X[:, channel_idx, :]
            
    elif method == 'pca':
        # Flatten and apply PCA: (n_epochs, n_components)
        print(f"Strategy: PCA (Components: {n_components})")
        X_flat = X.reshape(X.shape[0], -1)
        
        pca = PCA(n_components=n_components, random_state=random_state)
        X_processed = pca.fit_transform(X_flat)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"PCA explained variance: {explained_var:.2%}")
        
    else:
        raise ValueError(f"Invalid method '{method}'. Choose 'pca' or 'spatial'.")

    # Stratified subsampling to speed up computation
    _, X_sub, _, y_sub = train_test_split(
        X_processed, 
        y, 
        test_size=subset_ratio, 
        stratify=y, 
        random_state=random_state
    )
    
    print(f"Dataset reduced for complexity: {X_sub.shape}")
    print(f"Class distribution: {np.bincount(y_sub.astype(int))}")

    # Calculate complexity metrics
    dataset_dict = {
        'X': X_sub,
        'y': y_sub.astype(int)
    }

    # Initialize complexity object
    comp_obj = complexity.Complexity(dataset=dataset_dict, file_type="array")

    # Compute metrics
    print("Computing complexity metrics...")
    
    # Instance-level metrics
    comp_obj.kDN(imb=imb)  # k-Discordant Neighbors
    comp_obj.N3(imb=imb)   # Error Rate of NN classifier
    comp_obj.N4(imb=imb)   # Non-Linearity of NN classifier

    # Structural metrics
    comp_obj.N1(imb=imb)   # Fraction of borderline points
    comp_obj.N2(imb=imb)   # Ratio of intra/extra class NN distance

    # Multi-resolution metrics
    comp_obj.C1(imb=imb)   # Entropy of class proportions
    comp_obj.C2(imb=imb)   # Imbalance ratio (KL divergence)
    
    print("Complexity calculation completed.")
    print(f"Results: N1={comp_obj.N1:.4f}, N2={comp_obj.N2:.4f}, N3={comp_obj.N3:.4f}")
    
    return comp_obj
