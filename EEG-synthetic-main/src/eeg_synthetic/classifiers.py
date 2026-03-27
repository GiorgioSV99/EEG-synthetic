"""
Classification module for EEG P300 detection.

This module provides multiple classification approaches for P300 detection:
1. Riemannian geometry-based classification (xDAWN + MDM)
2. Deep learning with EEGNet
3. Traditional machine learning classifiers

All classifiers support SMOTE for handling class imbalance.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import cycle
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score
)
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM
from sklearn.pipeline import Pipeline

from oversampling import apply_smote_3d


# ============================================================================
# RIEMANNIAN GEOMETRY CLASSIFICATION
# ============================================================================

def run_p300_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_smote: bool = True
) -> Tuple[Pipeline, np.ndarray]:
    """
    Run P300 classification using Riemannian geometry.
    
    This function implements a state-of-the-art pipeline for EEG classification
    based on Riemannian geometry:
    1. xDAWN spatial filtering to enhance P300 component
    2. Covariance matrix estimation
    3. Minimum Distance to Mean (MDM) classification in Riemannian space
    
    The Riemannian approach is particularly effective for EEG because:
    - Covariance matrices lie on a Riemannian manifold
    - Euclidean metrics are inappropriate for covariance matrices
    - Riemannian metrics respect the manifold structure
    
    Args:
        X_train: Training data of shape (n_epochs, n_channels, n_times)
        y_train: Training labels of shape (n_epochs,)
        X_test: Test data of shape (n_epochs, n_channels, n_times)
        y_test: Test labels of shape (n_epochs,)
        use_smote: Whether to apply SMOTE for class balancing
        
    Returns:
        Tuple containing:
            - clf: Trained scikit-learn Pipeline
            - y_pred: Predictions on test set
            
    Example:
        >>> clf, y_pred = run_p300_classification(
        ...     X_train, y_train, X_test, y_test, use_smote=True
        ... )
        >>> print(classification_report(y_test, y_pred))
        
    References:
        Barachant, A., Bonnet, S., Congedo, M., & Jutten, C. (2012).
        Multiclass brain–computer interface classification by Riemannian geometry.
        IEEE Transactions on Biomedical Engineering, 59(4), 920-928.
    """
    print(f"\n--- P300 Classification (SMOTE: {'ON' if use_smote else 'OFF'}) ---")

    # Apply SMOTE if requested
    if use_smote:
        X_train_proc, y_train_proc = apply_smote_3d(X_train, y_train)
    else:
        X_train_proc, y_train_proc = X_train, y_train
        print(f"Using original data: {X_train_proc.shape[0]} epochs")

    # Define Riemannian pipeline
    clf = Pipeline([
        ('xdawn', XdawnCovariances(nfilter=4, estimator='oas')),
        ('mdm', MDM(metric='riemann'))
    ])

    # Train classifier
    clf.fit(X_train_proc, y_train_proc)

    # Predict and evaluate
    y_pred = clf.predict(X_test)

    # Calculate per-class accuracy (recall)
    cm = confusion_matrix(y_test, y_pred)
    acc_non_target = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    acc_target = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0

    print("-" * 50)
    print(f"Non-Target Accuracy (Recall): {acc_non_target:.2%}")
    print(f"Target Accuracy (Recall):     {acc_target:.2%}")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=['Non-Target', 'Target']))

    return clf, y_pred


def classify_eeg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    clf: Any = None,
    use_smote: bool = True
) -> pd.DataFrame:
    """
    General EEG classification function supporting multiple classifier types.
    
    This function provides a flexible interface for training and evaluating
    various classifiers on EEG data. It automatically handles:
    - SMOTE for class balancing
    - Data reshaping for non-pipeline classifiers
    - Performance evaluation with F1 scores
    
    Args:
        X_train: Training data of shape (n_epochs, n_channels, n_times)
        y_train: Training labels of shape (n_epochs,)
        X_test: Test data of shape (n_epochs, n_channels, n_times)
        y_test: Test labels of shape (n_epochs,)
        clf: Classifier object. If None, uses LogisticRegression.
            Can be any sklearn-compatible classifier or Pipeline.
        use_smote: Whether to apply SMOTE for class balancing
        
    Returns:
        DataFrame with F1 scores for each class on train and test sets
        
    Example:
        >>> from sklearn.svm import SVC
        >>> clf = SVC(kernel='rbf', C=1.0)
        >>> results = classify_eeg(X_train, y_train, X_test, y_test, clf=clf)
        >>> print(results)
           Class  Train_F1  Test_F1
        0  Class 0    0.95     0.92
        1  Class 1    0.93     0.90
    """
    if clf is None:
        clf = LogisticRegression(max_iter=1000, class_weight=None)
    
    # Apply SMOTE if requested
    if use_smote:
        X_processed, y_train_final = apply_smote_3d(X_train, y_train)
        status_smote = "ON"
        status_text = "Balancing completed"
    else:
        X_processed, y_train_final = X_train, y_train
        status_smote = "OFF"
        status_text = "Using original data"

    # Reshape data if not using Pipeline
    if not hasattr(clf, 'steps'):
        X_train_final = X_processed.reshape(X_processed.shape[0], -1)
        X_test_final = X_test.reshape(X_test.shape[0], -1)
    else:
        X_train_final = X_processed
        X_test_final = X_test

    # Train classifier
    clf.fit(X_train_final, y_train_final)

    # Predict
    y_pred_test = clf.predict(X_test_final)

    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred_test)
    acc_non_target = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    acc_target = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0

    # Print results
    print(f"--- EEG Classification (SMOTE: {status_smote}) ---")
    print(f"{status_text}: {X_train_final.shape[0]} epochs")
    print("-" * 50)
    print(f"Non-Target Accuracy (Recall): {acc_non_target:.2%}")
    print(f"Target Accuracy (Recall):     {acc_target:.2%}")
    print("-" * 50)
    print(classification_report(y_test, y_pred_test, target_names=['Non-Target', 'Target']))

    # Calculate F1 scores
    classes = np.unique(y_train)
    y_pred_train = clf.predict(X_train_final)
    f1_train = f1_score(y_train_final, y_pred_train, average=None)
    f1_test = f1_score(y_test, y_pred_test, average=None)

    df_result = pd.DataFrame({
        'Class': [f"Class {int(x)}" for x in classes],
        'Train_F1': f1_train,
        'Test_F1': f1_test
    })

    return df_result


# ============================================================================
# EEGNET DEEP LEARNING MODEL
# ============================================================================

class EEGNetModel(nn.Module):
    """
    EEGNet: Compact Convolutional Neural Network for EEG-based BCIs.
    
    EEGNet is a compact CNN architecture specifically designed for EEG
    classification. It uses depthwise and separable convolutions to
    reduce parameters while maintaining performance.
    
    Architecture:
    1. Temporal convolution (learns temporal filters)
    2. Depthwise spatial convolution (learns spatial filters per temporal filter)
    3. Separable convolution (learns feature combinations)
    4. Fully connected layer for classification
    
    Key features:
    - Depthwise convolutions reduce parameters
    - Max-norm constraints prevent overfitting
    - Dropout for regularization
    - ELU activation for better gradients
    
    Args:
        chans: Number of EEG channels (default: 8)
        classes: Number of output classes (default: 2)
        time_points: Number of time points per epoch (default: 64)
        temp_kernel: Temporal convolution kernel size (default: 16)
        f1: Number of temporal filters (default: 8)
        f2: Number of pointwise filters (default: 16)
        d: Depth multiplier for depthwise convolution (default: 2)
        pk1: First pooling kernel size (default: 4)
        pk2: Second pooling kernel size (default: 4)
        dropout_rate: Dropout probability (default: 0.5)
        max_norm1: Max-norm constraint for depthwise layer (default: 1.0)
        max_norm2: Max-norm constraint for FC layer (default: 0.25)
        
    Example:
        >>> model = EEGNetModel(chans=8, classes=2, time_points=128)
        >>> x = torch.randn(32, 1, 8, 128)  # (batch, 1, channels, time)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([32, 2])
        
    References:
        Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural
        network for EEG-based brain–computer interfaces. Journal of Neural
        Engineering, 15(5), 056013.
    """
    
    def __init__(
        self,
        chans: int = 8,
        classes: int = 2,
        time_points: int = 64,
        temp_kernel: int = 16,
        f1: int = 8,
        f2: int = 16,
        d: int = 2,
        pk1: int = 4,
        pk2: int = 4,
        dropout_rate: float = 0.5,
        max_norm1: float = 1.0,
        max_norm2: float = 0.25
    ):
        super(EEGNetModel, self).__init__()
        
        # Calculate fully connected layer input size
        linear_size = (time_points // (pk1 * pk2)) * f2

        # Block 1: Temporal convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, temp_kernel), padding='same', bias=False),
            nn.BatchNorm2d(f1),
        )
        
        # Block 2: Depthwise spatial convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1)),
            nn.Dropout(dropout_rate)
        )
        
        # Block 3: Separable convolution
        self.block3 = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16), groups=f2, bias=False, padding='same'),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2)),
            nn.Dropout(dropout_rate)
        )
        
        # Classifier
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(linear_size, classes)

        # Apply max-norm constraints
        self._apply_max_norm(self.block2[0], max_norm1)
        self._apply_max_norm(self.fc, max_norm2)

    def _apply_max_norm(self, layer: nn.Module, max_norm: float) -> None:
        """Apply max-norm weight constraint to a layer."""
        for name, param in layer.named_parameters():
            if 'weight' in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EEGNet.
        
        Args:
            x: Input tensor of shape (batch, 1, channels, time_points)
            
        Returns:
            Output logits of shape (batch, classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

class TrainModel:
    """
    Trainer class for EEGNet with support for GAN-based data augmentation.
    
    This trainer supports three training scenarios:
    1. **Imbalanced**: Train on imbalanced real data with oversampling
    2. **SMOTE**: Train on SMOTE-balanced data
    3. **GAN**: Train on mix of real and GAN-generated synthetic data
    
    The trainer uses a balanced batch strategy where each batch contains
    equal numbers of samples from each class.
    
    Attributes:
        device: PyTorch device (CPU or CUDA)
        
    Example:
        >>> trainer = TrainModel()
        >>> model = EEGNetModel(chans=8, classes=2, time_points=128)
        >>> trained_model = trainer.train_model(
        ...     model=model,
        ...     real_0_ds=dataset_class0,
        ...     real_1_ds=dataset_class1,
        ...     epochs=50,
        ...     batch_size=64
        ... )
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def train_model(
        self,
        model: nn.Module,
        real_0_ds: TensorDataset,
        real_1_ds: TensorDataset,
        fake_1_ds: Optional[TensorDataset] = None,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 50,
        fake_ratio_in_c1: float = 0.0
    ) -> nn.Module:
        """
        Train EEGNet model with optional GAN-based augmentation.
        
        Training scenarios:
        1. **Balanced real data** (fake_1_ds=None, fake_ratio=0):
           Uses oversampling to balance classes with real data only
           
        2. **SMOTE data** (pass SMOTE data as real_1_ds, fake_1_ds=None):
           Trains on SMOTE-augmented data
           
        3. **GAN augmentation** (provide fake_1_ds, set fake_ratio > 0):
           Mixes real and synthetic data for minority class
           fake_ratio controls the proportion of synthetic samples
        
        Args:
            model: EEGNet model instance
            real_0_ds: Dataset for majority class (Non-Target)
            real_1_ds: Dataset for minority class (Target)
            fake_1_ds: Optional dataset with GAN-generated samples for class 1
            learning_rate: Learning rate for Adam optimizer
            batch_size: Total batch size (split equally between classes)
            epochs: Number of training epochs
            fake_ratio_in_c1: Ratio of synthetic to total class 1 samples (0.0-1.0)
                - 0.0: No synthetic data
                - 0.5: Half synthetic, half real
                - 1.0: All synthetic
                
        Returns:
            Trained model
            
        Note:
            Model is saved to 'eegnet_model.pth' after training
        """
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Calculate batch composition
        n_c1_total = batch_size // 2
        n_c0 = batch_size - n_c1_total
        
        # Determine if using synthetic data
        use_fake = (fake_1_ds is not None and len(fake_1_ds) > 0 and fake_ratio_in_c1 > 0)
        
        if not use_fake:
            n_fake_1 = 0
            n_real_1 = n_c1_total
            print(f"MODE: Balanced Real Data (Oversampling class 1)")
        else:
            n_fake_1 = int(n_c1_total * fake_ratio_in_c1)
            n_fake_1 = max(1, n_fake_1)
            n_real_1 = n_c1_total - n_fake_1
            print(f"MODE: GAN Augmentation (Fake ratio: {fake_ratio_in_c1:.2%})")

        print(f"Batch composition: Class0={n_c0}, Class1_Real={n_real_1}, Class1_Fake={n_fake_1}")

        # Setup data loaders
        loader_0 = DataLoader(real_0_ds, batch_size=n_c0, shuffle=True, drop_last=True)
        loader_1_real = cycle(DataLoader(real_1_ds, batch_size=n_real_1, shuffle=True, drop_last=True))
        
        if use_fake:
            loader_1_fake = cycle(DataLoader(fake_1_ds, batch_size=n_fake_1, shuffle=True, drop_last=True))

        best_train_accuracy = 0.0

        # Training loop
        for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            
            for batch_0, label_0 in loader_0:
                batch_1r, label_1r = next(loader_1_real)
                
                # Combine batches
                if use_fake:
                    batch_1f, label_1f = next(loader_1_fake)
                    inputs = torch.cat([batch_0, batch_1r, batch_1f], dim=0)
                    labels = torch.cat([label_0, label_1r, label_1f], dim=0)
                else:
                    inputs = torch.cat([batch_0, batch_1r], dim=0)
                    labels = torch.cat([label_0, label_1r], dim=0)

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Shuffle within batch
                perm = torch.randperm(inputs.size(0))
                inputs, labels = inputs[perm], labels[perm]

                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()

                # Track metrics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_acc = correct / total
            if epoch_acc > best_train_accuracy:
                best_train_accuracy = epoch_acc
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/total:.4f}, "
                      f"Accuracy: {epoch_acc*100:.2f}%")

        print(f"Training completed. Best accuracy: {best_train_accuracy*100:.2f}%")
        torch.save(model.state_dict(), 'eegnet_model.pth')
        print("Model saved to 'eegnet_model.pth'")
        
        return model


class EvalModel:
    """
    Evaluation class for trained EEGNet models.
    
    Provides methods for:
    - Testing model accuracy
    - Plotting confusion matrices
    - Generating classification reports
    
    Attributes:
        model: Trained EEGNet model
        device: PyTorch device (CPU or CUDA)
        
    Example:
        >>> evaluator = EvalModel(trained_model)
        >>> accuracy = evaluator.test_model(test_dataset)
        >>> evaluator.plot_confusion_matrix(test_dataset, ['Non-Target', 'Target'])
    """
    
    def __init__(self, model: nn.Module):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def test_model(self, test_dataset: TensorDataset) -> float:
        """
        Evaluate model on test dataset.
        
        Args:
            test_dataset: PyTorch TensorDataset with test data
            
        Returns:
            Test accuracy as percentage (0-100)
        """
        self.model.eval()
        correct = 0
        total = 0
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = (correct / total) * 100
        print("=" * 50)
        print(f"Test Accuracy: {accuracy:.2f}%")
        print("=" * 50)
        return accuracy

    def plot_confusion_matrix(
        self,
        test_dataset: TensorDataset,
        classes: List[str]
    ) -> None:
        """
        Plot normalized confusion matrix for test dataset.
        
        Args:
            test_dataset: PyTorch TensorDataset with test data
            classes: List of class names for axis labels
            
        Returns:
            None (saves plot to 'confusion_matrix_model.png')
        """
        self.model.eval()
        y_pred = []
        y_true = []
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        # Normalize confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]

        df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)

        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='.2%', cbar_kws={'format': '%.0f%%'})
        plt.xlabel('Predicted Labels', fontsize=12)
        plt.ylabel('True Labels', fontsize=12)
        plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('confusion_matrix_model.png', dpi=300)
        print("Confusion matrix saved to 'confusion_matrix_model.png'")
        plt.show()
