# EEG Synthetic Data Generation and Classification

A comprehensive Python package for EEG data processing, synthetic data generation using GANs, and classification using deep learning and Riemannian geometry approaches.

## 📋 Overview

This project provides tools for:
- **EEG Data Loading & Preprocessing**: Load and preprocess BCI AUT P300 dataset
- **Data Augmentation**: SMOTE-based oversampling for imbalanced EEG data
- **Classification**: Multiple classifiers including EEGNet, Riemannian geometry (MDM), and traditional ML
- **Complexity Analysis**: Dataset complexity metrics for evaluating synthetic data quality
- **GAN Integration**: Support for synthetic EEG data generation using the EEG-GAN framework

## 🚀 Features

### Data Processing
- **BCIAUTLoader**: Flexible data loader for BCI AUT P300 dataset
  - Automatic preprocessing (filtering, resampling, baseline correction)
  - Support for multiple subjects and sessions
  - Metadata tracking (subject IDs, session IDs)
  - Z-score and Min-Max normalization

### Classification Methods
- **EEGNet**: State-of-the-art deep learning architecture for EEG classification
- **Riemannian Geometry**: MDM (Minimum Distance to Mean) with xDAWN covariances
- **Traditional ML**: Logistic Regression with feature extraction

### Data Augmentation
- **SMOTE**: Synthetic Minority Over-sampling Technique adapted for 3D EEG data
- **GAN-based**: Integration with EEG-GAN for realistic synthetic EEG generation

### Complexity Metrics
- Instance-level metrics (kDN, N3, N4)
- Structural metrics (N1, N2)
- Multi-resolution metrics (C1, C2)
- Support for PCA and spatial averaging strategies

## 📁 Project Structure

```
EEG-synthetic/
├── src/
│   └── eeg_synthetic/
│       ├── __init__.py
│       ├── data_loader.py          # BCI AUT data loader
│       ├── classifiers.py          # Classification models
│       ├── oversampling.py         # SMOTE implementation
│       └── complexity_metrics.py   # Complexity analysis
├── notebooks/
│   ├── analysis.ipynb              # Main analysis notebook
│   ├── experiments.ipynb           # Experimental results
│   └── test.ipynb                  # Testing notebook
├── eeggan_external/                # External EEG-GAN framework
│   ├── eeggan/                     # GAN implementation
│   └── docs/                       # Documentation
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation
└── README.md                       # This file
```

## 🔬 Experiments

The `notebooks/` directory contains Jupyter notebooks with:
- **analysis.ipynb**: Complete analysis pipeline with visualizations
- **experiments.ipynb**: Experimental results and comparisons
- **test.ipynb**: Testing and validation scripts

## 📊 Dataset

This project is designed for the **BCI AUT P300** dataset, which contains:
- 15 subjects
- 7 sessions per subject
- 8 EEG channels (C3, Cz, C4, CPz, P3, Pz, P4, POz)
- Binary classification (Target vs Non-Target)

Expected data structure:
```
data/
├── SBJ01/
│   ├── S01/
│   │   ├── Train/
│   │   │   ├── trainData.mat
│   │   │   └── trainTargets.txt
│   │   └── Test/
│   │       ├── testData.mat
│   │       └── testTargets.txt
│   └── ...
└── ...
```

## 🤝 EEG-GAN Integration

This project includes the EEG-GAN framework for synthetic data generation. See `eeggan_external/` for:
- GAN training scripts
- Pre-trained models
- Tutorials and documentation

For more details, refer to the [EEG-GAN documentation](eeggan_external/README.md).

## 📈 Performance Metrics

The package provides comprehensive evaluation metrics:
- **Accuracy** (overall and per-class)
- **F1-Score** (macro and per-class)
- **Confusion Matrix**
- **Classification Report**
- **Complexity Metrics** (N1, N2, N3, N4, C1, C2, kDN)

## 🔧 Configuration

Key parameters can be configured:
- **Preprocessing**: Frequency bands, resampling rate, epoch duration
- **Training**: Batch size, learning rate, epochs, dropout
- **SMOTE**: Sampling strategy, k-neighbors
- **Complexity**: PCA components, subset ratio
