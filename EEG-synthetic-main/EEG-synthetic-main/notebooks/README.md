# EEG Synthetic Data - Notebooks

This folder contains Jupyter notebooks for EEG data analysis, synthetic data generation, and classification experiments.

## 📚 Notebook Overview

### Main Experiments

#### 1. `ml_models_benchmark.ipynb` ⭐
**Comprehensive ML models benchmark for P300 classification**

**Contents:**
- 8 ML classifiers (SVM RBF/Linear, Random Forest, Decision Tree, XGBoost, KNN, Logistic Regression, LDA)
- 4 augmentation scenarios (Baseline, Copy-Paste, SMOTE, GAN)
- Signal preprocessing pipeline (filtering, baseline correction, normalization)
- PCA dimensionality reduction
- Comprehensive metrics (Accuracy, Balanced Accuracy, AUC-ROC)
- Comparative visualizations and heatmaps

**Use this for:** Benchmarking multiple ML models with different augmentation strategies.

---

#### 2. `experiments.ipynb`
**Experimental notebook for P300 classification**

**Contents:**
- Cross-session and cross-subject validation
- SMOTE data augmentation
- Multiple classifiers comparison (KNN, Logistic Regression, Random Forest, Decision Tree)
- EEGNet deep learning model
- Complexity metrics analysis

**Use this for:** Running comprehensive classification experiments with different configurations.

---

#### 3. `analysis.ipynb`
**EEG-GAN synthetic data generation and analysis**

**Contents:**
- EEG-GAN setup and tutorial
- Latent Autoencoder (LAE) training
- GAN training for synthetic EEG generation
- Synthetic sample generation
- Visual comparison of real vs synthetic data
- Quality evaluation metrics
- Classification experiments with synthetic data

**Use this for:** Generating synthetic EEG data using the EEG-GAN framework.

---

### Evaluation Notebooks

#### 4. `similarity_metrics_clean.ipynb`
**Comprehensive similarity metrics analysis**

**Contents:**
- Statistical similarity measures
- Frequency domain analysis
- Time domain comparisons
- Quality metrics for synthetic data

**Use this for:** Evaluating the quality and similarity of synthetic EEG data.

---

#### 5. `xdawn_evaluation_clean.ipynb`
**Xdawn-based EEG quality evaluation**

**Contents:**
- Xdawn spatial filtering
- Signal-to-noise ratio analysis
- P300 component extraction
- Quality assessment

**Use this for:** Advanced signal processing and quality evaluation using Xdawn.

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install -r ../requirements.txt

# For EEG-GAN (analysis.ipynb)
pip install eeggan
```

### Running Experiments

1. **ML Models Benchmark (RECOMMENDED):**
   ```python
   # Open ml_models_benchmark.ipynb
   # Run all cells to benchmark 8 ML models with 4 augmentation strategies
   # Results saved to results/ml_benchmark/
   ```

2. **Basic Classification Experiment:**
   ```python
   # Open experiments.ipynb
   # Run all cells to perform cross-session P300 classification
   ```

3. **Generate Synthetic Data:**
   ```python
   # Open analysis.ipynb
   # Follow sections 3-5 to train and generate synthetic EEG
   ```

4. **Evaluate Synthetic Quality:**
   ```python
   # Open similarity_metrics_clean.ipynb
   # Load real and synthetic data
   # Run evaluation metrics
   ```

---

## 📊 Data Structure

All notebooks expect data in the following structure:

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
│   └── S02/
│       └── ...
└── SBJ02/
    └── ...
```

**Data Format:**
- `trainData.mat` / `testData.mat`: EEG signals (channels × timepoints × trials)
- `trainTargets.txt` / `testTargets.txt`: Labels (0=Nontarget, 1=P300)

---

## 🔧 Configuration

### Key Parameters

**In `experiments.ipynb`:**
```python
subjects = [3]          # Subject IDs to use
sessions = [3]          # Session IDs to use
modes = ['Train']       # 'Train' or 'Test'
```

**In `analysis.ipynb`:**
```python
# LAE training parameters
latent_dim = 100        # Latent space dimension
n_epochs = 1000         # Training epochs
batch_size = 32         # Batch size

# GAN training parameters
n_epochs_gan = 1000     # GAN training epochs
```

---

## 📈 Expected Outputs

### ML Benchmark Results (`ml_models_benchmark.ipynb`)
- Complete results table (all models × all scenarios)
- Pivot tables for easy comparison
- Bar plots comparing Balanced Accuracy and AUC-ROC
- Heatmaps showing performance across scenarios
- Best model identification per scenario
- Saved to: `results/ml_benchmark/`

### Classification Results (`experiments.ipynb`)
- Accuracy, Balanced Accuracy, AUC-ROC
- Confusion matrices
- Classification reports
- ROC curves

### Synthetic Data Quality (`similarity_metrics_clean.ipynb`, `xdawn_evaluation_clean.ipynb`)
- Statistical similarity metrics
- Frequency spectrum comparisons
- Visual plots (real vs synthetic)
- Complexity metrics

---

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'DataLoader'**
   - Ensure you're running from the project root
   - Add project root to Python path:
     ```python
     import sys
     sys.path.append('..')
     ```

2. **File not found errors**
   - Check that data folder structure matches expected format
   - Verify file paths in configuration

3. **Memory errors**
   - Reduce batch size
   - Process fewer subjects/sessions at once
   - Use data generators instead of loading all data at once

---

## 📝 Notes

- **experiments.ipynb** and **analysis.ipynb** contain executed outputs for reference
- **similarity_metrics_clean.ipynb** and **xdawn_evaluation_clean.ipynb** are clean templates
- Always restart kernel when switching between notebooks to avoid variable conflicts

---

## 🤝 Contributing

When adding new notebooks:
1. Use clear markdown sections (## Section Title)
2. Add detailed comments to code cells
3. Include docstrings for custom functions
4. Document expected inputs/outputs
5. Add to this README

---

## 📚 References

- **EEG-GAN**: Hartmann et al. (2018) - "EEG-GAN: Generative adversarial networks for electroencephalographic (EEG) brain signals"
- **EEGNet**: Lawhern et al. (2018) - "EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces"
- **SMOTE**: Chawla et al. (2002) - "SMOTE: Synthetic Minority Over-sampling Technique"

---

Last updated: March 2026
