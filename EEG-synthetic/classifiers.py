from sklearn.metrics import confusion_matrix, classification_report
from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM
from sklearn.pipeline import Pipeline
from Oversampling import apply_smote_3d
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from itertools import cycle

import os
import mne
import math
import copy
import random
import scipy.io
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, random_split

# Scikit-Learn
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

def run_p300_classification(X_train, y_train, X_test, y_test, use_smote=True):
    """
    Gestisce il flusso di classificazione:
    - (Opzionale) Bilanciamento con SMOTE
    - Addestramento Pipeline Riemanniana
    - Valutazione
    """
    print(f"\n--- Avvio Classificazione (SMOTE: {'ON' if use_smote else 'OFF'}) ---")

    # FASE 1: Gestione Bilanciamento
    if use_smote:
        X_train_proc, y_train_proc = apply_smote_3d(X_train, y_train)
    else:
        X_train_proc, y_train_proc = X_train, y_train
        print(f"Utilizzo dati originali: {X_train_proc.shape[0]} epoche")

    # FASE 2: Definizione Modello
    clf = Pipeline([
        ('xdawn', XdawnCovariances(nfilter=4, estimator='oas')),
        ('mdm', MDM(metric='riemann'))
    ])

    # FASE 3: Addestramento
    clf.fit(X_train_proc, y_train_proc)

    # FASE 4: Predizione e Valutazione
    y_pred = clf.predict(X_test)

    # Calcolo metriche specifiche
    cm = confusion_matrix(y_test, y_pred)
    acc_non_target = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    acc_target = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0

    print("-" * 30)
    print(f"Accuracy Non-Target: {acc_non_target:.2%}")
    print(f"Accuracy Target:     {acc_target:.2%}")
    print("-" * 30)
    print(classification_report(y_test, y_pred, target_names=['Non-Target', 'Target']))

    return clf, y_pred

def classify_eeg(X_train, y_train, X_test, y_test, clf=LogisticRegression(max_iter=1000, class_weight=None), use_smote=True):
    # 1. GESTIONE SMOTE
    if use_smote:
        X_processed, y_train_final = apply_smote_3d(X_train, y_train)
        status_smote = "ON"
        status_text = "Bilanciamento completato"
    else:
        X_processed, y_train_final = X_train, y_train
        status_smote = "OFF"
        status_text = "Utilizzo dati originali"

    # 2. GESTIONE FORMATO (Flattening se non è una Pipeline)
    if not hasattr(clf, 'steps'):
        X_train_final = X_processed.reshape(X_processed.shape[0], -1)
        X_test_final = X_test.reshape(X_test.shape[0], -1)
    else:
        X_train_final = X_processed
        X_test_final = X_test

    # 3. TRAINING
    clf.fit(X_train_final, y_train_final)

    # 4. PREDIZIONE
    y_pred_test = clf.predict(X_test_final)

    # 5. CALCOLO METRICHE PER IL PRINT
    cm = confusion_matrix(y_test, y_pred_test)
    # Accuracy per classe (Recall)
    acc_non_target = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    acc_target = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0

    # --- PRINT DEL REPORT PERSONALIZZATO ---
    print(f"--- Avvio Classificazione (SMOTE: {status_smote}) ---")
    print(f"{status_text}: {X_train_final.shape[0]} epoche")
    print("-" * 30)
    print(f"Accuracy Non-Target: {acc_non_target:.2%}")
    print(f"Accuracy Target:     {acc_target:.2%}")
    print("-" * 30)
    print(classification_report(y_test, y_pred_test, target_names=['Non-Target', 'Target']))

    # 6. RITORNO DEL DATAFRAME (come nelle versioni precedenti)
    classes = np.unique(y_train)
    y_pred_train = clf.predict(X_train_final)
    f1_train = f1_score(y_train_final, y_pred_train, average=None)
    f1_test = f1_score(y_test, y_pred_test, average=None)

    df_result = pd.DataFrame({
        'Class': ["Class " + str(int(x)) for x in classes],
        'Train_F1': f1_train,
        'Test_F1': f1_test
    })

    return df_result

class EEGNetModel(nn.Module): # EEGNET-8,2
    def __init__(self, chans=8, classes=2, time_points=64, temp_kernel=16,
                 f1=8, f2=16, d=2, pk1=4, pk2=4, dropout_rate=0.5, max_norm1=1, max_norm2=0.25):
        super(EEGNetModel, self).__init__()
        # Calculating FC input features
        linear_size = (time_points//(pk1*pk2))*f2

        # Temporal Filters
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, temp_kernel), padding='same', bias=False),
            nn.BatchNorm2d(f1),
        )
        # Spatial Filters
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False), # Depthwise Conv
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1)),
            nn.Dropout(dropout_rate)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16),  groups=f2, bias=False, padding='same'), # Separable Conv
            nn.Conv2d(f2, f2, kernel_size=1, bias=False), # Pointwise Conv
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2)),
            nn.Dropout(dropout_rate)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(linear_size, classes)

        # Apply max_norm constraint to the depthwise layer in block2
        self._apply_max_norm(self.block2[0], max_norm1)

        # Apply max_norm constraint to the linear layer
        self._apply_max_norm(self.fc, max_norm2)

    def _apply_max_norm(self, layer, max_norm):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    

class TrainModel():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_model(self, model, real_0_ds, real_1_ds, fake_1_ds=None, 
                    learning_rate=0.001, batch_size=64, epochs=50, 
                    fake_ratio_in_c1=0.0):
        """
        Gestisce 3 scenari:
        1. Sbilanciato: fake_1_ds=None, fake_ratio=0 -> Batch bilanciato con oversampling reale
        2. SMOTE:       Passi il dataset SMOTE come real_1_ds, fake_1_ds=None -> Trattato come reale
        3. GAN:         Passi tutti e tre, regoli fake_ratio per controllare il poisoning
        """
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 1. Calcolo dinamico delle proporzioni del batch
        n_c1_total = batch_size // 2
        n_c0 = batch_size - n_c1_total
        
        # Determiniamo se usare dati fake (GAN)
        use_fake = (fake_1_ds is not None and len(fake_1_ds) > 0 and fake_ratio_in_c1 > 0)
        
        if not use_fake:
            n_fake_1 = 0
            n_real_1 = n_c1_total
            print(f"MODE: Batch Bilanciato Reale (Oversampling C1 se necessario)")
        else:
            n_fake_1 = int(n_c1_total * fake_ratio_in_c1)
            # Assicuriamoci che n_fake_1 non sia 0 se use_fake è True
            n_fake_1 = max(1, n_fake_1) 
            n_real_1 = n_c1_total - n_fake_1
            print(f"MODE: Data Augmentation GAN (Ratio Fake in C1: {fake_ratio_in_c1})")

        print(f"Composizione Batch: C0={n_c0}, C1_Real={n_real_1}, C1_Fake={n_fake_1}")

        # 2. Setup Dataloaders
        # loader_0 guida l'epoca (solitamente la classe più numerosa)
        loader_0 = DataLoader(real_0_ds, batch_size=n_c0, shuffle=True, drop_last=True)
        
        # Usiamo cycle per le classi minoritarie/sintetiche
        loader_1_real = cycle(DataLoader(real_1_ds, batch_size=n_real_1, shuffle=True, drop_last=True))
        
        if use_fake:
            loader_1_fake = cycle(DataLoader(fake_1_ds, batch_size=n_fake_1, shuffle=True, drop_last=True))

        highest_train_accuracy = 0.0

        for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            
            for batch_0, label_0 in loader_0:
                batch_1r, label_1r = next(loader_1_real)
                
                # Unione batch
                if use_fake:
                    batch_1f, label_1f = next(loader_1_fake)
                    inputs = torch.cat([batch_0, batch_1r, batch_1f], dim=0)
                    labels = torch.cat([label_0, label_1r, label_1f], dim=0)
                else:
                    inputs = torch.cat([batch_0, batch_1r], dim=0)
                    labels = torch.cat([label_0, label_1r], dim=0)

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Shuffle interno al batch
                perm = torch.randperm(inputs.size(0))
                inputs, labels = inputs[perm], labels[perm]

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_acc = correct / total
            if epoch_acc > highest_train_accuracy:
                highest_train_accuracy = epoch_acc
            

            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/total:.4f}, Accuracy: {(epoch_acc*100):.2f}%")

        torch.save(model.state_dict(), 'eegnet_model.pth')
        return model
class EvalModel():
    def __init__(self, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def test_model(self, test_dataset):
        self.model.eval()
        correct = 0
        total = 0
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = (correct / total) * 100
        print("/------------------------------/")
        print(f"Test Accuracy: {accuracy:.2f}%")
        print("/------------------------------/")
        return accuracy

    def plot_confusion_matrix(self, test_dataset, classes):
        self.model.eval()
        y_pred = []
        y_true = []
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.append(predicted.item())
                y_true.append(labels.item())

        cf_matrix = confusion_matrix(y_true, y_pred)
        cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]

        df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)

        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='.2f')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix_model.png')
        plt.show()