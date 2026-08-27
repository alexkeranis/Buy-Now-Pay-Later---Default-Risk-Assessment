# src/utils.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import pickle

# ------------------- Data loading -------------------
def load_processed_data(data_path='data/processed/final_data.npz'):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cache not found: {data_path}. Run data_prep.py first.")
    data = np.load(data_path, allow_pickle=True)
    return {
        'X_train': data['X_train_scaled'],
        'X_test': data['X_test_scaled'],
        'y_train': data['y_train'],
        'y_test': data['y_test'],
        'cat_train': data['cat_train'],
        'cat_test': data['cat_test'],
        'cat_cardinalities': data['cat_cardinalities'],
        'cat_cols': data['cat_cols'].tolist(),
        'num_cols': data['num_cols'].tolist(),
        'target_col': str(data['target_col'])
    }

def encode_target(y, mapping={'Low':0, 'Medium':1, 'High':2}):
    return np.array([mapping[x] for x in y])

# ------------------- PyTorch Dataset -------------------
class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

# ------------------- Model definition -------------------
class EmbeddingNet(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, emb_dim=8, hidden_dims=[128,64]):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cat_cardinalities])
        total_emb_dim = len(cat_cardinalities) * emb_dim
        total_input_dim = num_numeric + total_emb_dim
        layers = []
        prev_dim = total_input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.4)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 3))
        self.net = nn.Sequential(*layers)
    def forward(self, x_num, x_cat):
        cat_embs = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        x = torch.cat([x_num, cat_embs], dim=1)
        return self.net(x)

# ------------------- Training loop -------------------
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler,
                device, epochs=200, patience=15):
    model.to(device)
    best_val_f1 = 0.0
    patience_counter = 0
    best_state = None
    train_losses, val_losses, val_f1s = [], [], []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_num, X_cat, y in train_loader:
            X_num, X_cat, y = X_num.to(device), X_cat.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_num, X_cat), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_num.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        preds, targets = [], []
        with torch.no_grad():
            for X_num, X_cat, y in test_loader:
                X_num, X_cat, y = X_num.to(device), X_cat.to(device), y.to(device)
                outputs = model(X_num, X_cat)
                loss = criterion(outputs, y)
                val_loss += loss.item() * X_num.size(0)
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                targets.extend(y.cpu().numpy())
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        val_f1 = f1_score(targets, preds, average='macro')
        val_f1s.append(val_f1)

        scheduler.step(val_loss)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch+1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | LR: {lr:.2e}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    print(f"Best validation Macro F1: {best_val_f1:.4f}")
    return model, train_losses, val_losses, val_f1s

# ------------------- Evaluation helpers -------------------
def evaluate_model(model, test_loader, device, target_names=['Low','Medium','High']):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_num, X_cat, y in test_loader:
            X_num, X_cat = X_num.to(device), X_cat.to(device)
            outputs = model(X_num, X_cat)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(y.numpy())
    report = classification_report(targets, preds, target_names=target_names, digits=4, output_dict=True)
    macro_f1 = f1_score(targets, preds, average='macro')
    return report, macro_f1, preds, targets

# ------------------- XGBoost training -------------------
def train_xgboost(X_train, y_train, X_test, y_test):
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    model = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3,
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, eval_metric='mlogloss'
    )
    model.fit(X_train, y_train, sample_weight=sample_weights,
              eval_set=[(X_test, y_test)], verbose=False)
    return model