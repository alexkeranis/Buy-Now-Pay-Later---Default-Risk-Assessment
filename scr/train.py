# src/train.py
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
sys.path.append(os.path.dirname(__file__))
from utils import *

# --- 1. Load data ---
data = load_processed_data()
X_train, X_test = data['X_train'], data['X_test']
cat_train, cat_test = data['cat_train'], data['cat_test']
y_train = encode_target(data['y_train'])
y_test = encode_target(data['y_test'])
cat_cardinalities = data['cat_cardinalities']
target_names = ['Low','Medium','High']

# --- 2. XGBoost Baseline ---
print("\n" + "="*60)
print("XGBoost Baseline")
print("="*60)
xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
y_pred_xgb = xgb_model.predict(X_test)
xgb_report = classification_report(y_test, y_pred_xgb, target_names=target_names, digits=4, output_dict=True)
xgb_f1 = xgb_report['macro avg']['f1-score']
print(f"XGBoost Macro F1: {xgb_f1:.4f}")

# --- 3. PyTorch Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

batch_size = 256
train_ds = TabularDataset(X_train, cat_train, y_train)
test_ds  = TabularDataset(X_test,  cat_test,  y_test)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# Model
model = EmbeddingNet(num_numeric=X_train.shape[1], cat_cardinalities=cat_cardinalities)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Class weights for CrossEntropy
class_counts = np.bincount(y_train)
class_weights = torch.tensor([1.0/c for c in class_counts], dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * 3

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# --- 4. Train ---
print("\n" + "="*60)
print("PyTorch Training")
print("="*60)
model, train_loss, val_loss, val_f1 = train_model(
    model, train_loader, test_loader, criterion, optimizer, scheduler,
    device, epochs=200, patience=15
)

# Save best model
os.makedirs('../models', exist_ok=True)
torch.save(model.state_dict(), '../models/best_pytorch_model.pth')
print("Model saved to models/best_pytorch_model.pth")

# --- 5. Final evaluation ---
print("\n" + "="*60)
print("PyTorch Final Evaluation")
print("="*60)
pytorch_report, pytorch_f1, preds, targets = evaluate_model(model, test_loader, device, target_names)
print(classification_report(targets, preds, target_names=target_names, digits=4))

# --- 6. Comparison ---
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"XGBoost  Macro F1: {xgb_f1:.4f}")
print(f"PyTorch  Macro F1: {pytorch_f1:.4f}")
print(f"Improvement:       {pytorch_f1 - xgb_f1:+.4f}")